# test.py - FIXED để giống hệt baseline (chỉ thêm PeftModel đúng cách)
import argparse
import csv
import os
import re
import torch
from collections import defaultdict
from datasets import load_from_disk
from transformers import set_seed
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from peft import PeftModel

from config import DEFAULT_MODEL, SEED, TASKS


def load_model(model_name, load_in_4bit=True):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")
    tokenizer.padding_side = 'left'  # ← GIỐNG BASELINE
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def load_finetuned_model(checkpoint_path, model_name):
    """Load model đã fine-tune - GIỐNG CÁCH BASELINE LOAD"""
    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")
    tokenizer.padding_side = 'left'  # ← GIỐNG BASELINE
    
    print(f"📁 Loading LoRA weights from: {checkpoint_path}")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    
    # Merge LoRA weights để inference nhanh hơn (optional)
    # model = model.merge_and_unload()
    
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def predict_batch(model, tokenizer, prompts, task_type="classification", max_new_tokens=20):
    formatted_prompts = []
    for p in prompts:
        messages = [{"role": "user", "content": p.strip()}]
        formatted = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        formatted_prompts.append(formatted)
    
    if task_type == "qa":
        max_new_tokens = 50
        temperature = 0.5
        do_sample = True
    elif task_type == "regression":
        max_new_tokens = 20
        temperature = 0.1
        do_sample = False
    else:
        max_new_tokens = 5
        temperature = 0.1
        do_sample = False
    
    inputs = tokenizer(
        formatted_prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=1024,
        padding_side='left'  # ← GIỐNG BASELINE
    ).to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        top_p=0.9 if do_sample else None,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
    )
    
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    cleaned = []
    for d in decoded:
        if "assistant" in d.lower():
            d = d.lower().split("assistant")[-1].strip()
        
        if task_type == "classification":
            match = re.search(r'\b[0-3]\b', d)
            cleaned.append(match.group(0) if match else "")
        elif task_type == "regression":
            match = re.search(r'\b\d+(?:\.\d+)?\b', d)
            cleaned.append(match.group(0) if match else "")
        elif task_type == "qa":
            cleaned.append(d if d else "")
    
    return cleaned


def normalize_label(text):
    if text is None:
        return ""
    return re.sub(r"[^a-zA-Z0-9_ ]", "", str(text)).strip().lower().replace(" ", "_")


def _label_to_id(task, label_text):
    if task.task_type != "classification" or label_text is None:
        return label_text
    try:
        return int(label_text)
    except (ValueError, TypeError):
        reverse_map = {normalize_label(v): k for k, v in task.label_map.items()}
        return reverse_map.get(normalize_label(str(label_text)), None)


def main(args):
    set_seed(SEED)
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    print(f"Loading dataset from {args.dataset_dir}")
    dataset = load_from_disk(args.dataset_dir)

    print(f"\n🚀 Loading model: {args.model_name}")
    
    # Load đúng model dựa trên method
    if args.method == "checkpoint" and args.checkpoint:
        if not os.path.exists(args.checkpoint):
            print(f"⚠️ Checkpoint not found: {args.checkpoint}")
            print(f"   Falling back to baseline model")
            model, tokenizer = load_model(args.model_name, load_in_4bit=True)
        else:
            model, tokenizer = load_finetuned_model(args.checkpoint, args.model_name)
    else:
        model, tokenizer = load_model(args.model_name, load_in_4bit=True)

    if args.split not in dataset:
        print(f"⚠️ Split '{args.split}' not found")
        return
    
    full_split = dataset[args.split]
    task_map = {task.name: task for task in TASKS}
    
    print(f"\n📊 Processing {len(full_split)} samples from {args.split} split")
    
    if args.task:
        full_split = full_split.filter(lambda ex: ex["task"] == args.task)
        print(f"  Filtered to {len(full_split)} samples for task '{args.task}'")
    
    rows = []
    batch_size = 6
    
    for i in range(0, len(full_split), batch_size):
        batch_indices = list(range(i, min(i + batch_size, len(full_split))))
        
        batch_prompts = []
        batch_tasks = []
        batch_labels = []
        batch_task_types = []
        batch_task_objs = []
        
        for idx in batch_indices:
            example = full_split[idx]
            task_name = example["task"]
            task = task_map.get(task_name)
            
            if task is None:
                continue
            
            batch_prompts.append(example["prompt"])
            batch_tasks.append(task_name)
            batch_labels.append(example["response"])
            batch_task_types.append(task.task_type)
            batch_task_objs.append(task)
        
        if not batch_prompts:
            continue
        
        by_type = defaultdict(list)
        for j, task_type in enumerate(batch_task_types):
            by_type[task_type].append(j)
        
        all_preds = [None] * len(batch_prompts)
        
        for task_type, indices in by_type.items():
            type_prompts = [batch_prompts[j] for j in indices]
            decoded = predict_batch(model, tokenizer, type_prompts, task_type=task_type)
            
            for orig_idx, pred in zip(indices, decoded):
                task_obj = batch_task_objs[orig_idx]
                
                if task_type == "classification":
                    all_preds[orig_idx] = _label_to_id(task_obj, pred)
                elif task_type == "qa":
                    all_preds[orig_idx] = pred
                else:
                    try:
                        all_preds[orig_idx] = round(float(pred), 1) if pred else None
                    except:
                        all_preds[orig_idx] = None
        
        for j in range(len(batch_prompts)):
            rows.append({
                "task": batch_tasks[j],
                "split": args.split,
                "label": str(batch_labels[j]),
                "prediction": str(all_preds[j]) if all_preds[j] is not None else "",
                "method": args.method,
                "prompt": batch_prompts[j],
            })
        
        if (i + batch_size) % 100 == 0 or i + batch_size >= len(full_split):
            print(f"  Progress: {min(i+batch_size, len(full_split))}/{len(full_split)} samples")
    
    with open(args.output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "split", "label", "prediction", "method", "prompt"])
        writer.writeheader()
        writer.writerows(rows)
    
    task_counts = defaultdict(int)
    for row in rows:
        task_counts[row["task"]] += 1
    
    print(f"\n✅ Saved {len(rows)} predictions to {args.output_file}")
    print(f"\n📊 Task distribution:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["checkpoint", "baseline"], required=True)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--dataset_dir", type=str, default="data/merged")
    parser.add_argument("--output_file", type=str, default=os.path.join("outputs", "predictions", "predictions.csv"))
    args = parser.parse_args()
    main(args)