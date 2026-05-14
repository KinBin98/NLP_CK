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

from config import DEFAULT_MODEL, SEED, TASKS, MAX_SEQ_LENGTH


def load_model(model_name, load_in_4bit=True):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")
    tokenizer.padding_side = 'left'
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def load_finetuned_model(checkpoint_path, model_name):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")
    tokenizer.padding_side = 'left'

    print(f"Loading LoRA weights from: {checkpoint_path}")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def predict_batch(model, tokenizer, prompts, task=None, task_type="classification"):
    formatted = []
    for p in prompts:
        messages = [{"role": "user", "content": p.strip()}]
        formatted.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    if task_type == "qa":
        max_new_tokens, temperature, do_sample = 30, 0.5, True
    elif task_type == "regression":
        max_new_tokens, temperature, do_sample = 5, 0.1, False
    else:
        max_new_tokens, temperature, do_sample = 5, 0.1, False

    inputs = tokenizer(formatted, return_tensors="pt", padding=True, truncation=True, max_length=1024, padding_side='left')
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

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
            # Dynamic pattern based on task labels
            if task and hasattr(task, 'label_map') and task.label_map:
                valid_labels = list(task.label_map.keys())
                pattern = r'\b(?:' + '|'.join(map(str, valid_labels)) + r')\b'
            else:
                pattern = r'\b[0-3]\b'  # fallback
            match = re.search(pattern, d)
            cleaned.append(match.group(0) if match else "")
        elif task_type == "regression":
            match = re.search(r'\b\d+(?:\.\d+)?\b', d)
            cleaned.append(match.group(0) if match else "")
        else:
            cleaned.append(d)
    return cleaned


def label_to_id(task, label_text):
    if task.task_type != "classification" or label_text is None:
        return label_text
    try:
        return int(label_text)
    except (ValueError, TypeError):
        if task.label_map:
            reverse_map = {str(v).lower().replace(" ", "_"): k for k, v in task.label_map.items()}
            key = str(label_text).strip().lower().replace(" ", "_")
            return reverse_map.get(key, None)
        return None


def main(args):
    set_seed(SEED)
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    print(f"Loading dataset from {args.dataset_dir}")
    dataset = load_from_disk(args.dataset_dir)

    print(f"Loading model: {args.model_name}")
    if args.method == "checkpoint" and args.checkpoint:
        model, tokenizer = load_finetuned_model(args.checkpoint, args.model_name)
    else:
        model, tokenizer = load_model(args.model_name, load_in_4bit=True)

    if args.split not in dataset:
        print(f"Split '{args.split}' not found")
        return

    full_split = dataset[args.split]
    task_map = {task.name: task for task in TASKS}

    print(f"Processing {len(full_split)} samples from {args.split} split")
    if args.task:
        full_split = full_split.filter(lambda ex: ex["task"] == args.task)
        print(f"Filtered to {len(full_split)} samples for task '{args.task}'")

    rows = []
    batch_size = 6

    for i in range(0, len(full_split), batch_size):
        batch = full_split[i:i+batch_size]

        prompts, tasks, labels, task_types, task_objs = [], [], [], [], []
        for item in batch:
            task = task_map.get(item["task"])
            if not task:
                continue
            prompts.append(item["prompt"])
            tasks.append(item["task"])
            labels.append(item["response"])
            task_types.append(task.task_type)
            task_objs.append(task)

        if not prompts:
            continue

        by_type = defaultdict(list)
        for j, tt in enumerate(task_types):
            by_type[tt].append(j)

        preds = [None] * len(prompts)

        for tt, indices in by_type.items():
            type_prompts = [prompts[j] for j in indices]
            # Pass task object for dynamic pattern
            sample_task = task_objs[indices[0]] if indices else None
            decoded = predict_batch(model, tokenizer, type_prompts, task=sample_task, task_type=tt)

            for orig_idx, pred in zip(indices, decoded):
                task_obj = task_objs[orig_idx]
                if tt == "classification":
                    preds[orig_idx] = label_to_id(task_obj, pred)
                elif tt == "qa":
                    preds[orig_idx] = pred
                else:
                    try:
                        preds[orig_idx] = round(float(pred), 1) if pred else None
                    except:
                        preds[orig_idx] = None

        for j in range(len(prompts)):
            rows.append({
                "task": tasks[j],
                "split": args.split,
                "label": str(labels[j]),
                "prediction": str(preds[j]) if preds[j] is not None else "",
                "method": args.method,
                "prompt": prompts[j],
            })

        if (i + batch_size) % 100 == 0 or i + batch_size >= len(full_split):
            print(f"Progress: {min(i+batch_size, len(full_split))}/{len(full_split)}")

    with open(args.output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "split", "label", "prediction", "method", "prompt"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} predictions to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["checkpoint", "baseline"], required=True)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--dataset_dir", type=str, default="data/merged")
    parser.add_argument("--output_file", type=str, default=os.path.join("outputs", "predictions", "predictions.csv"))
    args = parser.parse_args()
    main(args)