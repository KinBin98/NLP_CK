import argparse
import csv
import os
import re
import torch
from datasets import load_from_disk
from transformers import set_seed
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

from config import DEFAULT_MODEL, SEED, TASKS


def load_model(model_name, load_in_4bit=True):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")
    tokenizer.padding_side = 'left'
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def load_checkpoint_model(checkpoint_path, model_name):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")
    tokenizer.padding_side = 'left'
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
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
    
    # Điều chỉnh max_new_tokens theo task_type
    if task_type == "qa":
        max_new_tokens = 50  # QA cần nhiều token hơn
    elif task_type == "regression":
        max_new_tokens = 20
    else:
        max_new_tokens = 5
    
    inputs = tokenizer(
        formatted_prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=1024,
        padding_side='left'
    ).to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
    )
    
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    cleaned = []
    for d in decoded:
        if "assistant" in d.lower():
            d = d.lower().split("assistant")[-1].strip()
        
        if task_type == "classification":
            # Tìm số phù hợp với số class
            match = re.search(r'\b[0-3]\b', d)
            cleaned.append(match.group(0) if match else "")
        
        elif task_type == "regression":
            # Tìm float
            match = re.search(r'\b\d+(?:\.\d+)?\b', d)
            cleaned.append(match.group(0) if match else "")
        
        elif task_type == "qa":
            # Trả về toàn bộ câu trả lời
            cleaned.append(d if d else "")
    
    return cleaned

def extract_answer(generated, _):
    if not generated:
        return ""
    # Lấy số hoặc float đầu tiên
    match = re.search(r'\b\d+(?:\.\d+)?\b', generated)
    return match.group(0) if match else generated.split()[0] if generated else ""


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


def _get_task_split(dataset, task_name, split_name):
    if split_name not in dataset:
        return None
    return dataset[split_name].filter(lambda ex: ex["task"] == task_name)


def run_task(task, dataset, method, model, tokenizer, split_name):
    split = _get_task_split(dataset, task.name, split_name)
    if split is None:
        return [], [], []

    prompts = split["prompt"]
    y_true = split["response"]
    y_pred = []
    batch_size = 4

    for i in range(0, len(prompts), batch_size):
        decoded = predict_batch(model, tokenizer, prompts[i:i+batch_size])
        for gen in decoded:
            ans = extract_answer(gen, "")
            if task.task_type == "classification":
                y_pred.append(_label_to_id(task, ans))
            elif task.task_type == "qa":
                y_pred.append(ans)
            else:
                match = re.search(r"[-+]?\d*\.?\d+", ans)
                y_pred.append(float(match.group(0)) if match else None)
    return y_true, y_pred, prompts


def main(args):
    set_seed(SEED)
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    print(f"Loading dataset from {args.dataset_dir}")
    dataset = load_from_disk(args.dataset_dir)

    print(f"\n🚀 Loading model: {args.model_name}")
    model, tokenizer = load_model(args.model_name, load_in_4bit=True)

    rows = []
    for task in TASKS:
        if args.task and task.name != args.task:
            continue
        print(f"\n📊 Processing task: {task.name}")
        y_true, y_pred, prompts = run_task(task, dataset, args.method, model, tokenizer, args.split)
        
        for prompt, label, pred in zip(prompts, y_true, y_pred):
            rows.append({
                "task": task.name, "split": args.split,
                "label": str(label), "prediction": str(pred) if pred is not None else "",
                "method": args.method, "prompt": prompt,
            })

    with open(args.output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "split", "label", "prediction", "method", "prompt"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["checkpoint", "baseline"], required=True)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--dataset_dir", type=str, default="data_processed")
    parser.add_argument("--output_file", type=str, default=os.path.join("outputs", "predictions", "predictions.csv"))
    args = parser.parse_args()
    main(args)