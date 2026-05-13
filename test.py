import argparse
import csv
import json
import os
import re
import torch
from datasets import load_from_disk
from transformers import set_seed
from unsloth import FastLanguageModel

from config import DEFAULT_MODEL, SEED, TASKS


def load_model(model_name, load_in_4bit=True):
    """Load model pretrain (chưa fine-tune)"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def load_checkpoint_model(checkpoint_path, model_name):
    """Load model đã fine-tune từ checkpoint"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def predict_batch(model, tokenizer, prompts, max_new_tokens=50):  # Tăng lên 50
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # Xử lý output rỗng
    for i, d in enumerate(decoded):
        if not d or len(d.strip()) == 0:
            print(f"  ⚠️ Empty output for sample {i+1}, using fallback")
            decoded[i] = prompts[i] + " unknown"
    
    return decoded


def extract_answer(generated, prompt):
    if not generated or len(generated.strip()) == 0:
        return ""
    
    if generated.startswith(prompt):
        generated = generated[len(prompt):]
    
    if "Answer:" in generated:
        generated = generated.split("Answer:")[-1]
    
    lines = generated.strip().splitlines()
    if len(lines) == 0:
        return ""
    
    answer = lines[0].strip()
    return answer


def normalize_label(text):
    if text is None:
        return ""
    cleaned = re.sub(r"[^a-zA-Z0-9_ ]", "", str(text))
    return cleaned.strip().lower().replace(" ", "_")


def _normalize_classification_output(text):
    if text is None:
        return ""
    cleaned = re.sub(r"[^a-zA-Z0-9_ ]", " ", str(text))
    cleaned = cleaned.strip().lower()
    return cleaned.split()[0] if cleaned else ""


def _extract_first_float(text):
    if text is None:
        return None
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(text))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _label_to_id(task, label_text):
    if task.task_type != "classification":
        return label_text
    if label_text is None:
        return None
    reverse_map = {normalize_label(v): k for k, v in task.label_map.items()}
    return reverse_map.get(normalize_label(str(label_text)), None)


def _label_to_str(label):
    if isinstance(label, (list, dict)):
        return json.dumps(label, ensure_ascii=True)
    return "" if label is None else str(label)


def _prediction_to_str(pred):
    return "" if pred is None else str(pred)


def _get_task_split(dataset, task_name, split_name):
    if split_name not in dataset:
        return None
    split = dataset[split_name]
    return split.filter(lambda ex: ex["task"] == task_name)


def run_task(task, dataset, method, model, tokenizer, split_name):
    split = _get_task_split(dataset, task.name, split_name)
    if split is None:
        print(f"Warning: {task.full_name} has no '{split_name}' split. Skipping.")
        return [], [], []

    

    prompts = split["prompt"]
    y_true_text = split["response"]

    # BASELINE = DÙNG LLM PRETRAIN (chưa fine-tune)
    if method == "baseline":
        print(f"  📌 Baseline: using pretrained LLM (no fine-tuning) for {task.name}")
        y_pred = []
        batch_size = 8
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            decoded = predict_batch(model, tokenizer, batch_prompts)
            for prompt, gen in zip(batch_prompts, decoded):
                ans = extract_answer(gen, prompt)
                if task.task_type == "classification":
                    pred = _label_to_id(task, _normalize_classification_output(ans))
                    y_pred.append(pred)
                elif task.task_type == "qa":
                    y_pred.append(ans)
                else:
                    y_pred.append(_extract_first_float(ans))
        return y_true_text, y_pred, prompts

    # CHECKPOINT = DÙNG LLM ĐÃ FINE-TUNE
    elif method == "checkpoint":
        print(f"  📌 Checkpoint: using fine-tuned LLM for {task.name}")
        y_pred = []
        batch_size = 8
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            decoded = predict_batch(model, tokenizer, batch_prompts)
            for prompt, gen in zip(batch_prompts, decoded):
                ans = extract_answer(gen, prompt)
                if task.task_type == "classification":
                    pred = _label_to_id(task, _normalize_classification_output(ans))
                    y_pred.append(pred)
                elif task.task_type == "qa":
                    y_pred.append(ans)
                else:
                    y_pred.append(_extract_first_float(ans))
        return y_true_text, y_pred, prompts

    else:
        raise ValueError(f"Unknown method: {method}")


def main(args):
    set_seed(SEED)
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    dataset = load_from_disk(args.dataset_dir)

    model = None
    tokenizer = None
    
    if args.method == "baseline":
        # Baseline: load model pretrain (chưa fine-tune)
        print(f"\n🚀 Loading PRETRAINED model: {args.model_name}")
        model, tokenizer = load_model(args.model_name, load_in_4bit=True)
        
    elif args.method == "checkpoint":
        # Checkpoint: load model đã fine-tune
        if not args.checkpoint:
            raise ValueError("--checkpoint is required when method=checkpoint")
        print(f"\n🚀 Loading CHECKPOINT model from: {args.checkpoint}")
        model, tokenizer = load_checkpoint_model(args.checkpoint, args.model_name)
    
    else:
        raise ValueError(f"Unknown method: {args.method}")

    rows = []
    for task in TASKS:
        if args.task and task.name != args.task:
            continue

        y_true_text, y_pred, prompts = run_task(
            task,
            dataset,
            args.method,
            model,
            tokenizer,
            args.split,
        )
        if not y_true_text:
            continue

        for prompt, label_text, pred in zip(prompts, y_true_text, y_pred):
            if task.task_type == "classification":
                label = _label_to_id(task, label_text)
            elif task.task_type == "qa":
                label = [label_text] if label_text else []
            else:
                try:
                    label = float(label_text)
                except ValueError:
                    label = label_text

            rows.append(
                {
                    "task": task.name,
                    "split": args.split,
                    "label": _label_to_str(label),
                    "prediction": _prediction_to_str(pred),
                    "method": args.method,
                    "prompt": prompt if args.include_prompt else "",
                }
            )

    with open(args.output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task", "split", "label", "prediction", "method", "prompt"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Saved predictions to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["checkpoint", "baseline"], required=True,
                        help="baseline: pretrained LLM, checkpoint: fine-tuned LLM")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (required for method=checkpoint)")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL,
                        help="Base model name for pretrained LLM")
    parser.add_argument("--dataset_dir", type=str, default="data_processed")
    parser.add_argument("--include_prompt", action="store_true", 
                        help="Include prompt column in output CSV")
    parser.add_argument("--output_file", type=str, default=os.path.join("outputs", "predictions", "predictions.csv"))
    args = parser.parse_args()

    main(args)