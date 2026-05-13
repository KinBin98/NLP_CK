import argparse
import csv
import json
import os
import torch
from datasets import load_from_disk
from transformers import set_seed
from unsloth import FastLanguageModel

from config import DEFAULT_MODEL, SEED, TASKS


def load_model(checkpoint_path, model_name):
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


def predict_batch(model, tokenizer, prompts, max_new_tokens=20):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded


def extract_answer(generated, prompt):
    if generated.startswith(prompt):
        generated = generated[len(prompt):]
    answer = generated.strip().splitlines()[0]
    return answer.strip()


def normalize_label(text):
    return text.strip().lower().replace(" ", "_")


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


def run_task(task, dataset, method, model, tokenizer, split_name, max_samples=None):
    split = _get_task_split(dataset, task.name, split_name)
    if split is None:
        print(f"Warning: {task.full_name} has no '{split_name}' split. Skipping.")
        return [], [], []

    if max_samples:
        split = split.select(range(min(max_samples, len(split))))

    prompts = split["prompt"]
    y_true_text = split["response"]

    if method == "baseline":
        if task.task_type == "classification":
            label_ids = [_label_to_id(task, v) for v in y_true_text]
            label_ids = [v for v in label_ids if v is not None]
            majority = max(set(label_ids), key=label_ids.count) if label_ids else None
            y_pred = [majority] * len(y_true_text)
        elif task.task_type == "qa":
            majority = max(set(y_true_text), key=y_true_text.count) if y_true_text else ""
            y_pred = [majority] * len(y_true_text)
        else:
            values = [float(v) for v in y_true_text]
            mean_val = float(sum(values) / len(values)) if values else 0.0
            y_pred = [mean_val] * len(y_true_text)
        return y_true_text, y_pred, prompts

    y_pred = []
    batch_size = 8
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        decoded = predict_batch(model, tokenizer, batch_prompts)
        for prompt, gen in zip(batch_prompts, decoded):
            ans = extract_answer(gen, prompt)
            if task.task_type == "classification":
                pred = _label_to_id(task, ans)
                y_pred.append(pred)
            elif task.task_type == "qa":
                y_pred.append(ans)
            else:
                try:
                    y_pred.append(float(ans))
                except ValueError:
                    y_pred.append(None)
    return y_true_text, y_pred, prompts


def main(args):
    set_seed(SEED)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    dataset = load_from_disk(args.dataset_dir)

    model = None
    tokenizer = None
    if args.method == "checkpoint":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required when method=checkpoint")
        model, tokenizer = load_model(args.checkpoint, args.model_name)

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
            args.max_samples,
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
                }
            )

    with open(args.output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task", "split", "label", "prediction", "method"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved predictions to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["checkpoint", "baseline"], required=True)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--dataset_dir", type=str, default="data_processed")
    parser.add_argument(
        "--output_file",
        type=str,
        default=os.path.join("outputs", "predictions", "predictions.csv"),
    )
    args = parser.parse_args()

    main(args)
