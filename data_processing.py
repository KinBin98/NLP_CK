import argparse
import os
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from config import TASKS


def make_prompt(task, example):
    parts = [f"Task: {task.name.upper()}"]
    for field in task.text_fields:
        parts.append(f"{field.replace('_', ' ').title()}: {example[field]}")
    parts.append("Answer:")
    return "\n".join(parts)


def load_task_dataset(task):
    if os.path.exists(task.dataset):
        print(f"Loading local dataset from {task.dataset}")
        return load_dataset(task.dataset)

    print(f"Loading Hugging Face dataset: {task.dataset}:{task.name}")

    try:
        return load_dataset(task.dataset, task.name)
    except ValueError:
        return load_dataset(task.dataset)


def format_task(task):
    print(f"Processing {task.full_name}...")
    raw = load_task_dataset(task)
    if isinstance(raw, Dataset):
        raw = DatasetDict({"train": raw})

    def _get_nested_value(example, field_path):
        if not field_path:
            return None
        current = example
        for key in field_path.split("."):
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current

    def _map_fn(ex):
        prompt = make_prompt(task, ex)
        label_value = _get_nested_value(ex, task.label_field)
        if label_value in (-1, None):
            # Test splits can be unlabeled in GLUE.
            label_text = ""
        elif task.task_type == "classification":
            label_text = task.label_map[int(label_value)]
        elif task.task_type == "qa":
            if isinstance(label_value, list):
                label_text = label_value[0] if label_value else ""
            elif isinstance(label_value, dict):
                text_list = label_value.get("text") or []
                label_text = text_list[0] if text_list else ""
            else:
                label_text = str(label_value) if label_value is not None else ""
        else:
            label_text = str(float(label_value))
        return {"prompt": prompt, "response": label_text, "task": task.name}

    mapped = DatasetDict()
    for split_name, split_ds in raw.items():
        mapped[split_name] = split_ds.map(
            _map_fn,
            remove_columns=split_ds.column_names,
        )
    return mapped


def build_multitask_dataset(
    output_dir="data_processed",
    max_train=10000,
    max_val=2000,
    max_test=3000,
):
    merged = None

    def _limit_split(ds, split_name, max_samples, task_name):
        if split_name not in ds:
            if max_samples is not None:
                print(
                    f"Warning: {task_name} has no '{split_name}' split. Skipping limit."
                )
            return ds

        split_len = len(ds[split_name])
        if split_len < max_samples:
            print(
                f"Warning: {task_name} '{split_name}' has only {split_len:,} samples;"
                f" requested {max_samples:,}."
            )
        ds[split_name] = (
            ds[split_name]
            .shuffle(seed=42)
            .select(range(min(max_samples, split_len)))
        )
        return ds

    for task in TASKS:
        ds = format_task(task)
        ds = _limit_split(ds, "train", max_train, task.full_name)
        ds = _limit_split(ds, "validation", max_val, task.full_name)
        ds = _limit_split(ds, "test", max_test, task.full_name)

        if merged is None:
            merged = ds
        else:
            merged = DatasetDict(
                {
                    split: concatenate_datasets([merged[split], ds[split]])
                    for split in merged
                    if split in ds
                }
            )

    if "train" in merged:
        merged["train"] = merged["train"].shuffle(seed=42)

    merged.save_to_disk(output_dir)
    print(f"Dataset saved to {output_dir}")
    print(f"Total train samples: {len(merged['train']):,}")
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data_processed")
    parser.add_argument("--max_train", type=int, default=10000)
    parser.add_argument("--max_val", type=int, default=2000)
    parser.add_argument("--max_test", type=int, default=3000)
    args = parser.parse_args()

    build_multitask_dataset(
        args.output_dir,
        args.max_train,
        args.max_val,
        args.max_test,
    )
