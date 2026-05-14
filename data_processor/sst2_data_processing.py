import argparse
import os
from datasets import DatasetDict, concatenate_datasets, load_dataset


def make_prompt(example):
    return f"""Task: SST-2 (Sentiment Analysis)

Classify the sentiment of the sentence.

Classification:
- 0 = Negative sentiment
- 1 = Positive sentiment

Sentence: {example.get('sentence', '')}

Answer (0 or 1):"""


def map_example(example):
    label_value = example.get("label", -1)
    response = str(label_value) if label_value in (0, 1) else ""
    return {
        "prompt": make_prompt(example),
        "response": response,
        "task": "sst2",
    }


def _select_from_train(train_raw, used_indices, count):
    if count <= 0:
        return train_raw.select([]), used_indices
    selected = []
    for idx in range(len(train_raw)):
        if idx in used_indices:
            continue
        selected.append(idx)
        used_indices.add(idx)
        if len(selected) >= count:
            break
    return train_raw.select(selected), used_indices


def build_sst2_dataset(
    output_dir="data/sst2",
    max_train=10000,
    max_val=2000,
    max_test=3000,
    seed=42,
):
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading SST-2 dataset...")
    raw = load_dataset("glue", "sst2")

    train_raw = raw["train"].shuffle(seed=seed)
    val_raw = raw.get("validation")

    ds = DatasetDict()
    used_indices = set()

    print(f"  Splitting data...")
    
    if val_raw is not None:
        base_val = val_raw.select(range(min(max_val, len(val_raw))))
    else:
        base_val = train_raw.select([])

    need_val = max_val - len(base_val)
    extra_val, used_indices = _select_from_train(train_raw, used_indices, need_val)
    ds["validation"] = base_val if len(extra_val) == 0 else concatenate_datasets([base_val, extra_val])

    ds["test"], used_indices = _select_from_train(train_raw, used_indices, max_test)
    ds["train"], used_indices = _select_from_train(train_raw, used_indices, max_train)

    print(f"  Converting to instruction format...")
    for split in ds:
        ds[split] = ds[split].map(map_example, remove_columns=ds[split].column_names)

    ds.save_to_disk(output_dir)
    
    print(f"Saved SST-2 dataset to {output_dir}")
    print("Statistics:")
    for split in ds:
        print(f"  {split}: {len(ds[split]):,} samples")
    
    print("Sample check:")
    sample = ds["train"][0]
    print(f"  Task: {sample['task']}")
    print(f"  Response: '{sample['response']}'")
    
    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/sst2")
    parser.add_argument("--max_train", type=int, default=10000)
    parser.add_argument("--max_val", type=int, default=2000)
    parser.add_argument("--max_test", type=int, default=3000)
    args = parser.parse_args()

    build_sst2_dataset(
        args.output_dir,
        args.max_train,
        args.max_val,
        args.max_test,
    )