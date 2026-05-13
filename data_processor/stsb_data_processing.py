import argparse
import os
from datasets import DatasetDict, concatenate_datasets, load_dataset


def make_prompt(example):
    return f"""Task: STS-B (Semantic Textual Similarity)

Rate the similarity between two sentences on a scale from 0 to 5.
Output a decimal number rounded to 1 decimal place (e.g., 0.0, 1.5, 2.3, 4.8, 5.0).

Scale:
- 0.0: Completely different meaning
- 1.0: Mostly different
- 2.0: Slightly similar
- 3.0: Moderately similar
- 4.0: Very similar
- 5.0: Completely equivalent meaning

Sentence 1: {example.get('sentence1', '')}
Sentence 2: {example.get('sentence2', '')}

Similarity score (0.0 to 5.0, one decimal):"""


def map_example(example):
    label_value = example.get("label", None)
    # ✅ Giữ nguyên float (STS-B là regression)
    if label_value in (-1, None):
        response = ""
    else:
        response = f"{float(label_value):.1f}"  # Định dạng 1 số thập phân
    return {
        "prompt": make_prompt(example),
        "response": response,
        "task": "stsb",
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


def build_stsb_dataset(output_dir="data/stsb", max_train=10000, max_val=2000, max_test=3000, seed=42):
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📥 Loading STS-B dataset...")
    raw = load_dataset("gokuls/glue_augmented_stsb")

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
    
    print(f"\n✅ Saved STS-B dataset to {output_dir}")
    print(f"📊 Statistics:")
    for split in ds:
        print(f"  {split}: {len(ds[split]):,} samples")
    
    # Kiểm tra mẫu
    print(f"\n📝 Sample check:")
    sample = ds["train"][0]
    print(f"  Task: {sample['task']}")
    print(f"  Response: '{sample['response']}'")
    
    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/stsb")
    parser.add_argument("--max_train", type=int, default=10000)
    parser.add_argument("--max_val", type=int, default=2000)
    parser.add_argument("--max_test", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_stsb_dataset(
        output_dir=args.output_dir,
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=args.max_test,
        seed=args.seed,
    )