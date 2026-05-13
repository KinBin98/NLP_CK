import argparse
import os
from datasets import DatasetDict, concatenate_datasets, load_dataset


def make_prompt(example):
    return "\n".join(
        [
            "Task: STS-B",
            f"Sentence1: {example.get('sentence1', '')}",
            f"Sentence2: {example.get('sentence2', '')}",
            "Answer:",
        ]
    )


def map_example(example):
    label_value = example.get("label", None)
    if label_value in (-1, None):
        label_text = ""
    else:
        label_text = str(float(label_value))
    return {
        "prompt": make_prompt(example),
        "response": label_text,
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


def build_stsb_dataset(output_dir="data_stsb", max_train=10000, max_val=2000, max_test=3000, seed=42):
    output_dir = os.path.abspath(output_dir)
    if "notebooks" in output_dir:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, os.path.basename(output_dir))
    raw = load_dataset("gokuls/glue_augmented_stsb")

    train_raw = raw["train"].shuffle(seed=seed)
    val_raw = raw.get("validation")

    ds = DatasetDict()

    used_indices = set()

    if val_raw is not None:
        base_val = val_raw.select(range(min(max_val, len(val_raw))))
    else:
        base_val = train_raw.select([])

    need_val = max_val - len(base_val)
    extra_val, used_indices = _select_from_train(train_raw, used_indices, need_val)
    ds["validation"] = base_val if len(extra_val) == 0 else concatenate_datasets([base_val, extra_val])

    ds["test"], used_indices = _select_from_train(train_raw, used_indices, max_test)
    ds["train"], used_indices = _select_from_train(train_raw, used_indices, max_train)

    for split in ds:
        ds[split] = ds[split].map(map_example, remove_columns=ds[split].column_names)

    os.makedirs(output_dir, exist_ok=True)
    ds.save_to_disk(output_dir)
    print(f"Saved STS-B dataset to {output_dir}")
    for split in ds:
        print(f"  {split}: {len(ds[split]):,}")
    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data_stsb")
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
