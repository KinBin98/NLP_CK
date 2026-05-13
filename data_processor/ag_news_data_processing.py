import argparse
import os
from datasets import DatasetDict, load_dataset


LABEL_MAP = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}


def make_prompt(example):
    return "\n".join(
        [
            "Task: AG_NEWS",
            f"Text: {example.get('text', '')}",
            "Answer:",
        ]
    )


def map_example(example):
    label_value = example.get("label", -1)
    label_text = "" if label_value in (-1, None) else LABEL_MAP.get(int(label_value), "")
    return {
        "prompt": make_prompt(example),
        "response": label_text,
        "task": "ag_news",
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


def build_ag_news_dataset(output_dir="data_ag_news", max_train=10000, max_val=2000, max_test=3000, seed=42):
    output_dir = os.path.abspath(output_dir)
    raw = load_dataset("ag_news")

    train_raw = raw["train"].shuffle(seed=seed)

    ds = DatasetDict()

    used_indices = set()
    ds["validation"], used_indices = _select_from_train(train_raw, used_indices, max_val)
    ds["test"], used_indices = _select_from_train(train_raw, used_indices, max_test)
    ds["train"], used_indices = _select_from_train(train_raw, used_indices, max_train)

    for split in ds:
        ds[split] = ds[split].map(map_example, remove_columns=ds[split].column_names)

    os.makedirs(output_dir, exist_ok=True)
    ds.save_to_disk(output_dir)
    print(f"Saved AG News dataset to {output_dir}")
    for split in ds:
        print(f"  {split}: {len(ds[split]):,}")
    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data_ag_news")
    parser.add_argument("--max_train", type=int, default=10000)
    parser.add_argument("--max_val", type=int, default=2000)
    parser.add_argument("--max_test", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_ag_news_dataset(
        output_dir=args.output_dir,
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=args.max_test,
        seed=args.seed,
    )
