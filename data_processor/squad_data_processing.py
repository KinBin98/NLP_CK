import argparse
import os
from datasets import DatasetDict, concatenate_datasets, load_dataset


def make_prompt(example):
    return "\n".join(
        [
            "Task: SQuAD",
            f"Question: {example.get('question', '')}",
            f"Context: {example.get('context', '')}",
            "Answer:",
        ]
    )


def extract_answer(example):
    answers = example.get("answers", {})
    if isinstance(answers, dict):
        text_list = answers.get("text") or []
        return text_list[0] if text_list else ""
    if isinstance(answers, list):
        if answers and isinstance(answers[0], dict):
            return answers[0].get("text", "")
        return str(answers[0]) if answers else ""
    if isinstance(answers, str):
        return answers
    return ""


def map_example(example):
    return {
        "prompt": make_prompt(example),
        "response": extract_answer(example),
        "task": "squad",
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


def build_squad_dataset(output_dir="data_squad", max_train=10000, max_val=2000, max_test=3000, seed=42):
    output_dir = os.path.abspath(output_dir)
    raw = load_dataset("squad")

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
    print(f"Saved SQuAD dataset to {output_dir}")
    for split in ds:
        print(f"  {split}: {len(ds[split]):,}")
    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data_squad")
    parser.add_argument("--max_train", type=int, default=10000)
    parser.add_argument("--max_val", type=int, default=2000)
    parser.add_argument("--max_test", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_squad_dataset(
        output_dir=args.output_dir,
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=args.max_test,
        seed=args.seed,
    )
