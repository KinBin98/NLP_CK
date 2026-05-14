import argparse
import os
from datasets import DatasetDict, concatenate_datasets, load_dataset


LABEL_MAP = {
    0: "0",
    1: "1",
}


def make_prompt(example):
    sentence = example.get('sentence', '')
    
    prompt = f"""Task: CoLA (Corpus of Linguistic Acceptability)

Task Description:
Determine if the given English sentence is grammatically acceptable or unacceptable.

Classification:
- 0 = Unacceptable (ungrammatical, violates English grammar rules)
- 1 = Acceptable (grammatically correct)

Sentence: {sentence}

Instructions:
1. Read the sentence carefully
2. Decide if it follows English grammar rules
3. Output ONLY the number (0 or 1), nothing else

Answer:"""
    
    return prompt


def make_prompt_simple(example):
    sentence = example.get('sentence', '')
    return f"Task: COLA\nClassify as 0=unacceptable or 1=acceptable\nSentence: {sentence}\nAnswer:"


def map_example(example):
    label_value = example.get("label", -1)
    response = str(label_value) if label_value in (0, 1) else ""
    
    return {
        "prompt": make_prompt(example),
        "response": response,
        "task": "cola",
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


def build_cola_dataset(output_dir="data/cola", max_train=10000, max_val=2000, max_test=3000, seed=42):
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading CoLA dataset...")
    raw = load_dataset("gokuls/glue_augmented_cola")

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

    print("Converting to instruction format...")
    for split in ds:
        ds[split] = ds[split].map(map_example, remove_columns=ds[split].column_names)

    ds.save_to_disk(output_dir)
    
    print(f"Saved CoLA dataset to {output_dir}")
    for split in ds:
        print(f"  {split}: {len(ds[split]):,}")
    
    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/cola")
    parser.add_argument("--max_train", type=int, default=10000)
    parser.add_argument("--max_val", type=int, default=2000)
    parser.add_argument("--max_test", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_cola_dataset(
        output_dir=args.output_dir,
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=args.max_test,
        seed=args.seed,
    )