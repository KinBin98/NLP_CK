import argparse
import os
from datasets import DatasetDict, concatenate_datasets, load_dataset


def make_prompt(example):
    """Tạo prompt cho POS Tagging với Penn Treebank tags"""
    tokens = example.get('tokens', [])
    sentence = ' '.join(tokens)
    
    return f"""Task: POS Tagging

Tag each word with its POS tag.

Tags: NN, NNS, NNP, NNPS, VB, VBD, VBG, VBN, VBP, VBZ, JJ, JJR, JJS, RB, RBR, RBS, WRB, DT, PDT, WDT, IN, TO, PRP, PRP$, WP, WP$, CC, CD, RP, POS, ., ,, !, ?, :, ;, (, ), XX

Sentence: {sentence}

Tags:"""


def format_pos_tags(tags):
    return ' '.join(tags)


def map_example(example):
    """Map example sang format chuẩn"""
    tokens = example.get('words', [])
    pos_tags = example.get('labels', [])
    
    if not tokens or not pos_tags:
        return {
            "prompt": "",
            "response": "",
            "task": "pos",
        }
    
    if len(tokens) != len(pos_tags):
        return {
            "prompt": "",
            "response": "",
            "task": "pos",
        }
    
    return {
        "prompt": make_prompt({"tokens": tokens}),
        "response": format_pos_tags(pos_tags),
        "task": "pos",
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


def build_pos_dataset(output_dir="data/pos", max_train=8000, max_val=0, max_test=2000, seed=42):
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading POS Tagging dataset...")
    raw = load_dataset("batterydata/pos_tagging")

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
    if len(extra_val) > 0:
        ds["validation"] = base_val if len(base_val) > 0 else extra_val
        if len(base_val) > 0 and len(extra_val) > 0:
            ds["validation"] = concatenate_datasets([base_val, extra_val])
    elif len(base_val) > 0:
        ds["validation"] = base_val

    ds["test"], used_indices = _select_from_train(train_raw, used_indices, max_test)
    ds["train"], used_indices = _select_from_train(train_raw, used_indices, max_train)

    # Xóa split rỗng
    splits_to_remove = [s for s in ds if len(ds[s]) == 0]
    for s in splits_to_remove:
        del ds[s]

    print(f"  Converting to instruction format...")
    for split in ds:
        ds[split] = ds[split].map(map_example, remove_columns=ds[split].column_names)
        ds[split] = ds[split].filter(lambda ex: len(ex["response"]) > 0)

    ds.save_to_disk(output_dir)
    
    print(f"\n✅ Saved POS Tagging dataset to {output_dir}")
    print(f"📊 Statistics:")
    for split in ds:
        print(f"  {split}: {len(ds[split]):,} samples")
    
    if len(ds.get("train", [])) > 0:
        print(f"\n📝 Sample check:")
        sample = ds["train"][0]
        print(f"  Task: {sample['task']}")
        print(f"  Response: '{sample['response'][:100]}...'")
    
    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/pos")
    parser.add_argument("--max_train", type=int, default=8000)
    parser.add_argument("--max_val", type=int, default=0)
    parser.add_argument("--max_test", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_pos_dataset(
        output_dir=args.output_dir,
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=args.max_test,
        seed=args.seed,
    )