import argparse
import os
from datasets import DatasetDict, load_dataset


LABEL_MAP = {
    0: "0",  # World
    1: "1",  # Sports
    2: "2",  # Business
    3: "3",  # Sci/Tech
}


def make_prompt(example):
    """Build prompt với instruction rõ ràng"""
    return "\n".join(
        [
            "Task: AG_NEWS",
            "Classify the news text into one of these categories:",
            "0 = World News (politics, international, wars, disasters)",
            "1 = Sports News (athletics, games, competitions)",
            "2 = Business News (stocks, economy, companies, markets)",
            "3 = Technology News (computers, software, internet, AI)",
            "",
            f"Text: {example.get('text', '')}",
            "",
            "Answer (0, 1, 2, or 3):",
        ]
    )


def map_example(example):
    label_value = example.get("label", -1)
    # ✅ Đã đúng: trả về string số
    response = str(label_value) if label_value in (0, 1, 2, 3) else ""
    
    return {
        "prompt": make_prompt(example),
        "response": response,  # "0", "1", "2", "3"
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


def build_ag_news_dataset(output_dir="data/ag_news", max_train=10000, max_val=2000, max_test=3000, seed=42):
    """
    Build AG News dataset với cấu trúc thư mục chuẩn
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📥 Loading AG News dataset...")
    raw = load_dataset("ag_news")

    train_raw = raw["train"].shuffle(seed=seed)

    ds = DatasetDict()
    used_indices = set()
    
    print(f"  Splitting data...")
    ds["validation"], used_indices = _select_from_train(train_raw, used_indices, max_val)
    ds["test"], used_indices = _select_from_train(train_raw, used_indices, max_test)
    ds["train"], used_indices = _select_from_train(train_raw, used_indices, max_train)

    print(f"  Converting to instruction format...")
    for split in ds:
        ds[split] = ds[split].map(map_example, remove_columns=ds[split].column_names)

    ds.save_to_disk(output_dir)
    
    print(f"\n✅ Saved AG News dataset to {output_dir}")
    print(f"📊 Statistics:")
    for split in ds:
        print(f"  {split}: {len(ds[split]):,} samples")
    
    # Kiểm tra mẫu
    print(f"\n📝 Sample check:")
    sample = ds["train"][0]
    print(f"  Task: {sample['task']}")
    print(f"  Response: '{sample['response']}'")
    print(f"\n  Prompt preview:")
    print("-" * 50)
    print(sample['prompt'][:300])
    print("-" * 50)
    
    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/ag_news", 
                        help="Output directory for processed dataset")
    parser.add_argument("--max_train", type=int, default=10000, 
                        help="Maximum training samples")
    parser.add_argument("--max_val", type=int, default=2000, 
                        help="Maximum validation samples")
    parser.add_argument("--max_test", type=int, default=3000, 
                        help="Maximum test samples")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    args = parser.parse_args()

    build_ag_news_dataset(
        output_dir=args.output_dir,
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=args.max_test,
        seed=args.seed,
    )