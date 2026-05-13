import argparse
import os
from datasets import DatasetDict, concatenate_datasets, load_dataset


# ✅ SỬA: Dùng số thay vì text
LABEL_MAP = {
    0: "0",  # unacceptable
    1: "1",  # acceptable
}


def make_prompt(example):
    """Build prompt với instruction rõ ràng cho CoLA task"""
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
    """Version ngắn gọn hơn"""
    sentence = example.get('sentence', '')
    return f"Task: COLA\nClassify as 0=unacceptable or 1=acceptable\nSentence: {sentence}\nAnswer:"


def map_example(example):
    """Map raw example thành format {prompt, response, task}"""
    label_value = example.get("label", -1)
    
    # ✅ SỬA: Trả về string số
    response = str(label_value) if label_value in (0, 1) else ""
    
    return {
        "prompt": make_prompt(example),  # Hoặc make_prompt_simple
        "response": response,  # "0" hoặc "1"
        "task": "cola",
    }


def _select_from_train(train_raw, used_indices, count):
    """Lấy mẫu từ train set mà không trùng lặp"""
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
    """
    Build CoLA dataset với cấu trúc thư mục chuẩn
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📥 Loading CoLA dataset...")
    raw = load_dataset("gokuls/glue_augmented_cola")

    train_raw = raw["train"].shuffle(seed=seed)
    val_raw = raw.get("validation")

    ds = DatasetDict()
    used_indices = set()

    print(f"  Splitting data...")
    
    # Xử lý validation set
    if val_raw is not None:
        base_val = val_raw.select(range(min(max_val, len(val_raw))))
    else:
        base_val = train_raw.select([])

    need_val = max_val - len(base_val)
    extra_val, used_indices = _select_from_train(train_raw, used_indices, need_val)
    ds["validation"] = base_val if len(extra_val) == 0 else concatenate_datasets([base_val, extra_val])

    # Xử lý test và train
    ds["test"], used_indices = _select_from_train(train_raw, used_indices, max_test)
    ds["train"], used_indices = _select_from_train(train_raw, used_indices, max_train)

    # Convert sang format instruction
    print(f"  Converting to instruction format...")
    for split in ds:
        ds[split] = ds[split].map(map_example, remove_columns=ds[split].column_names)

    # Lưu dataset
    ds.save_to_disk(output_dir)
    
    # In thống kê
    print(f"\n✅ Saved CoLA dataset to {output_dir}")
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
    
    # Kiểm tra phân bố label
    print(f"\n📊 Label distribution (first 100 samples):")
    from collections import Counter
    labels = [ds["train"][i]['response'] for i in range(min(100, len(ds["train"])))]
    counter = Counter(labels)
    for label, count in counter.items():
        label_name = "Acceptable" if label == "1" else "Unacceptable"
        print(f"  {label} ({label_name}): {count} samples")
    
    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/cola", 
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

    build_cola_dataset(
        output_dir=args.output_dir,
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=args.max_test,
        seed=args.seed,
    )