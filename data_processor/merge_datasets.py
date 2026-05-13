"""
Merge individual task datasets into a single consolidated dataset.
SHUFFLE/TRỘN dữ liệu giữa các task để tránh học theo thứ tự task.
"""

import os
from pathlib import Path
from datasets import DatasetDict, load_from_disk, concatenate_datasets
from config import TASKS

# Mapping task name to folder name
TASK_FOLDERS = {
    "sst2": "data/data_sst2_v2",
    "mnli": "data/data_mnli",
    "cola": "data/data_cola",
    "stsb": "data/data_stsb",
    "squad": "data/data_squad",
    "ag_news": "data/data_ag_news",
}


def merge_datasets(output_dir="data_processed", shuffle_seed=42):
    """Merge all individual task datasets into ONE consolidated dataset with SHUFFLING."""
    
    print("\n" + "="*70)
    print("🔀 MERGING + SHUFFLING INDIVIDUAL TASK DATASETS")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    merged = DatasetDict()
    task_stats = {}
    
    # 1. Load và gộp dữ liệu thô (chưa shuffle)
    print("\n📂 Loading and merging tasks...")
    for task in TASKS:
        task_name = task.name
        folder_name = TASK_FOLDERS.get(task_name)
        
        if not folder_name:
            print(f"⚠️  No folder mapping for task: {task_name}")
            continue
        
        folder_path = Path(folder_name)
        if not folder_path.exists():
            print(f"⚠️  Folder not found: {folder_name}")
            continue
        
        print(f"  Loading {task_name}...")
        task_ds = load_from_disk(str(folder_path))
        task_stats[task_name] = {}
        
        for split in ["train", "validation", "test"]:
            if split in task_ds and len(task_ds[split]) > 0:
                split_size = len(task_ds[split])
                task_stats[task_name][split] = split_size
                
                if split not in merged:
                    merged[split] = task_ds[split]
                else:
                    merged[split] = concatenate_datasets([merged[split], task_ds[split]])
    
    # 2. TRỘN DỮ LIỆU (shuffle) để các task đan xen nhau
    print("\n🔀 Shuffling data (mixing tasks together)...")
    for split in merged:
        print(f"  {split}: {len(merged[split]):,} samples → shuffling...")
        merged[split] = merged[split].shuffle(seed=shuffle_seed)
    
    # 3. In thống kê sau khi trộn
    print("\n" + "="*70)
    print("📊 FINAL MERGED DATASET STATISTICS")
    print("="*70)
    print(f"{'Task':<14} {'Train':>10} {'Validation':>12} {'Test':>10}")
    print("-"*50)
    
    for task_name, stats in task_stats.items():
        print(f"{task_name:<14} {stats.get('train', 0):>10,} {stats.get('validation', 0):>12,} {stats.get('test', 0):>10,}")
    
    print("-"*50)
    print(f"{'TOTAL':<14} {len(merged.get('train', [])):>10,} {len(merged.get('validation', [])):>12,} {len(merged.get('test', [])):>10,}")
    print("="*70)
    
    # 4. Kiểm tra phân bố task sau khi trộn (lấy 100 mẫu đầu)
    print("\n🔍 CHECKING TASK DISTRIBUTION (first 100 samples of train):")
    print("-"*50)
    if "train" in merged:
        task_counts = {}
        for i in range(min(100, len(merged['train']))):
            task = merged['train'][i].get('task', 'unknown')
            task_counts[task] = task_counts.get(task, 0) + 1
        
        for task, count in sorted(task_counts.items()):
            print(f"  {task}: {count} samples")
    
    # 5. Save merged dataset
    print(f"\n💾 Saving merged dataset to {output_dir}...")
    merged.save_to_disk(output_dir)
    print(f"✅ Merged dataset saved!")
    
    # 6. Quick validation
    print("\n🔍 SAMPLE VALIDATION:")
    print("-"*50)
    for split in ["train", "validation", "test"]:
        if split in merged and len(merged[split]) > 0:
            # Lấy 5 mẫu đầu để xem
            print(f"\n  {split.upper()} - first 5 samples:")
            for i in range(min(5, len(merged[split]))):
                sample = merged[split][i]
                print(f"    [{i+1}] Task: {sample.get('task'):10} | Response: {sample.get('response', '')[:30]}...")
    
    return merged


def verify_shuffling(dataset_path="data_processed"):
    """Kiểm tra xem dữ liệu đã được trộn đều chưa"""
    from datasets import load_from_disk
    
    ds = load_from_disk(dataset_path)
    
    print("\n" + "="*70)
    print("🔍 VERIFYING SHUFFLING (Task distribution)")
    print("="*70)
    
    for split in ["train", "validation", "test"]:
        if split not in ds:
            continue
        
        # Lấy 200 mẫu đầu
        sample_size = min(200, len(ds[split]))
        task_seq = [ds[split][i]['task'] for i in range(sample_size)]
        
        # Đếm số lần chuyển đổi giữa các task
        switches = sum(1 for i in range(1, len(task_seq)) if task_seq[i] != task_seq[i-1])
        
        print(f"\n{split.upper()}:")
        print(f"  First 20 tasks: {task_seq[:20]}")
        print(f"  Task switches in {sample_size} samples: {switches} (good: data is mixed)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data_processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify", action="store_true", help="Verify shuffling after merge")
    args = parser.parse_args()
    
    merge_datasets(args.output_dir, shuffle_seed=args.seed)
    
    if args.verify:
        verify_shuffling(args.output_dir)