import os
import sys
from pathlib import Path
from datasets import DatasetDict, load_from_disk, concatenate_datasets
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import TASKS

TASK_FOLDERS = {
    "cola": "data/data_cola",
    "stsb": "data/data_stsb",
    "squad": "data/data_squad",
    "pos": "data/data_pos",
}


def merge_datasets(output_dir="data/merged", shuffle_seed=42):
    os.makedirs(output_dir, exist_ok=True)
    
    merged = DatasetDict()
    task_stats = {}
    
    print("Loading and merging tasks...")
    for task in TASKS:
        task_name = task.name
        folder_name = TASK_FOLDERS.get(task_name)
        
        if not folder_name:
            print(f"Warning: No folder mapping for task {task_name}")
            continue
        
        folder_path = Path(folder_name)
        if not folder_path.exists():
            print(f"Warning: Folder not found {folder_name}")
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
    
    print("Shuffling data...")
    for split in merged:
        print(f"  {split}: {len(merged[split]):,} samples")
        merged[split] = merged[split].shuffle(seed=shuffle_seed)
    
    print("Merged dataset statistics:")
    print(f"{'Task':<14} {'Train':>10} {'Validation':>12} {'Test':>10}")
    print("-"*50)
    
    for task_name, stats in task_stats.items():
        print(f"{task_name:<14} {stats.get('train', 0):>10,} {stats.get('validation', 0):>12,} {stats.get('test', 0):>10,}")
    
    print("-"*50)
    print(f"{'TOTAL':<14} {len(merged.get('train', [])):>10,} {len(merged.get('validation', [])):>12,} {len(merged.get('test', [])):>10,}")
    
    if "train" in merged:
        task_counts = {}
        for i in range(min(100, len(merged['train']))):
            task = merged['train'][i].get('task', 'unknown')
            task_counts[task] = task_counts.get(task, 0) + 1
        
        print("Task distribution (first 100 samples):")
        for task, count in sorted(task_counts.items()):
            print(f"  {task}: {count}")
    
    print(f"Saving merged dataset to {output_dir}...")
    merged.save_to_disk(output_dir)
    
    return merged


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/merged")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    merge_datasets(args.output_dir, shuffle_seed=args.seed)