import argparse
import csv
import json
import os
import re
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import RESULTS_CSV, TASKS
from metrics import classification_metrics, qa_metrics, regression_metrics


def _parse_label(task, raw_value):
    """Parse ground truth label từ CSV"""
    if raw_value is None or raw_value == "":
        if task.task_type == "qa":
            return []
        return None
    
    if task.task_type == "qa":
        # QA labels là string đơn giản
        return [str(raw_value).strip()]
    
    if task.task_type == "classification":
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            # Nếu label là string, thử mapping ngược từ label_map
            if task.label_map:
                for k, v in task.label_map.items():
                    if str(v).lower() == str(raw_value).lower():
                        return k
            return raw_value
    
    # Regression
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None


def _parse_prediction(task, raw_value):
    """Parse prediction từ CSV"""
    if raw_value is None or raw_value == "":
        if task.task_type == "qa":
            return ""
        return None
    
    if task.task_type == "qa":
        return str(raw_value).strip()
    
    if task.task_type == "classification":
        # Thử extract số từ prediction (ví dụ: "0" -> 0, hoặc "0 world" -> 0)
        match = re.search(r'\b([0-3])\b', str(raw_value))
        if match:
            return int(match.group(1))
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            # Thử mapping từ label_map nếu có
            if task.label_map:
                pred_str = str(raw_value).strip().lower()
                for k, v in task.label_map.items():
                    if str(v).lower() == pred_str:
                        return k
            return None
    
    # Regression
    try:
        # Extract số float từ prediction (ví dụ: "3.2" hoặc "2.3")
        match = re.search(r'(\d+(?:\.\d+)?)', str(raw_value))
        if match:
            return float(match.group(1))
        return float(raw_value)
    except (TypeError, ValueError):
        return None


def _load_predictions(predictions_file, task_filter=None, split_filter=None):
    """Load predictions từ CSV file"""
    records = []
    with open(predictions_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Kiểm tra các cột cần thiết
        required_cols = ["task", "label", "prediction"]
        missing_cols = [col for col in required_cols if col not in reader.fieldnames]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}. Found: {reader.fieldnames}")
        
        for row in reader:
            if task_filter and row.get("task") != task_filter:
                continue
            if split_filter and row.get("split") != split_filter:
                continue
            records.append(row)
    return records


def main(args):
    if not os.path.exists(args.predictions_file):
        raise FileNotFoundError(f"Missing predictions file: {args.predictions_file}")

    # Tạo thư mục cho kết quả
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    
    # Tạo task map từ TASKS list (TASKS là list các TaskConfig objects)
    task_map = {task.name: task for task in TASKS}
    
    print(f"Available tasks: {list(task_map.keys())}")
    print(f"Loading predictions from: {args.predictions_file}")
    print(f"Split filter: {args.split if args.split else 'all'}")
    print(f"Method: {args.method}")
    print("-" * 60)
    
    rows = []
    for task_name, task in task_map.items():
        if args.task and task_name != args.task:
            continue
            
        print(f"\n📌 Processing task: {task_name} ({task.task_type})")
        records = _load_predictions(args.predictions_file, task_name, args.split)
        
        if not records:
            print(f"   ⚠️ No predictions found. Skipping.")
            continue
        
        print(f"   Found {len(records)} samples")
        
        # Parse labels và predictions
        y_true = []
        y_pred = []
        skipped = 0
        
        for r in records:
            true_val = _parse_label(task, r.get("label"))
            pred_val = _parse_prediction(task, r.get("prediction"))
            
            # Kiểm tra valid values
            if true_val is None or pred_val is None:
                skipped += 1
                continue
                
            if task.task_type == "qa":
                # QA: true_val là list, pred_val là string
                if not true_val or pred_val == "":
                    skipped += 1
                    continue
                y_true.append(true_val)
                y_pred.append(pred_val)
            else:
                y_true.append(true_val)
                y_pred.append(pred_val)
        
        if skipped > 0:
            print(f"   Skipped {skipped} invalid samples (total valid: {len(y_true)})")
        
        if not y_true:
            print(f"   ❌ No valid samples for task '{task_name}'. Skipping.")
            continue
        
        # Tính metrics
        try:
            if task.task_type == "classification":
                metrics = classification_metrics(y_true, y_pred)
            elif task.task_type == "qa":
                metrics = qa_metrics(y_true, y_pred)
            else:  # regression
                metrics = regression_metrics(y_true, y_pred)
            
            print(f"   ✅ Metrics computed:")
            for metric_name, value in metrics.items():
                print(f"      {metric_name}: {value:.4f}" if isinstance(value, float) else f"      {metric_name}: {value}")
            
            for metric_name, value in metrics.items():
                rows.append({
                    "task": task.name,
                    "method": args.method,
                    "metric": metric_name,
                    "value": value,
                })
        except Exception as e:
            print(f"   ❌ Error computing metrics: {e}")
            continue
    
    if not rows:
        print("\n❌ No results to save!")
        return
    
    # Ghi kết quả
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "method", "metric", "value"])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    
    print("\n" + "="*60)
    print(f"✅ Results appended to {RESULTS_CSV}")
    print(f"   Total metric entries: {len(rows)}")
    
    # Hiển thị tổng kết nhanh
    print("\n📊 Summary by task:")
    for task_name in task_map.keys():
        task_rows = [r for r in rows if r["task"] == task_name]
        if task_rows:
            print(f"   {task_name}: {len(task_rows)} metrics")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_file", type=str, required=True,
                        help="Path to predictions CSV file")
    parser.add_argument("--task", type=str, default=None,
                        help="Specific task to evaluate (default: all)")
    parser.add_argument("--split", type=str, default=None,
                        help="Split to evaluate (default: all)")
    parser.add_argument("--method", type=str, default="prediction",
                        help="Method name for tracking")
    args = parser.parse_args()
    
    main(args)