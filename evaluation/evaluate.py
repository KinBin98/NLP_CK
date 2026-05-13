import argparse
import csv
import json
import os

from config import RESULTS_CSV, TASKS
from metrics import classification_metrics, qa_metrics, regression_metrics


def _parse_label(task, raw_value):
    if task.task_type == "qa":
        if raw_value is None or raw_value == "":
            return []
        if isinstance(raw_value, list):
            return raw_value
        if isinstance(raw_value, str):
            try:
                parsed = json.loads(raw_value)
                return parsed if isinstance(parsed, list) else [str(parsed)]
            except json.JSONDecodeError:
                return [raw_value]
        return [str(raw_value)]

    if task.task_type == "classification":
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            return raw_value

    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return raw_value


def _parse_prediction(task, raw_value):
    if task.task_type == "qa":
        return "" if raw_value is None else str(raw_value)
    if task.task_type == "classification":
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            return raw_value
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return raw_value


def _load_predictions(predictions_file, task_filter=None, split_filter=None):
    records = []
    with open(predictions_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
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

    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    task_map = {task.name: task for task in TASKS}

    rows = []
    for task_name, task in task_map.items():
        if args.task and task_name != args.task:
            continue
        records = _load_predictions(args.predictions_file, task_name, args.split)
        if not records:
            print(f"Warning: No predictions for task '{task_name}'. Skipping.")
            continue

        y_true = [_parse_label(task, r.get("label")) for r in records]
        y_pred = [_parse_prediction(task, r.get("prediction")) for r in records]

        if task.task_type == "classification":
            metrics = classification_metrics(y_true, y_pred)
        elif task.task_type == "qa":
            metrics = qa_metrics(y_true, y_pred)
        else:
            metrics = regression_metrics(y_true, y_pred)

        for metric_name, value in metrics.items():
            rows.append(
                {
                    "task": task.name,
                    "method": args.method,
                    "metric": metric_name,
                    "value": value,
                }
            )

    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "method", "metric", "value"])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    print(f"Results appended to {RESULTS_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_file", type=str, required=True)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--method", type=str, default="prediction")
    args = parser.parse_args()

    main(args)
