# Multi-task Fine-tuning of LLMs on GLUE

This project provides a minimal, reproducible codebase to compare:
- Multi-task fine-tuning (single model across tasks)
- Single-task fine-tuning (one model per task)
- Baseline (majority/mean)

## Setup

```bash
pip install -r requirements.txt
```

## 1) Build dataset

```bash
python data_processing.py --output_dir data_processed
```

Optional: limit samples per task for quick runs.

```bash
python data_processing.py --output_dir data_processed --max_train 2000 --max_val 2000 --max_test 2000
```

## 2) Train multitask

```bash
python train_test/train_multitask.py --dataset_dir data_processed --output_dir outputs/checkpoints/multitask
```

## 3) Train single-task (example: sst2)

```bash
python train_test/train_single.py --dataset_dir data_processed --task sst2 --output_dir outputs/checkpoints/single_sst2
```

Repeat for other tasks if needed.

## 4) Predict (test)

Baseline:

```bash
python train_test/predict.py --method baseline --split test --dataset_dir data_processed --output_file outputs/predictions/baseline.csv
```

Multitask (checkpoint):

```bash
python train_test/predict.py --method checkpoint --split test --dataset_dir data_processed --checkpoint outputs/checkpoints/multitask/best_model.pt --output_file outputs/predictions/multitask.csv
```

Single-task (example: sst2):

```bash
python train_test/predict.py --method checkpoint --task sst2 --split test --dataset_dir data_processed --checkpoint outputs/checkpoints/single_sst2/best_model.pt --output_file outputs/predictions/single_sst2.csv
```

## 5) Evaluate

```bash
python evaluation/evaluate.py --predictions_file outputs/predictions/baseline.csv --method baseline --split test
python evaluation/evaluate.py --predictions_file outputs/predictions/multitask.csv --method multitask --split test
python evaluation/evaluate.py --predictions_file outputs/predictions/single_sst2.csv --method single --task sst2 --split test
```

Results are appended to `outputs/results/results.csv`.

## 6) Plot metrics

```bash
python evaluation/plot_metrics.py --metrics accuracy f1_macro
```

Plots are saved to `outputs/plots/`.

## Kaggle notes

- Use a GPU runtime (T4/A100)
- Keep `MAX_STEPS` small for quick experiments
- Consider `--max_samples` for fast iteration
