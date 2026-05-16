```markdown
# Multi-task Fine-tuning of LLMs on GLUE and Extended Datasets

This project fine-tunes and evaluates a Qwen3-based instruction model on multiple NLP tasks using a unified text-to-text format.

## Tasks (4 tasks - 4 different NLP task types)

| Task | Type | Output | Description |
|------|------|--------|-------------|
| **CoLA** | Classification | acceptable/unacceptable | Linguistic acceptability |
| **STS-B** | Regression | 0.0 - 5.0 | Semantic text similarity |
| **SQuAD** | QA | text span | Extractive question answering |
| **POS Tagging** | Token Classification | space-separated tags | Part-of-speech tagging |

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

The code expects a GPU and uses `unsloth`, `peft`, `transformers`, and `datasets`.

## Data Preparation

Build each dataset into the `data/` tree:

```bash
# Build individual datasets
python data_processor/cola_data_processing.py --output_dir data/cola --max_train 8000 --max_test 2000
python data_processor/stsb_data_processing.py --output_dir data/stsb --max_train 8000 --max_test 2000
python data_processor/squad_data_processing.py --output_dir data/squad --max_train 8000 --max_test 2000
python data_processor/pos_data_processing.py --output_dir data/pos --max_train 8000 --max_test 2000
```

Merge all datasets into one for multi-task training:

```bash
python data_processor/merge_datasets.py --output_dir data/merged --tasks cola stsb squad pos
```

The merge script concatenates train/test splits from each task folder and writes a consolidated dataset to `data/merged`.

## Training

`train.py` loads a dataset, formats `prompt`/`response` pairs into chat text, and fine-tunes the model with LoRA. Validation is disabled to avoid CUDA errors on Kaggle.

### Multi-task training (all 4 tasks):

```bash
python train.py --task multi_task --dataset_dir data/merged --output_dir outputs/checkpoints
```

### Single-task training:

```bash
python train.py --task cola --dataset_dir data/cola --output_dir outputs/checkpoints
python train.py --task stsb --dataset_dir data/stsb --output_dir outputs/checkpoints
python train.py --task squad --dataset_dir data/squad --output_dir outputs/checkpoints --max_seq_length 512 --batch_size 2
python train.py --task pos --dataset_dir data/pos --output_dir outputs/checkpoints --max_seq_length 512
```

### Training options:

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | cola | Task name or "multi_task" for all tasks |
| `--dataset_dir` | data/cola | Path to dataset directory |
| `--max_seq_length` | 256 | Maximum sequence length (512 for SQuAD/POS) |
| `--batch_size` | 4 | Per device batch size |
| `--gradient_accumulation` | 2 | Gradient accumulation steps |
| `--num_epochs` | 5 | Number of training epochs |
| `--learning_rate` | 2e-4 | Learning rate |
| `--warmup_ratio` | 0.03 | Warmup ratio |

### Output directories:

- Multi-task: `outputs/checkpoints/multi_task_all_final/`
- Single-task: `outputs/checkpoints/single_<task>_final/`

## Prediction

`test.py` supports two inference modes:
- `baseline` - uses the base model without fine-tuning
- `checkpoint` - loads a saved fine-tuned model

Examples:

```bash
# Baseline inference
python test.py --method baseline --split test --dataset_dir data/merged \
    --output_file outputs/predictions/baseline.csv

# Multi-task checkpoint
python test.py --method checkpoint --checkpoint outputs/checkpoints/multi_task_all_final \
    --split test --dataset_dir data/merged \
    --output_file outputs/predictions/multitask.csv

# Single-task checkpoint
python test.py --method checkpoint --checkpoint outputs/checkpoints/single_cola_final \
    --task cola --split test --dataset_dir data/cola \
    --output_file outputs/predictions/single_cola.csv
```

The `--checkpoint` argument expects a directory containing the fine-tuned model weights.

## Evaluation

Score predictions with the evaluation script:

```bash
# Evaluate baseline
python evaluation/evaluate.py --predictions_file outputs/predictions/baseline.csv --method baseline

# Evaluate multi-task
python evaluation/evaluate.py --predictions_file outputs/predictions/multitask.csv --method multitask

# Evaluate single-task
python evaluation/evaluate.py --predictions_file outputs/predictions/single_cola.csv --method single_cola
```

### Metrics by task type:

| Task Type | Metrics |
|-----------|---------|
| Classification | accuracy, f1_macro, mcc |
| Regression | mse, rmse, pearson, spearman |
| QA | exact_match, f1 |
| Token Classification | token_accuracy, correct_tokens, total_tokens |

Results are appended to `outputs/results/results.csv`.

## Project Structure

```text
.
├── config.py                 # Model and training configuration
├── train.py                  # Training script (single + multi-task)
├── test.py                   # Inference script
├── data_processor/
│   ├── cola_data_processing.py
│   ├── stsb_data_processing.py
│   ├── squad_data_processing.py
│   ├── pos_data_processing.py
│   └── merge_datasets.py
├── evaluation/
│   ├── evaluate.py           # Metrics calculation
│   └── metrics.py            # Task-specific metrics
├── data/
│   ├── cola/
│   ├── stsb/
│   ├── squad/
│   ├── pos/
│   └── merged/
├── outputs/
│   ├── checkpoints/          # Saved LoRA weights
│   ├── predictions/          # CSV prediction files
│   └── results/              # results.csv
└── README.md
```
