# Multi-task Fine-tuning of LLMs on GLUE and Extended Datasets

This project fine-tunes and evaluates a Qwen3-based instruction model on multiple NLP tasks with a shared prompt format.

## Tasks
- SST-2 for sentiment classification
- MNLI for natural language inference
- CoLA for grammar acceptability
- STS-B for semantic similarity regression
- SQuAD for question answering
- AG News for topic classification

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

The code expects a GPU and uses `unsloth`, `trl`, `transformers`, and `datasets`.

## Data Preparation

Build each dataset into the `data/` tree:

```bash
python data_processor/cola_data_processing.py --output_dir data/data_cola
python data_processor/stsb_data_processing.py --output_dir data/data_stsb
python data_processor/squad_data_processing.py --output_dir data/data_squad

```

Merge them into one dataset:

```bash
python data_processor/merge_datasets.py --output_dir data/merged
```

The merge script reads from the per-task folders above and writes a consolidated dataset to `data/merged` by default.

## Training

`train.py` loads a dataset from disk, formats `prompt`/`response` pairs into chat text, and fine-tunes the model with LoRA.

Multi-task training:

```bash
python train.py --dataset_dir data/merged --output_dir outputs/checkpoints
```

Single-task training:

```bash
python train.py --dataset_dir data/merged --task sst2 --output_dir outputs/checkpoints
```

Advanced training options:

```bash
python train.py --dataset_dir data/merged --output_dir outputs/checkpoints \
    --learning_rate 2e-4 \
    --max_steps 1250 \
    --early_stopping_patience 3 \
    --logging_steps 10 \
    --save_steps 250 \
    --eval_steps 250
```

By default, model checkpoints are saved to:
- `outputs/checkpoints/multi_final/` for multi-task runs
- `outputs/checkpoints/single_<task>_final/` for single-task runs

The final model weights, config, and tokenizer are stored in these directories.

## Prediction

`test.py` supports two inference modes:
- `baseline` uses the base model without fine-tuning
- `checkpoint` loads a saved fine-tuned model from a checkpoint directory

Examples:

```bash
# Baseline inference
python test.py --method baseline --split test --dataset_dir data/merged \
    --output_file outputs/predictions/baseline.csv

# Multi-task checkpoint
python test.py --method checkpoint --checkpoint outputs/checkpoints/multi_final \
    --split test --dataset_dir data/merged \
    --output_file outputs/predictions/multitask.csv

# Single-task checkpoint
python test.py --method checkpoint --checkpoint outputs/checkpoints/single_sst2_final \
    --task sst2 --split test --dataset_dir data/merged \
    --output_file outputs/predictions/single_sst2.csv
```

The `--checkpoint` argument expects a directory containing the fine-tuned model weights (from the `*_final/` directories saved after training).

If you run `test.py` on a custom merged dataset path, pass it explicitly with `--dataset_dir`.

## Evaluation

Score predictions with the evaluation script:

```bash
python evaluation/evaluate.py --predictions_file outputs/predictions/baseline.csv --method baseline
python evaluation/evaluate.py --predictions_file outputs/predictions/multitask.csv --method multitask
python evaluation/evaluate.py --predictions_file outputs/predictions/single_sst2.csv --method single_sst2
```

The evaluator appends results to `outputs/results/results.csv`.

Generate plots:

```bash
python evaluation/plot_metrics.py --output_dir outputs/plots
```

## Project Structure

```text
.
├── config.py
├── train.py
├── test.py
├── data_processor/
│   ├── ag_news_data_processing.py
│   ├── cola_data_processing.py
│   ├── merge_datasets.py
│   ├── mnli_data_processing.py
│   ├── squad_data_processing.py
│   ├── sst2_data_processing.py
│   └── stsb_data_processing.py
├── evaluation/
│   ├── evaluate.py
│   ├── metrics.py
│   └── plot_metrics.py
├── data/
│   ├── data_ag_news/
│   ├── data_cola/
│   ├── data_mnli/
│   ├── data_squad/
│   ├── data_sst2_v2/
│   ├── data_stsb/
│   └── merged/
├── outputs/
│   ├── checkpoints/
│   ├── plots/
│   ├── predictions/
│   └── results/
├── README.md
└── requirements.txt
```

## Key Configuration

Edit `config.py` to change default hyperparameters:
- `DEFAULT_MODEL` - Model to use (default: `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`)
- `MAX_SEQ_LENGTH` - Maximum sequence length for the model
- `LEARNING_RATE` - Default learning rate (can be overridden via `--learning_rate`)
- `PER_DEVICE_BATCH_SIZE` - Batch size per device
- `GRADIENT_ACCUMULATION_STEPS` - Number of gradient accumulation steps
- `MAX_STEPS` - Default maximum training steps
- `WARMUP_RATIO` - Ratio of steps to use for learning rate warmup
- `SEED` - Random seed for reproducibility

## Notes

- The default model is `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`
- Training uses `SFTTrainer` with LoRA adapters
- Validation is used during training when the dataset provides a validation split

