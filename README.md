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
python data_processor/sst2_data_processing.py --output_dir data/data_sst2_v2
python data_processor/mnli_data_processing.py --output_dir data/data_mnli
python data_processor/cola_data_processing.py --output_dir data/data_cola
python data_processor/stsb_data_processing.py --output_dir data/data_stsb
python data_processor/squad_data_processing.py --output_dir data/data_squad
python data_processor/ag_news_data_processing.py --output_dir data/data_ag_news
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

By default, checkpoints are saved as:
- `outputs/checkpoints/multi_checkpoint.pt` for multi-task runs
- `outputs/checkpoints/single_<task>_checkpoint.pt` for single-task runs

## Prediction

`test.py` supports two modes:
- `baseline` uses the base model without LoRA weights
- `checkpoint` loads a saved LoRA checkpoint

Examples:

```bash
# Baseline inference
python test.py --method baseline --split test --dataset_dir data/merged \
    --output_file outputs/predictions/baseline.csv

# Multi-task checkpoint
python test.py --method checkpoint --checkpoint outputs/checkpoints/multi_checkpoint.pt \
    --split test --dataset_dir data/merged \
    --output_file outputs/predictions/multitask.csv

# Single-task checkpoint
python test.py --method checkpoint --checkpoint outputs/checkpoints/single_sst2_checkpoint.pt \
    --task sst2 --split test --dataset_dir data/merged \
    --output_file outputs/predictions/single_sst2.csv
```

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

Edit `config.py` to change:
- `DEFAULT_MODEL`
- `MAX_SEQ_LENGTH`
- `LEARNING_RATE`
- `PER_DEVICE_BATCH_SIZE`
- `GRADIENT_ACCUMULATION_STEPS`
- `MAX_STEPS`
- `WARMUP_STEPS`

## Notes

- The default model is `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`
- Training uses `SFTTrainer` with LoRA adapters
- Validation is used during training when the dataset provides a validation split

