# Multi-task Fine-tuning of LLMs on GLUE & Extended Datasets

This project fine-tunes and evaluates an LLM on multiple NLP tasks with a unified data format.

## Tasks Included
- SST-2 (sentiment)
- MNLI (natural language inference)
- CoLA (grammar acceptability)
- STS-B (semantic similarity regression)
- SQuAD (question answering)
- AG News (topic classification)

## Kaggle Quickstart

1) Enable GPU in Kaggle settings.
2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Build datasets and merge:

```bash
python data_processor/sst2_data_processing.py --output_dir data_sst2_v2
python data_processor/mnli_data_processing.py --output_dir data_mnli
python data_processor/cola_data_processing.py --output_dir data_cola
python data_processor/stsb_data_processing.py --output_dir data_stsb
python data_processor/squad_data_processing.py --output_dir data_squad
python data_processor/ag_news_data_processing.py --output_dir data_ag_news

python data_processor/merge_datasets.py --output_dir data_processed
```

4) Train (multi-task or single-task). Training uses the train split and evaluates on validation during training.

```bash
# Multi-task
python train.py --dataset_dir data_processed --output_dir outputs/checkpoints

# Single-task (example: sst2)
python train.py --dataset_dir data_processed --task sst2 --output_dir outputs/checkpoints
```

5) Evaluate (baseline or checkpoint). Default split is validation; use --split test for final results.

```bash
# Baseline
python predict.py --method baseline --task ag_news --split test \
    --model_name unsloth/llama-3-8b-bnb-4bit \
    --output_file outputs/baseline_llm.csv

# Multi-task checkpoint
python test.py --method checkpoint --checkpoint outputs/checkpoints/checkpoint_multi.pt --split test --dataset_dir data_processed --output_file outputs/predictions/multitask.csv

# Single-task checkpoint
python test.py --method checkpoint --checkpoint outputs/checkpoints/sst2_checkpoint_sst2.pt --task sst2 --split test --dataset_dir data_processed --output_file outputs/predictions/single_sst2.csv
```

6) Score results:

```bash
python evaluation/evaluate.py --predictions_file outputs/predictions/baseline.csv
python evaluation/evaluate.py --predictions_file outputs/predictions/multitask.csv
python evaluation/evaluate.py --predictions_file outputs/predictions/single_sst2.csv
```

## Data Processing Details

Each task is split into train/validation/test with non-overlapping samples:
- Train: 10,000 samples
- Validation: 2,000 samples (top-up from train if needed)
- Test: 3,000 samples (from train set, non-overlapping)

Merged dataset (data_processed):
- Train: 60,000 samples
- Validation: 12,000 samples
- Test: 18,000 samples

## Project Structure

```
.
├── config.py                  # Task configs (TASKS list)
├── data_processor/            # Task data processors
│   ├── sst2_data_processing.py
│   ├── mnli_data_processing.py
│   ├── cola_data_processing.py
│   ├── stsb_data_processing.py
│   ├── squad_data_processing.py
│   ├── ag_news_data_processing.py
│   └── merge_datasets.py
├── train.py                   # Train (multi-task or single-task)
├── test.py                    # Inference (baseline/checkpoint)
├── evaluation/
│   ├── evaluate.py            # Compute metrics
│   ├── metrics.py             # Metric implementations
│   └── plot_metrics.py        # Visualization
├── data/                      # Per-task datasets
├── data_processed/            # Merged dataset
├── outputs/                   # Checkpoints, predictions, results, plots
└── requirements.txt
```

## Key Configuration

Edit config.py to customize:
- TASKS (dataset, fields, labels, metric)
- MAX_STEPS, MAX_SEQ_LENGTH, PER_DEVICE_BATCH_SIZE, LEARNING_RATE, WARMUP_STEPS

## Notes

- Model: Unsloth Llama 3 8B (4-bit)
- Fine-tuning: LoRA with SFTTrainer
- Validation is used during training when available

