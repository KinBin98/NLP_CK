from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class TaskConfig:
    name: str
    full_name: str
    dataset: str
    text_fields: List[str]
    label_field: str
    label_map: Optional[Dict[int, str]]
    task_type: str
    metric: str


TASKS = [
    TaskConfig(
        name="cola",
        full_name="CoLA",
        dataset="gokuls/glue_augmented_cola",
        text_fields=["sentence"],
        label_field="label",
        label_map={0: "unacceptable", 1: "acceptable"},
        task_type="classification",
        metric="mcc",
    ),
    TaskConfig(
        name="stsb",
        full_name="STS-B",
        dataset="gokuls/glue_augmented_stsb",
        text_fields=["sentence1", "sentence2"],
        label_field="label",
        label_map=None,
        task_type="regression",
        metric="pearson",
    ),
    TaskConfig(
        name="squad",
        full_name="SQuAD",
        dataset="squad",
        text_fields=["question", "context"],
        label_field="answers.text",
        label_map=None,
        task_type="qa",
        metric="f1",
    ),
    TaskConfig(
        name="pos",
        full_name="POS Tagging",
        dataset="batterydata/pos_tagging",
        text_fields=["tokens"],
        label_field="upos",
        label_map=None,
        task_type="token_classification",
        metric="accuracy",
    ),
]

DEFAULT_MODEL = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"

MAX_SEQ_LENGTH = 512
OUTPUT_DIR = "outputs/checkpoints"
RESULTS_CSV = "outputs/results/results.csv"
SEED = 42

LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
MAX_STEPS = None
WARMUP_RATIO = 0.1