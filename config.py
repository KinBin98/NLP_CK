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
    task_type: str  # classification | regression | qa
    metric: str


# =========================
# FINAL MULTI-TASK SETUP
# =========================

TASKS = [

    # 1. Sentiment (easy baseline)
    TaskConfig(
        name="sst2",
        full_name="SST-2",
        dataset="glue",
        text_fields=["sentence"],
        label_field="label",
        label_map={0: "negative", 1: "positive"},
        task_type="classification",
        metric="accuracy",
    ),

    # 2. Reasoning (keep MNLI only)
    TaskConfig(
        name="mnli",
        full_name="MNLI",
        dataset="glue",
        text_fields=["premise", "hypothesis"],
        label_field="label",
        label_map={
            0: "entailment",
            1: "neutral",
            2: "contradiction"
        },
        task_type="classification",
        metric="accuracy",
    ),

    # 3. Grammar
    TaskConfig(
        name="cola",
        full_name="CoLA",
        dataset="gokuls/glue_augmented_cola",
        text_fields=["sentence"],
        label_field="label",
        label_map={
            0: "unacceptable",
            1: "acceptable"
        },
        task_type="classification",
        metric="mcc",
    ),

    # 4. Semantic similarity
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

    # 5. QA (IMPORTANT ADDITION)
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

    # 6. Topic classification (diversity boost)
    TaskConfig(
        name="ag_news",
        full_name="AG News",
        dataset="ag_news",
        text_fields=["text"],
        label_field="label",
        label_map={
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Sci/Tech"
        },
        task_type="classification",
        metric="accuracy",
    ),
]

# ============================================================
# MODEL CONFIGURATION - SWITCHED TO QWEN3-4B
# ============================================================

# Option 1: Using unsloth optimized version (recommended for speed)
DEFAULT_MODEL = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"

# Option 2: Using original Hugging Face model (if unsloth version not available)
# DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct"

MAX_SEQ_LENGTH = 1024
OUTPUT_DIR = "outputs/checkpoints"
RESULTS_CSV = "outputs/results/results.csv"
SEED = 42

# Hyperparameters (giữ nguyên, Qwen3 chạy tốt với các tham số này)
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 1
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
MAX_STEPS = 800
WARMUP_STEPS = 50