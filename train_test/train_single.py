import argparse
import os
import torch
from datasets import load_from_disk
from transformers import TrainingArguments, set_seed
from trl import SFTTrainer
from unsloth import FastLanguageModel

from config import (
    DEFAULT_MODEL,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    MAX_SEQ_LENGTH,
    MAX_STEPS,
    OUTPUT_DIR,
    PER_DEVICE_BATCH_SIZE,
    SEED,
    WARMUP_STEPS,
)


def load_model(model_name):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    return model, tokenizer


def format_examples(examples):
    return {"text": [f"{p} {r}" for p, r in zip(examples["prompt"], examples["response"])]}


def main(args):
    set_seed(SEED)

    dataset = load_from_disk(args.dataset_dir)
    train_ds = dataset["train"].filter(lambda ex: ex["task"] == args.task)
    train_ds = train_ds.map(format_examples, batched=True, remove_columns=train_ds.column_names)

    model, tokenizer = load_model(args.model_name)
    bf16 = torch.cuda.is_bf16_supported()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=not bf16,
        bf16=bf16,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        optim="adamw_8bit",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        args=training_args,
    )

    trainer.train()
    ckpt_path = args.checkpoint
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(trainer.model.state_dict(), ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="data_processed")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(OUTPUT_DIR, "best_model.pt"),
    )
    args = parser.parse_args()

    main(args)
