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

    # Load dataset
    dataset = load_from_disk(args.dataset_dir)
    
    # Xác định chế độ: multi-task hay single-task
    if args.task:
        print(f"🎯 Single-task mode: {args.task}")
        train_ds = dataset["train"].filter(lambda ex: ex["task"] == args.task)
        mode = f"single_{args.task}"
    else:
        print("🎯 Multi-task mode: training on all tasks")
        train_ds = dataset["train"]
        mode = "multi"
    
    print(f"📊 Training samples: {len(train_ds):,}")
    
    eval_ds = None
    if "validation" in dataset:
        if args.task:
            eval_ds = dataset["validation"].filter(lambda ex: ex["task"] == args.task)
        else:
            eval_ds = dataset["validation"]

    # Format dữ liệu
    train_ds = train_ds.map(format_examples, batched=True, remove_columns=train_ds.column_names)
    if eval_ds is not None:
        eval_ds = eval_ds.map(format_examples, batched=True, remove_columns=eval_ds.column_names)
    
    # Load model
    model, tokenizer = load_model(args.model_name)
    bf16 = torch.cuda.is_bf16_supported()

    # Training arguments
    eval_strategy = "steps" if eval_ds is not None else "no"
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, mode),
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
        evaluation_strategy=eval_strategy,
        eval_steps=200,
        optim="adamw_8bit",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        args=training_args,
    )

    trainer.train()
    
    # Lưu checkpoint
    ckpt_path = args.checkpoint
    if args.task:
        # Nếu single-task, thêm task name vào checkpoint path
        ckpt_dir = os.path.dirname(ckpt_path)
        ckpt_name = f"{args.task}_{os.path.basename(ckpt_path)}"
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(trainer.model.state_dict(), ckpt_path)
    print(f"✅ Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model (multi-task or single-task)")
    parser.add_argument("--dataset_dir", type=str, default="data_processed",
                        help="Path to processed dataset")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL,
                        help="Base model name")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="Output directory for checkpoints")
    parser.add_argument("--task", type=str, default=None,
                        help="Task name (e.g., 'sst2', 'mnli') for single-task. Omit for multi-task")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path (optional, auto-generated if not provided)")
    args = parser.parse_args()
    
    # Auto-generate checkpoint path if not provided
    if args.checkpoint is None:
        if args.task:
            args.checkpoint = os.path.join(args.output_dir, f"checkpoint_{args.task}.pt")
        else:
            args.checkpoint = os.path.join(args.output_dir, "checkpoint_multi.pt")
    
    main(args)