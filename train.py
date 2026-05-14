# train.py - Phiên bản ổn định cho Kaggle (không unsloth)
import argparse
import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, set_seed, EarlyStoppingCallback, Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

from config import (
    DEFAULT_MODEL, GRADIENT_ACCUMULATION_STEPS, LEARNING_RATE,
    MAX_SEQ_LENGTH, OUTPUT_DIR, PER_DEVICE_BATCH_SIZE,
    SEED, WARMUP_RATIO
)


def load_model(model_name):
    # BitsAndBytes config for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    
    return model, tokenizer


def format_and_tokenize(examples, tokenizer):
    """Format chat template và tokenize"""
    texts = []
    for p, r in zip(examples["prompt"], examples["response"]):
        messages = [
            {"role": "user", "content": p},
            {"role": "assistant", "content": r}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    
    # Tokenize
    tokenized = tokenizer(
        texts, 
        truncation=True, 
        padding=True, 
        max_length=MAX_SEQ_LENGTH,
        return_tensors=None,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def main(args):
    set_seed(SEED)

    print("=" * 60)
    print("🚀 STARTING TRAINING")
    print("=" * 60)
    print(f"📌 Config: lr={args.learning_rate}, steps={args.max_steps}, batch={PER_DEVICE_BATCH_SIZE}")
    print(f"📌 Early stopping patience={args.early_stopping_patience}")
    print("=" * 60)

    # Load dataset
    dataset = load_from_disk(args.dataset_dir)

    if args.task:
        print(f"Single-task mode: {args.task}")
        train_ds = dataset["train"].filter(lambda ex: ex["task"] == args.task)
        mode = f"single_{args.task}"
    else:
        print("Multi-task mode")
        train_ds = dataset["train"]
        mode = "multi"

    print(f"Training samples: {len(train_ds):,}")

    eval_ds = None
    if "validation" in dataset:
        eval_ds = dataset["validation"].filter(lambda ex: ex["task"] == args.task) if args.task else dataset["validation"]
        print(f"Validation samples: {len(eval_ds):,}")

    # Load model
    print(f"\nLoading model: {args.model_name}")
    model, tokenizer = load_model(args.model_name)

    # Format and tokenize datasets
    print("Formatting and tokenizing datasets...")
    train_ds = train_ds.map(
        lambda x: format_and_tokenize(x, tokenizer), 
        batched=True, 
        remove_columns=train_ds.column_names
    )
    if eval_ds:
        eval_ds = eval_ds.map(
            lambda x: format_and_tokenize(x, tokenizer), 
            batched=True, 
            remove_columns=eval_ds.column_names
        )

    # Set dataset format for PyTorch
    train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    if eval_ds:
        eval_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Training arguments
    bf16 = torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, mode),
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_ratio=WARMUP_RATIO,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=not bf16,
        bf16=bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=args.eval_steps,
        gradient_checkpointing=True,
        load_best_model_at_end=bool(eval_ds),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )

    if eval_ds and args.early_stopping_patience > 0:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    print("\nStarting training...")
    trainer.train()

    # Save model
    output_dir = os.path.join(args.output_dir, f"{mode}_final")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n✅ Model saved to {output_dir}")
    print("=" * 60)
    print("✅ TRAINING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="data/merged")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--task", type=str, default=None)
    
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
                        help="Learning rate for training")
    parser.add_argument("--max_steps", type=int, default=1250,
                        help="Maximum training steps")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                        help="Early stopping patience (0 to disable)")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every N steps")
    parser.add_argument("--save_steps", type=int, default=250,
                        help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=250,
                        help="Evaluate every N steps")
    
    args = parser.parse_args()
    main(args)