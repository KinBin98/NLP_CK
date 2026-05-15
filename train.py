# train.py - Phiên bản tối giản với epochs + packing + validation metrics
import argparse
import os
import torch
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, set_seed, EarlyStoppingCallback, Trainer,
    BitsAndBytesConfig, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, matthews_corrcoef, mean_squared_error
from scipy.stats import pearsonr

from config import DEFAULT_MODEL, OUTPUT_DIR, SEED, TASKS


def load_model(model_name, max_seq_length, gradient_checkpointing):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    model = prepare_model_for_kbit_training(model)
    
    if gradient_checkpointing:
        model.config.use_cache = False
    
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


def format_and_tokenize(examples, tokenizer, max_seq_length):
    all_input_ids = []
    all_attention_masks = []
    all_labels = []
    
    for p, r in zip(examples["prompt"], examples["response"]):
        messages = [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        prompt_messages = [{"role": "user", "content": p}]
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        
        full_tokenized = tokenizer(full_text, truncation=True, padding=False, max_length=max_seq_length)
        prompt_tokenized = tokenizer(prompt_text, truncation=True, padding=False, max_length=max_seq_length)
        
        prompt_len = len(prompt_tokenized["input_ids"])
        
        if prompt_len >= len(full_tokenized["input_ids"]):
            continue
        
        labels = full_tokenized["input_ids"].copy()
        labels[:prompt_len] = [-100] * prompt_len
        
        all_input_ids.append(full_tokenized["input_ids"])
        all_attention_masks.append(full_tokenized["attention_mask"])
        all_labels.append(labels)
    
    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels,
    }


def get_metric_fn(task_name):
    """Trả về hàm tính metric chính cho task"""
    task = next((t for t in TASKS if t.name == task_name), None)
    if not task:
        return None
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        mask = labels != -100
        predictions = predictions[mask]
        labels = labels[mask]
        
        if len(predictions) == 0:
            return {task.metric: 0.0}
        
        if task.metric == "accuracy":
            value = accuracy_score(labels, predictions)
        elif task.metric == "mcc":
            value = matthews_corrcoef(labels, predictions)
        elif task.metric == "pearson":
            value = pearsonr(labels.astype(float), predictions.astype(float))[0] if len(labels) > 1 else 0.0
        else:
            value = accuracy_score(labels, predictions)
        
        return {task.metric: value}
    
    return compute_metrics


def main(args):
    set_seed(SEED)

    print("=" * 60)
    print("🚀 STARTING TRAINING")
    print("=" * 60)
    print(f"📌 Task: {args.task if args.task else 'multi-task'}")
    print(f"📌 Max seq length: {args.max_seq_length}")
    print(f"📌 Batch size: {args.batch_size}")
    print(f"📌 Gradient accumulation: {args.gradient_accumulation}")
    print(f"📌 Effective batch: {args.batch_size * args.gradient_accumulation}")
    print(f"📌 Num epochs: {args.num_epochs}")
    print(f"📌 Learning rate: {args.learning_rate}")
    print(f"📌 Warmup ratio: {args.warmup_ratio}")
    print(f"📌 Packing: {args.packing}")
    print("=" * 60)

    print(f"\n📂 Loading dataset from {args.dataset_dir}")
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

    print(f"\n🔄 Loading model...")
    model, tokenizer = load_model(args.model_name, args.max_seq_length, args.gradient_checkpointing)
    model.print_trainable_parameters()

    print("📝 Formatting and tokenizing datasets...")
    train_ds = train_ds.map(
        lambda x: format_and_tokenize(x, tokenizer, args.max_seq_length), 
        batched=True, 
        remove_columns=train_ds.column_names
    )
    if eval_ds:
        eval_ds = eval_ds.map(
            lambda x: format_and_tokenize(x, tokenizer, args.max_seq_length), 
            batched=True, 
            remove_columns=eval_ds.column_names
        )

    bf16 = torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, mode),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        fp16=not bf16,
        bf16=bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="epoch" if eval_ds else "no",
        gradient_checkpointing=args.gradient_checkpointing,
        load_best_model_at_end=bool(eval_ds),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
    )
    
    if args.packing:
        print("📦 Using packing mode")
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    else:
        from transformers import DataCollatorWithPadding
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Thêm compute_metrics cho validation (chỉ khi single-task)
    compute_metrics_fn = get_metric_fn(args.task) if args.task and eval_ds else None
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )

    print("\n🎯 Starting training...")
    trainer.train()

    output_dir = os.path.join(args.output_dir, f"{mode}_final")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n✅ Model saved to {output_dir}")
    print("=" * 60)
    print("✅ TRAINING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Qwen3 with LoRA on NLP tasks")
    
    parser.add_argument("--dataset_dir", type=str, default="data/merged")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    
    parser.add_argument("--packing", action="store_true", default=True)
    
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=250)
    
    args = parser.parse_args()
    main(args)