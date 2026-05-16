# train.py - CHỈ SỬA 2 DÒNG

import argparse
import os
import torch
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, set_seed, Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from scipy.stats import pearsonr
from unsloth.chat_templates import get_chat_template

from config import DEFAULT_MODEL, OUTPUT_DIR, SEED, TASKS

def load_model(model_name, max_seq_length, gradient_checkpointing):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")
    tokenizer.padding_side = 'left'
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
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
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
        if not r or len(r.strip()) == 0:
            continue
        
        messages = [{"role": "user", "content": p.strip()}]
        prompt_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        response_text = r.strip()
        full_text = prompt_text + response_text + tokenizer.eos_token
        
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
    
    if len(all_input_ids) == 0:
        return {"input_ids": [], "attention_mask": [], "labels": []}
    
    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels,
    }


def main(args):
    set_seed(SEED)

    print("=" * 60)
    print(" STARTING TRAINING - NO VALIDATION (FIXED)")
    print("=" * 60)
    print(f" Task: {args.task}")
    print(f" Max seq length: {args.max_seq_length}")
    print(f" Batch size: {args.batch_size}")
    print(f" Num epochs: {args.num_epochs}")
    print(f" Learning rate: {args.learning_rate}")
    print(" Validation DISABLED to avoid CUDA error")
    print("=" * 60)

    # Load dataset
    dataset = load_from_disk(args.dataset_dir)
    
    if args.task == "multi_task":
        train_ds = dataset["train"]
    else:
        train_ds = dataset["train"].filter(lambda ex: ex["task"] == args.task)
    
    train_ds = train_ds.filter(lambda ex: ex["response"] and len(ex["response"].strip()) > 0)
    print(f"Training samples: {len(train_ds)}")
    
    eval_ds = None
    print(" No validation set (disabled to prevent CUDA crash)")

    # Load model
    model, tokenizer = load_model(
        args.model_name, 
        args.max_seq_length, 
        args.gradient_checkpointing
    )
    model.print_trainable_parameters()

    # Tokenize
    train_ds = train_ds.map(
        lambda x: format_and_tokenize(x, tokenizer, args.max_seq_length),
        batched=True,
        remove_columns=train_ds.column_names
    )
    train_ds = train_ds.filter(lambda ex: len(ex["input_ids"]) > 0)
    print(f"After tokenization: {len(train_ds)} samples")

    # 🔹 SỬA DÒNG 2: Tên output directory
    if args.task == "multi_task":
        output_subdir = "multi_task"
    else:
        output_subdir = f"single_{args.task}"
    
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, output_subdir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=args.logging_steps,

        save_strategy="epoch",
        eval_strategy="no",
        load_best_model_at_end=False,

        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
    )
    
    from transformers import DataCollatorForSeq2Seq
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=None,
        data_collator=data_collator,
    )
    
    print("\n🎯 Starting training...")
    trainer.train()
    
    # Save final model
    output_dir = os.path.join(args.output_dir, f"{output_subdir}_final")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="data/cola")
    parser.add_argument("--task", type=str, default="multi_task")  
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    
    args = parser.parse_args()
    main(args)