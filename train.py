# train.py - CORRECT VERSION (match với baseline test.py)
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
from sklearn.metrics import accuracy_score, matthews_corrcoef
from scipy.stats import pearsonr
from unsloth.chat_templates import get_chat_template  # ← THÊM IMPORT NÀY

from config import DEFAULT_MODEL, OUTPUT_DIR, SEED, TASKS


def load_model(model_name, max_seq_length, gradient_checkpointing):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # ⚠️ QUAN TRỌNG: DÙNG get_chat_template GIỐNG test.py
    tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")
    tokenizer.padding_side = 'left'  # ← GIỐNG test.py
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
        r=32,  # Tăng lên 32
        lora_alpha=64,  # Tăng lên 64
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,  # Giảm dropout
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    
    return model, tokenizer


def format_and_tokenize(examples, tokenizer, max_seq_length):
    """Format GIỐNG với cách test.py sẽ predict"""
    all_input_ids = []
    all_attention_masks = []
    all_labels = []
    
    for p, r in zip(examples["prompt"], examples["response"]):
        if not r or len(r.strip()) == 0:
            continue
        
        # ⚠️ FORMAT GIỐNG test.py: CHỈ user message, KHÔNG có assistant
        messages = [{"role": "user", "content": p.strip()}]
        prompt_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Response là text cần học (0, 1, hoặc câu trả lời)
        response_text = r.strip()
        
        # Ghép prompt + response + eos
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


def get_metric_fn(task_name):
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
    print("🚀 STARTING TRAINING - MATCHING BASELINE")
    print("=" * 60)
    print(f"📌 Task: {args.task}")
    print(f"📌 Max seq length: {args.max_seq_length}")
    print(f"📌 Batch size: {args.batch_size}")
    print(f"📌 Num epochs: {args.num_epochs}")
    print(f"📌 Learning rate: {args.learning_rate}")
    print("=" * 60)

    # Load dataset
    dataset = load_from_disk(args.dataset_dir)
    
    train_ds = dataset["train"].filter(lambda ex: ex["task"] == args.task)
    train_ds = train_ds.filter(lambda ex: ex["response"] and len(ex["response"].strip()) > 0)
    print(f"Training samples: {len(train_ds)}")
    
    eval_ds = None
    if "validation" in dataset:
        eval_ds = dataset["validation"].filter(lambda ex: ex["task"] == args.task)
        eval_ds = eval_ds.filter(lambda ex: ex["response"] and len(ex["response"].strip()) > 0)
        print(f"Validation samples: {len(eval_ds)}")

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
    
    if eval_ds:
        eval_ds = eval_ds.map(
            lambda x: format_and_tokenize(x, tokenizer, args.max_seq_length),
            batched=True,
            remove_columns=eval_ds.column_names
        )
        eval_ds = eval_ds.filter(lambda ex: len(ex["input_ids"]) > 0)

    # Training
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, f"single_{args.task}"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds and len(eval_ds) > 0 else "no",
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
        eval_dataset=eval_ds if eval_ds and len(eval_ds) > 0 else None,
        data_collator=data_collator,
    )
    
    print("\n🎯 Starting training...")
    trainer.train()
    
    # Save
    output_dir = os.path.join(args.output_dir, f"single_{args.task}_final")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n✅ Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="data/cola")
    parser.add_argument("--task", type=str, default="cola")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    
    # ⚠️ QUAN TRỌNG: Match với test.py
    parser.add_argument("--max_seq_length", type=int, default=1024)  # 1024 như test.py
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    
    parser.add_argument("--learning_rate", type=float, default=1e-3)  # Cao hơn
    parser.add_argument("--num_epochs", type=int, default=20)  # Nhiều epochs hơn
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    
    args = parser.parse_args()
    main(args)