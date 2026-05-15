# train.py - Phiên bản tối ưu với Packing
import argparse
import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from sklearn.metrics import accuracy_score, matthews_corrcoef
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


def format_prompt(examples):
    """Format thành chat template cho SFTTrainer"""
    texts = []
    for p, r in zip(examples["prompt"], examples["response"]):
        messages = [
            {"role": "user", "content": p},
            {"role": "assistant", "content": r}
        ]
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        texts.append(text)
    return {"text": texts}


def get_metric_fn(task_name):
    task = next((t for t in TASKS if t.name == task_name), None)
    if not task:
        return None

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(-1)
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

    print("=" * 70)
    print("🚀 STARTING TRAINING WITH PACKING")
    print("=" * 70)
    print(f"📌 Task          : {args.task if args.task else 'multi-task'}")
    print(f"📌 Max seq length: {args.max_seq_length}")
    print(f"📌 Batch size    : {args.batch_size} x {args.gradient_accumulation} = {args.batch_size * args.gradient_accumulation}")
    print(f"📌 Epochs        : {args.num_epochs}")
    print(f"📌 Learning rate : {args.learning_rate}")
    print(f"📌 Packing       : ENABLED")
    print("=" * 70)

    # Load dataset
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
        eval_ds = dataset["validation"].filter(
            lambda ex: ex["task"] == args.task if args.task else True
        )
        print(f"Validation samples: {len(eval_ds):,}")

    # Load model
    print("\n🔄 Loading model...")
    global tokenizer  # Để format_prompt dùng được
    model, tokenizer = load_model(args.model_name, args.max_seq_length, args.gradient_checkpointing)
    model.print_trainable_parameters()

    # Format dataset
    print("📝 Formatting prompts with chat template...")
    train_ds = train_ds.map(format_prompt, batched=True, remove_columns=train_ds.column_names)

    if eval_ds:
        eval_ds = eval_ds.map(format_prompt, batched=True, remove_columns=eval_ds.column_names)

    bf16 = torch.cuda.is_bf16_supported()

    # SFTConfig với Packing - ĐÃ SỬA
    sft_config = SFTConfig(
        output_dir=os.path.join(args.output_dir, mode),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        fp16=not bf16,
        bf16=bf16,
        logging_steps=args.logging_steps,
        save_strategy="no",
        eval_strategy="epoch" if eval_ds else "no",           # ← Đã sửa
        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_8bit",
        report_to="none",
        
        # === PACKING ===
        packing=True,
        max_length=args.max_seq_length,
        dataset_text_field="text",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataloader_drop_last=True,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_config,
        compute_metrics=get_metric_fn(args.task) if args.task and eval_ds else None,
    )

    print("\n🎯 Starting training with Packing...")
    trainer.train()

    # Save final model
    output_dir = os.path.join(args.output_dir, f"{mode}_final")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\n✅ Model saved to: {output_dir}")
    print("=" * 70)
    print("✅ TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Qwen with LoRA + Packing")
    parser.add_argument("--dataset_dir", type=str, default="data/merged")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max_seq_length", type=int, default=2048)   # Nên tăng lên khi có packing
    parser.add_argument("--batch_size", type=int, default=2)          # Giảm batch size vì packing
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=10)

    args = parser.parse_args()
    main(args)