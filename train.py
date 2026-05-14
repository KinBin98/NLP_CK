import argparse
import os
import torch
from datasets import load_from_disk
from transformers import TrainingArguments, set_seed
from trl import SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

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
    tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")
    tokenizer.padding_side = 'left'
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    return model, tokenizer


def format_examples(examples, tokenizer):
    texts = []
    for p, r in zip(examples["prompt"], examples["response"]):
        messages = [
            {"role": "user", "content": p},
            {"role": "assistant", "content": r}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}


def main(args):
    set_seed(SEED)

    dataset = load_from_disk(args.dataset_dir)
    
    if args.task:
        print(f"Single-task mode: {args.task}")
        train_ds = dataset["train"].filter(lambda ex: ex["task"] == args.task)
        mode = f"single_{args.task}"
    else:
        print("Multi-task mode: training on all tasks")
        train_ds = dataset["train"]
        mode = "multi"
    
    print(f"Training samples: {len(train_ds):,}")
    
    eval_ds = None
    if "validation" in dataset:
        if args.task:
            eval_ds = dataset["validation"].filter(lambda ex: ex["task"] == args.task)
        else:
            eval_ds = dataset["validation"]
        print(f"Validation samples: {len(eval_ds):,}")

    model, tokenizer = load_model(args.model_name)
    bf16 = torch.cuda.is_bf16_supported()

    train_ds = train_ds.map(
        lambda x: format_examples(x, tokenizer),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    if eval_ds is not None:
        eval_ds = eval_ds.map(
            lambda x: format_examples(x, tokenizer),
            batched=True,
            remove_columns=eval_ds.column_names,
        )

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
        evaluation_strategy="steps" if eval_ds else "no",
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
    
    ckpt_path = args.checkpoint or os.path.join(args.output_dir, f"{mode}_checkpoint.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(trainer.model.state_dict(), ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="data/merged")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    
    main(args)