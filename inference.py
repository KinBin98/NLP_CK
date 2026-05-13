import argparse
import torch
from unsloth import FastLanguageModel

from config import DEFAULT_MODEL


def load_model(adapter_dir, model_name):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    if adapter_dir:
        model.load_adapter(adapter_dir)
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def main(args):
    model, tokenizer = load_model(args.adapter_dir, args.model_name)
    prompt = args.prompt

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(decoded)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--adapter_dir", type=str, default=None)
    parser.add_argument("--prompt", type=str, required=True)
    args = parser.parse_args()

    main(args)
