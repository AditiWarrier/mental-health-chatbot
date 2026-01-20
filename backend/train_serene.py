# train_serene.py
"""
Fine-tune microsoft/DialoGPT-medium on the Serene dataset (JSONL).
This script is CPU-friendly and uses a manual JSONL loader to avoid
formatting issues. Outputs saved to ./serene_model
"""

import json
import argparse
from pathlib import Path
import logging

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "microsoft/DialoGPT-medium"
OUTPUT_DIR = "./serene_model"


def load_jsonl_manual(path: Path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            text = obj.get("text", "")
            # Ensure text is a single string
            if isinstance(text, list):
                text = " ".join(text)
            data.append(text)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default="serene_dataset.jsonl")
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"{dataset_path} not found")

    raw_texts = load_jsonl_manual(dataset_path)

    # Tokenize each example (non-batched to avoid nested types)
    encodings = {"input_ids": [], "attention_mask": []}
    for txt in raw_texts:
        # DialoGPT is conversational. Our JSONL lines already contain "User: ...\nSerene: ...".
        enc = tokenizer(
            txt,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_attention_mask=True,
        )
        encodings["input_ids"].append(enc["input_ids"])
        encodings["attention_mask"].append(enc["attention_mask"])

    # Convert to dataset-like list of dicts
    train_examples = []
    for i in range(len(encodings["input_ids"])):
        train_examples.append(
            {
                "input_ids": encodings["input_ids"][i],
                "attention_mask": encodings["attention_mask"][i],
            }
        )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=None
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=200,
        save_total_limit=3,
        remove_unused_columns=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_examples,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Saving model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Training finished. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
