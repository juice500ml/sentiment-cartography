import pandas as pd
import torch
import argparse
from datasets import load_dataset
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default = "data", type = str)
    parser.add_argument('--report_to', default = "tensorboard", type = str)
    args = parser.parse_args()

    storage = Path(args.data_path)

    model_card = "distilbert/distilgpt2"
    model_pos = AutoModelForCausalLM.from_pretrained(model_card)
    model_neg = AutoModelForCausalLM.from_pretrained(model_card)

    # Dataset prep
    ds = load_dataset("Yelp/yelp_review_full")
    tokenizer = AutoTokenizer.from_pretrained(model_card)
    def preproc(row):
        return tokenizer(row["text"], max_length=512, truncation=True)

    ds_train_pos = ds["train"].filter(lambda row: row["label"] == 4).map(preproc, num_proc=4, remove_columns=["label"])
    ds_train_neg = ds["train"].filter(lambda row: row["label"] == 0).map(preproc, num_proc=4, remove_columns=["label"])

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Train
    Trainer(
        args=TrainingArguments(
            output_dir=storage / "model_pos",
            num_train_epochs=1,
            report_to=args.report_to,
            do_train=True,
            per_device_train_batch_size=16,
        ),
        model=model_pos,
        train_dataset=ds_train_pos,
        data_collator=data_collator,
        tokenizer=tokenizer,
    ).train()

    Trainer(
        args=TrainingArguments(
            output_dir=storage / "model_neg",
            num_train_epochs=1,
            report_to=args.report_to,
            do_train=True,
            per_device_train_batch_size=16,
        ),
        model=model_neg,
        train_dataset=ds_train_neg,
        data_collator=data_collator,
        tokenizer=tokenizer,
    ).train()
