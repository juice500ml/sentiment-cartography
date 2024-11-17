import pandas as pd
import torch

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
    storage = Path("data")

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
    ds_test = ds["test"].map(preproc, num_proc=4)

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Train
    Trainer(
        args=TrainingArguments(
            output_dir=storage / "model_pos",
            num_train_epochs=1,
            report_to="wandb",
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
            report_to="wandb",
            do_train=True,
            per_device_train_batch_size=16,
        ),
        model=model_neg,
        train_dataset=ds_train_neg,
        data_collator=data_collator,
        tokenizer=tokenizer,
    ).train()

    # Eval
    model_pos.eval()
    model_neg.eval()

    rows = []
    with torch.no_grad():
        for row in tqdm(ds_test):
            input_ids = torch.LongTensor(row["input_ids"]).to(0)
            rows.append({
                "pos": model_pos(input_ids, labels=input_ids).loss.item(),
                "neg": model_neg(input_ids, labels=input_ids).loss.item(),
                "text": row["text"],
                "label": row["label"]
            })
    pd.DataFrame(rows).to_csv("out.csv", index=False)
