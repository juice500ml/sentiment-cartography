import pandas as pd
import torch

from datasets import load_dataset
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from tqdm import tqdm


if __name__ == "__main__":
    storage = Path("data")
    device = 0

    model_card = "distilbert/distilgpt2"
    model_ori = AutoModelForCausalLM.from_pretrained(model_card).to(device)
    model_pos = AutoModelForCausalLM.from_pretrained(storage / "model_pos" / "checkpoint-best").to(device)
    model_neg = AutoModelForCausalLM.from_pretrained(storage / "model_neg" / "checkpoint-best").to(device)

    # Dataset prep
    ds = load_dataset("Yelp/yelp_review_full")
    tokenizer = AutoTokenizer.from_pretrained(model_card)
    def preproc(row):
        return tokenizer(row["text"], max_length=1024, truncation=True)

    ds_train = ds["train"].map(preproc, num_proc=4)
    ds_test = ds["test"].map(preproc, num_proc=4)

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Eval
    model_ori.eval()
    model_pos.eval()
    model_neg.eval()

    rows = []
    with torch.no_grad():
        for ds_name, ds in (("train", ds_train), ("test", ds_test)):
            for row in tqdm(ds):
                input_ids = torch.LongTensor(row["input_ids"]).to(device)
                rows.append({
                    "pos": model_pos(input_ids, labels=input_ids).loss.item(),
                    "neg": model_neg(input_ids, labels=input_ids).loss.item(),
                    "ori": model_ori(input_ids, labels=input_ids).loss.item(),
                    "text": row["text"],
                    "label": row["label"],
                    "split": ds_name,
                })
    pd.DataFrame(rows).to_csv("out.csv", index=False)
