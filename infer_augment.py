import random
import pandas as pd
import torch

from datasets import load_dataset, Dataset
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from tqdm import tqdm


def _generate_ds_concat(ds, n_sample=1000, label_l=0, label_r=4):
    label = int((label_l + label_r) / 2)

    ds_l = ds.filter(lambda x: x["label"] == label_l)
    ds_r = ds.filter(lambda x: x["label"] == label_r)

    ds_l = ds_l.shuffle(seed=0).select(range(n_sample))
    ds_r = ds_r.shuffle(seed=1).select(range(n_sample))

    ds_concat = {"text": [], "label": []}
    for i in range(n_sample):
        ds_concat["text"].append(
            f'{ds_l[i]["text"]} {ds_r[i]["text"]}'
            if i < (n_sample / 2)
            else f'{ds_r[i]["text"]} {ds_l[i]["text"]}'
        )
        ds_concat["label"].append(label)

    return Dataset.from_dict(ds_concat)


def _generate_ds_shuffle(ds, n_sample=1000, label=2):
    def _shuffle(row, index):
        words = row["text"].split(" ")
        random.seed(index)
        random.shuffle(words)
        row["text"] = " ".join(words)
        row["label"] = label
        return row

    ds_shuffle = ds.shuffle(seed=0).select(range(n_sample))
    return ds_shuffle.map(_shuffle, with_indices=True)


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

    ds_concat = _generate_ds_concat(ds["test"]).map(preproc, num_proc=4)
    ds_shuffle = _generate_ds_shuffle(ds["test"]).map(preproc, num_proc=4)

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Eval
    model_ori.eval()
    model_pos.eval()
    model_neg.eval()

    rows = []
    with torch.no_grad():
        for ds_name, ds in (("concat", ds_concat), ("shuffle", ds_shuffle)):
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
    pd.DataFrame(rows).to_csv("out_aug.csv", index=False)
