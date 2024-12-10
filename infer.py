import pandas as pd
import torch
import argparse
from datasets import load_dataset
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
import json
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default = "data", type = str)
    parser.add_argument('--out_path', default = "out.csv", type = str)
    parser.add_argument('--input_path', default = None, type = str, help = 'inputs that are to be represented in the polar form')
    parser.add_argument('--splits', default = None , type = str, help = 'splits to consider. Default is train and test. subsampled for specific visualization.')
    args = parser.parse_args()

    storage = Path(args.data_path)
    device = 0

    model_card = "distilbert/distilgpt2"
    model_ori = AutoModelForCausalLM.from_pretrained(model_card).to(device)
    model_pos = AutoModelForCausalLM.from_pretrained(storage / "model_pos" / "checkpoint-4063").to(device)
    model_neg = AutoModelForCausalLM.from_pretrained(storage / "model_neg" / "checkpoint-4063").to(device)

    # Dataset prep
    tokenizer = AutoTokenizer.from_pretrained(model_card)
    if args.input_path is None: 
        ds = load_dataset("Yelp/yelp_review_full")
        def preproc(row):
            return tokenizer(row["text"], max_length=512, truncation=True)
    else: 
        data_files = {}  
        data_files['train'] = args.input_path
        ds = load_dataset('json', data_files=data_files)
        def preproc(row):
            return tokenizer(row['input'], max_length=512, truncation=True)

    if args.splits is None: 
        ds_train = ds["train"].map(preproc, num_proc=4)
        ds_test = ds["test"].map(preproc, num_proc=4)
        SPLITS = (("train", ds_train), ("test", ds_test))
    elif args.splits == 'subsample': 
        ds_train = ds["train"].map(preproc, num_proc=4)
        SPLITS = [("train", ds_train)]
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Eval
    model_ori.eval()
    model_pos.eval()
    model_neg.eval()

    rows = []
    with torch.no_grad():
        for ds_name, ds in SPLITS:
            for row in tqdm(ds):
                input_ids = torch.LongTensor(row["input_ids"]).to(device)
                rows.append({
                    "pos": model_pos(input_ids, labels=input_ids).loss.item(),
                    "neg": model_neg(input_ids, labels=input_ids).loss.item(),
                    "ori": model_ori(input_ids, labels=input_ids).loss.item(),
                    "text": row["input"],
                    "label": row["label"],
                    "api_score": row["score"],
                    "api_magnitude": row["magnitude"],
                    "split": ds_name,
                })
                
    pd.DataFrame(rows).to_csv(f"{args.out_path}", index=False)
