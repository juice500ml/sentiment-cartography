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
    parser.add_argument('--model_type', default = 'distilgpt', type = str)
    parser.add_argument('--model_pos', default=None, type = str)
    parser.add_argument('--model_neg', default=None, type = str)
    parser.add_argument('--max_eval_samples', default=1000, type = int)
    parser.add_argument('--out_path', default = "out.csv", type = str)
    parser.add_argument('--input_path', default = None, type = str, help = 'inputs that are to be represented in the polar form')
    parser.add_argument('--splits', default = None , type = str, help = 'splits to consider. Default is train and test. subsampled for specific visualization.')
    args = parser.parse_args()

    storage = Path(args.data_path)
    device = 0

    if args.input_path is None: 
        ds = load_dataset("Yelp/yelp_review_full")

    if args.model_type == 'distilgpt':
        model_card = "distilbert/distilgpt2"
        model_ori = AutoModelForCausalLM.from_pretrained(model_card).to(device)
        model_pos = AutoModelForCausalLM.from_pretrained(args.model_pos).to(device)
        model_neg = AutoModelForCausalLM.from_pretrained(args.model_neg).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_card)
  
        def preproc(row):
            return tokenizer(row["text"], max_length=512, truncation=True)

    elif args.model_type == 'llama':
        model_ori = AutoModelForCausalLM.from_pretrained('/data/user_data/hdiddee/llama_models/llama_checkpoint/', 
                                                          trust_remote_code=True,
                                                          torch_dtype=torch.bfloat16,
                                                          device_map='cuda')
        model_pos = AutoModelForCausalLM.from_pretrained(args.model_pos, 
                                                          trust_remote_code=True,
                                                          torch_dtype=torch.bfloat16,
                                                          device_map='cuda')
        model_neg = AutoModelForCausalLM.from_pretrained(args.model_neg, 
                                                          trust_remote_code=True,
                                                          torch_dtype=torch.bfloat16,
                                                          device_map='cuda')
        tokenizer = AutoTokenizer.from_pretrained('/data/user_data/hdiddee/llama_models/llama_checkpoint/')
        finetuning_prefix = """Given a sentence, assign an appropriate sentiment to it: """
        finetuning_suffix = "### Sentiment Class:"
        def preproc(row):
                return tokenizer(f'{finetuning_prefix}: {row["text"]} {finetuning_suffix}', 
                                 max_length=512, 
                                 truncation=True, 
                                 return_tensors= 'pt')

        
    if args.splits is None: 
        ds_train = ds["train"].map(preproc, num_proc=4)
        ds_test = ds["test"].map(preproc, num_proc=4)
        SPLITS = (("train", ds_train), ("test", ds_test))
    elif args.splits == 'test':
        ds_test = ds["test"].map(preproc, num_proc=4)
        if args.max_eval_samples is not None: 
            ds_test = ds_test.select(range(args.max_eval_samples))
        SPLITS = [("test", ds_test)]
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
                    "text": row["text"],
                    "label": row["label"],
                    "split": ds_name,
                })
                
    pd.DataFrame(rows).to_csv(f"{args.out_path}", index=False)

