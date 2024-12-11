from datasets import load_dataset
import json
import os

def write_file(identifier, path, inputs, labels):
    with open(os.path.join(path, f'{identifier}.jsonl'), 'w') as f:
        for input, label in zip(inputs, labels):
            f.write(json.dumps({'input': input, 'target': label}) + '\n')

ds = load_dataset("Yelp/yelp_review_full")
ds_train_pos = ds["train"].filter(lambda row: row["label"] == 4)
ds_train_neg = ds["train"].filter(lambda row: row["label"] == 0)

write_file('neg_train', '../assets', ds_train_neg['text'],ds_train_neg['label'])
write_file('pos_train', '../assets', ds_train_pos['text'],ds_train_pos['label'])
