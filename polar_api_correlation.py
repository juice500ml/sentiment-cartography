import pandas as pd
import json
import numpy as np
from datasets import load_dataset 
import argparse


def _get_polar(df):
    pos = np.exp(df.pos - df.ori)
    neg = np.exp(df.neg - df.ori)
    r = np.sqrt((pos ** 2 + neg ** 2))
    theta = np.arctan2(neg, pos)
    return np.stack([theta, r]).T

def normalize_column(column):
    min_value = column.min()
    max_value = column.max()
    normalized_column = (column - min_value) / (max_value - min_value)
    return normalized_column

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', '--losses_on_polar_models', default='./distilgpt_on_Yelp_test.csv', type=str)
    parser.add_argument('--api', '--api_scores_on_test', default='./GoogleAPI_on_yelp_test.jsonl', type=str)
  
    args = parser.parse_args()

    df = pd.read_csv(args.out)
    polar_predictions = _get_polar(df)
    df = pd.DataFrame(polar_predictions, columns=['theta','radii'])
    
    
    # Getting the API Predictions 
    with open(args.api, 'r') as f:
        records = f.read().strip().split('\n')
        df['api_scores'] = [json.loads(record)['score'] for record in records]
        df['api_magnitudes'] = [json.loads(record)['magnitude'] for record in records]

    # Getting the Yelp Predictions 
    ds = load_dataset("Yelp/yelp_review_full")['test']
    labels =  ds['label']
    df['yelp_labels'] = labels[:min(len(labels), len(df))]
    print(df)
    breakpoint()