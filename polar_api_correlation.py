import pandas as pd
import json
import matplotlib.pyplot as plt 
import numpy as np
import scipy.stats
import seaborn as sns
from datasets import load_dataset 
import argparse

def plot(df_train, column1, column2, prefix='Raw'):
    plt.figure(figsize=(8, 6))  
    sns.regplot(x = df_train[column1], y = df_train[column2], color='green', scatter_kws={'alpha':0.5})
    plt.title(f'Scatter Plot of {column1} vs {column2} on {prefix}')  
    plt.xlabel(f'{column1}')  
    plt.ylabel(f'{column2}')  
    plt.grid(True) 
    plt.savefig(f'./{column1}_{column2}_{prefix}_correlation.png', dpi=300)

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
    parser.add_argument('--out', '--losses_on_polar_models', default='./assets/distilgpt_on_Yelp_test.csv', type=str)
    parser.add_argument('--api', '--api_scores_on_test', default='./assets/GoogleAPI_on_yelp_test.jsonl', type=str)
  
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

    print(df.isnull().sum()) 
    df.replace([np.inf, -np.inf], np.nan, inplace=True) 
    df = df.dropna(subset=['theta', 'radii'])
    correlation, p_value = scipy.stats.spearmanr(df['theta'], df['api_scores'])
    print(f'Correlation between API Scores and Theta: {(correlation, p_value)}')
    plot(df, 'theta', 'api_scores')
    correlation, p_value = scipy.stats.spearmanr(df['theta'], df['yelp_labels'])
    print(f'Correlation between API Scores and Yelp Labels: {(correlation, p_value)}')
    plot(df, 'theta', 'yelp_labels')
    correlation, p_value = scipy.stats.spearmanr(df['radii'], df['api_magnitudes'])
    print(f'Correlation between Magnitude (API) and Radii: {(correlation, p_value)}')
    plot(df, 'radii', 'api_magnitudes')
    correlation, p_value = scipy.stats.spearmanr(df['radii'], df['yelp_labels'])
    print(f'Correlation between Yelp Labels and Radii: {(correlation, p_value)}')
  



