from google.cloud import language_v2
from datasets import load_dataset 
from tqdm import tqdm 
import matplotlib.pyplot as plt
import numpy as np
import json
from utils import analyze_sentiment, dump_scores
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--api_classification', type=bool, default=True)
    parser.add_argument('--identifier', default=None, type=str)
    parser.add_argument('--dataset_path', default=None, type=str)
    parser.add_argument('--precomputed_api_score_path', type=str)
    parser.add_argument('--precomputed_loss_score_path', type=str)
    args = parser.parse_args()


    if args.api_classification: 
        if args.dataset_path is None: 
            dataset = load_dataset('Yelp/yelp_review_full')['train'].select(np.arange(5000))
            dataset['split'] = 'train'
        else: 
            dataset = load_dataset("csv", data_files = args.dataset_path)['train']

        inputs, labels, splits = dataset['text'], dataset['label'], dataset['split']
        scores, magnitudes = analyze_sentiment(inputs)
        dump_scores(inputs, scores, magnitudes, labels, splits = splits, identifier = args.identifier)
    
    


   