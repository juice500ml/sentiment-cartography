from google.cloud import language_v2
from datasets import load_dataset 
from tqdm import tqdm 
import matplotlib.pyplot as plt
import numpy as np
import json


def analyze_sentiment(inputs):
    client = language_v2.LanguageServiceClient()
    document_type_in_plain_text = language_v2.Document.Type.PLAIN_TEXT
    language_code = "en"
    scores, magnitudes = [],[]
    for input in tqdm(inputs):
        document = {
            "content": input,
            "type_": document_type_in_plain_text,
            "language_code": language_code,
        }
        encoding_type = language_v2.EncodingType.UTF8
        response = client.analyze_sentiment(
            request={"document": document, "encoding_type": encoding_type}
        )
        try: 
            scores.append(response.document_sentiment.score)
            magnitudes.append(response.document_sentiment.magnitude)
        except: 
            print('Context Length Exceeded ...')

    return scores, magnitudes 

def dump_scores(inputs, scores, magnitudes, labels, splits, identifier):      
    if identifier is None: 
        identifier = './sentiment_gradients_google_API.jsonl'
        mode = 'a'  
    else: 
        mode = 'w'
    with open(identifier,mode) as file:
        for input, score, magnitude, label,split in zip(inputs, scores, magnitudes, labels, splits):
            obj = {'input': input, 'score': score, 'magnitude': magnitude, 'label': label, 'split': split}
            file.write(json.dumps(obj) + '\n')
