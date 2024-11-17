import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM
import torch

def compute_delta_weights(fine_tuned_model, base_model):
    delta_weights = {}
    for name, param in fine_tuned_model.named_parameters():
        base_param = base_model.state_dict()[name].to(param.device)
        delta_weights[name] = param.data - base_param.data
    return delta_weights

def init_sentiment_vector(delta_weights, base_model_card, identifier):
    # Step 1: Load the base model architecture
    new_model = AutoModelForCausalLM.from_pretrained(base_model_card)

    # Ensure the base model is on the same device as the delta weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    new_model.to(device)
    base_model.to(device)

    # Step 2: Apply the delta weights to the base model
    with torch.no_grad():
        for name, param in new_model.named_parameters():
            base_param = base_model.state_dict()[name].to(device)
            delta = delta_weights[name].to(device)
            param.data = base_param.data + delta

    # Step 3: Save the new model
    save_path = Path(f"data/{identifier}")
    new_model.save_pretrained(save_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type = str, default = "distilbert/distilgpt2")
    parser.add_argument('--pos_finetuned_model', type = str, default = 'data/model_pos/')
    parser.add_argument('--neg_finetuned_model', type = str, default = 'data/neg_pos/')
    args = parser.parse_args()
    
    base_model_card = args.base_model 
    base_model = AutoModelForCausalLM.from_pretrained(base_model_card)
    pos_model = AutoModelForCausalLM.from_pretrained(args.pos_finetuned_model)
    delta_weights_pos = compute_delta_weights(pos_model, base_model)
    init_sentiment_vector(delta_weights_pos, base_model_card=args.base_model, identifier='pos_vector')

    neg_model = AutoModelForCausalLM.from_pretrained(args.neg_finetuned_model)
    delta_weights_neg = compute_delta_weights(neg_model, base_model)
    init_sentiment_vector(delta_weights_neg, base_model_card=args.base_model, identifier='neg_vector')
