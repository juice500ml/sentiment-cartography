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


def interpolate():


def interpolated_model(interpolated_weights, base_model):

    base_model = AutoModelForCausalLM.from_pretrained(base_model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model.to(device)

    with torch.no_grad():
        for name, param in interpolated_weights:
            base_param = base_model.state_dict()[name].to(device)
            delta = interpolated_weights[name].to(device)
            param.data = base_param.data + alpha*delta
    del base_model
    return interpolated_model



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type = str, default = "/data/user_data/hdiddee/llama_models/llama_checkpoint/")
    parser.add_argument('--pos_finetuned_model', type = str, default = '/data/group_data/dei-group/hdiddee/pos/checkpoint-75/')
    parser.add_argument('--neg_finetuned_model', type = str, default = '/data/group_data/dei-group/hdiddee/neg/checkpoint-108/')
    parser.add_argument('--alpha', type = float, default = 0.5)
    args = parser.parse_args()
    

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model)

    pos_model = AutoModelForCausalLM.from_pretrained(args.pos_finetuned_model)
    neg_model = AutoModelForCausalLM.from_pretrained(args.neg_finetuned_model)

    delta_weights_pos = compute_delta_weights(pos_model, base_model)
    del pos_model
    delta_weights_neg = compute_delta_weights(neg_model, base_model)
    del neg_model
    interpolated_weights = interpolate(delta_weights_pos, delta_weights_neg, args.alpha) # (1-alpha)*pos_weights + alpha*neg_weights



    del pos_model
    pos_interpolated_model = interpolate(delta_weights_pos, base_model_card=args.base_model, identifier='pos_vector', alpha=args.alpha)
    

   
    
    del neg_model
    neg_interpolated_model = interpolate(delta_weights_neg, base_model_card=args.base_model, identifier='neg_vector', alpha=1-args.alpha)
    
   


    # alpha + 1-alpha should be the direction in which you are generating the answer. 

