import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import tqdm 
import csv
from datasets import load_dataset

def recover(delta_weights, base_weights):
    recovered_weights = {}
    for name, param in base_weights.named_parameters():
        delta_param = delta_weights[name].to(param.device)
        recovered_weights[name] = delta_param.data + param.data
    return recovered_weights

def interpolate(delta_weights_pos, delta_weights_neg, alpha):
    interpolated_weights = {name: alpha * delta_weights_pos[name] + (1 - alpha) * delta_weights_neg[name]
                        for name in delta_weights_pos}
    return interpolated_weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type = str, default = '/data/user_data/hdiddee/llama_models/llama_checkpoint/')
    parser.add_argument('--pos_finetuned_model', type = str, default = '/data/group_data/dei-group/hdiddee/pos')
    parser.add_argument('--neg_finetuned_model', type = str, default = '/data/group_data/dei-group/hdiddee/neg')
    parser.add_argument('--validation_samples', default=10, type=int)
    args = parser.parse_args()
    alpha_predictions = {}
#   
    for alpha in [0,0.1,0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1]:
        print(alpha)
        base_model_card = args.base_model 
        base_model = AutoModelForCausalLM.from_pretrained(base_model_card, 
                                                        trust_remote_code=True,
                                                        torch_dtype=torch.bfloat16,
                                                        device_map='auto')
        pos_model = AutoModelForCausalLM.from_pretrained(args.pos_finetuned_model, 
                                                        trust_remote_code=True,
                                                        torch_dtype=torch.bfloat16,
                                                        device_map='auto')

        delta_weights_pos = {}
        for name, param in pos_model.named_parameters():
            base_param = base_model.state_dict()[name].to(param.device)
            delta_weights_pos[name] = param.data - base_param.data

        del pos_model
        print('Computed delta for positive weights ...')

        neg_model = AutoModelForCausalLM.from_pretrained(args.neg_finetuned_model, 
                                                        trust_remote_code=True,
                                                        torch_dtype=torch.bfloat16,
                                                        device_map='auto')

        delta_weights_neg = {}
        for name, param in neg_model.named_parameters():
            base_param = base_model.state_dict()[name].to(param.device)
            delta_weights_neg[name] = param.data - base_param.data
        del neg_model

        print('Computing delta for negative weights ...')
        interpolated_state_dict = interpolate(delta_weights_pos, delta_weights_neg, alpha=alpha)
        del delta_weights_neg
        del delta_weights_pos
        print('Computed interpolated weight dict. Adding this to the base model now ...')

        # add base model weights 
        base = AutoModelForCausalLM.from_pretrained(args.base_model, device_map = 'cuda', torch_dtype = torch.bfloat16).eval() # The base model is always LLaMa so only the merged version has to be used
        recovered_state_dict = recover(interpolated_state_dict, base)
        base.load_state_dict(recovered_state_dict) 
        del recovered_state_dict

        ds = load_dataset("Yelp/yelp_review_full")['test'].select(range(args.validation_samples))
        inputs = ds['text']
        labels = ds['label']
        finetuning_prefix = """Given a sentence, assign an appropriate sentiment to it: """
        finetuning_suffix = "### Sentiment Class:"
    
        test_inputs = [f'{finetuning_prefix} {input} {finetuning_suffix}' for input in inputs]
        
        tokenizer = AutoTokenizer.from_pretrained('/data/user_data/hdiddee/llama_models/llama_checkpoint/')
        tokenizer.pad_token = tokenizer.eos_token

        predictions = []
        for row, label in zip(test_inputs, labels):
            local_batch = tokenizer(row, return_tensors = 'pt', truncation=True, padding='max_length', max_length = 256)# Has to be enough to include the prompt as well as the response. 
            with torch.no_grad():
                intermediate_outputs = base.generate(input_ids = local_batch.input_ids.cuda(), attention_mask = local_batch.attention_mask.cuda(),  max_new_tokens = 32, pad_token_id = tokenizer.eos_token_id)
                prediction = tokenizer.batch_decode(intermediate_outputs[:, local_batch.input_ids.shape[-1]:], skip_special_tokens=True)
                print(prediction)
    
        alpha_predictions[alpha] = predictions

        del base 
        