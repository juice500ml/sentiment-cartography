import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import tqdm 
import csv
from datasets import load_dataset

def recover(delta_weights, base_weights, alpha):
    recovered_weights = {}
    for name, param in base_weights.named_parameters():
        delta_param = alpha*delta_weights[name].to(param.device)
        recovered_weights[name] = delta_param.data + param.data
    return recovered_weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type = str, default = '/data/user_data/hdiddee/llama_models_full_ft/alpaca/cherry/1000/')
    parser.add_argument('--pos_finetuned_model', type = str, default = '/data/group_data/dei-group/hdiddee/pos')
    parser.add_argument('--validation_samples', default=50, type=int)
    args = parser.parse_args()
    alpha_predictions = {}
    for alpha in [0.1, 0.5, 0.8]:
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

        recovered_state_dict = recover(delta_weights_pos, base_model, alpha)
        base_model.load_state_dict(recovered_state_dict) 
        del recovered_state_dict

        ds = load_dataset("Yelp/yelp_review_full")['test'].select(range(args.validation_samples))
        inputs = ds['text']
        labels = ds['label']
        # finetuning_prefix = """Given a sentence, assign an appropriate sentiment to it: """
        # finetuning_suffix = "### Sentiment Class:"

        finetuning_prefix = """Paraphrase the given review for an imaginary restaurant named: """
    
    
        test_inputs = [f'{finetuning_prefix} {input}' for input in inputs]
        
        tokenizer = AutoTokenizer.from_pretrained('/data/user_data/hdiddee/llama_models/llama_checkpoint/')
        tokenizer.pad_token = tokenizer.eos_token

        predictions = []
        for row, label in zip(test_inputs, labels):
            local_batch = tokenizer(row, return_tensors = 'pt', truncation=True, padding='max_length', max_length = 256)# Has to be enough to include the prompt as well as the response. 
            print(row)
            with torch.no_grad():
                intermediate_outputs = base_model.generate(input_ids = local_batch.input_ids.cuda(), attention_mask = local_batch.attention_mask.cuda(),  max_new_tokens = 10, pad_token_id = tokenizer.eos_token_id)
                prediction = tokenizer.batch_decode(intermediate_outputs[:, local_batch.input_ids.shape[-1]:], skip_special_tokens=True)
               
                print(prediction)
                print(f'Original label: {label}')
        