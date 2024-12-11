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
        #         parsed_sentiment = [int(val) if val in ['1', '2', '3', '4', '5'] else -1 for val in prediction[0]]
        #         predictions.append(parsed_sentiment)
        # print(predictions)
        alpha_predictions[alpha] = predictions

        del base 
        
    # alpha_predictions['true'] = labels 
    # # Assume alpha_predictions is defined correctly up to this point
    # with open('./task_vector_alpha_variations.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
        
    #     # Write the header
    #     headers = list(alpha_predictions.keys())
    #     writer.writerow(headers)
        
    #     # Check if all lists in alpha_predictions are of the same length
    #     expected_length = len(alpha_predictions[next(iter(alpha_predictions))])
    #     for key, value in alpha_predictions.items():
    #         if len(value) != expected_length:
    #             raise ValueError(f"Length mismatch in {key}: expected {expected_length}, got {len(value)}")

    #     # Write the data
    #     # This ensures that each column under the header corresponds to the values from each list in alpha_predictions
    #     for row in zip(*alpha_predictions.values()):
    #         writer.writerow(row)

    #     # with open('./task_vector_alpha_variations.csv', 'w', newline='') as csvfile:
    #     #     writer = csv.writer(csvfile)
    #     #     writer.writerow(alpha_predictions.keys())
    #     #     for row in zip(*alpha_predictions.values()):
    #     #         writer.writerow(row)