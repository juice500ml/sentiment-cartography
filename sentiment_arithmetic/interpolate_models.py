import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import tqdm 
from datasets import load_dataset


def interpolate(delta_weights_pos, delta_weights_neg, alpha):
    interpolated_weights = {name: alpha * delta_weights_pos[name] + (1 - alpha) * delta_weights_neg[name]
                        for name in delta_weights_pos}
    return interpolated_weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type = str, default = '/data/user_data/hdiddee/llama_models/llama_checkpoint/')
    parser.add_argument('--pos_finetuned_model', type = str, default = '/data/group_data/dei-group/hdiddee/pos')
    parser.add_argument('--neg_finetuned_model', type = str, default = '/data/group_data/dei-group/hdiddee/neg')
    parser.add_argument('--alpha', default=1, type=float)
    args = parser.parse_args()
    
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
    interpolated_state_dict = interpolate(delta_weights_pos, delta_weights_neg, alpha=0.5)
    del delta_weights_neg
    del delta_weights_pos
    print('Computed interpolated weight dict')

    # add base model weights 
    dummy = AutoModelForCausalLM.from_pretrained(args.base_model, device_map = 'cuda', torch_dtype = torch.bfloat16).eval() # The base model is always LLaMa so only the merged version has to be used
    dummy.load_state_dict(interpolated_state_dict) 
    del base_model
    
    # add to the base_model 
    # model trained on only positive samples, model trained on only negative samples - at test time if you give it sentimented sentence - it will mostly give you an aligned class label
    # but if we interpolate between the weights - we can recover the models ability to predict the neutral class. benefits are sample efficiency and controllability. 
    # Using only positive and negative samples, can you improve performance on the minority class. (especially since neutral sentences are often a minority)

    ds = load_dataset("Yelp/yelp_review_full")['test'].select(range(10))
    inputs = ds['text']
    finetuning_prefix = """Given a sentence, assign an appropriate sentiment to it: """
    finetuning_suffix = "### Sentiment Class:"
 
    test_inputs = [f'{finetuning_prefix} {input} {finetuning_suffix}' for input in inputs]
    
    tokenizer = AutoTokenizer.from_pretrained('/data/user_data/hdiddee/llama_models/llama_checkpoint/')
    tokenizer.pad_token = tokenizer.eos_token
    # finetuning_prefix = """Given a sentence, assign an appropriate sentiment to it: """
    # finetuning_suffix = "### Sentiment Class:"
    # def preproc(row):
    #     return tokenizer(f'{finetuning_prefix}: {row["text"]} {finetuning_suffix}', 
    #                     max_length=512, 
    #                     truncation=True, 
    #                     return_tensors= 'pt')

    # ds = ds.map(preproc, num_proc = 4)

    for row in tqdm.tqdm(test_inputs):
        local_batch = tokenizer(row, return_tensors = 'pt', truncation=True, padding='max_length', max_length = 256)# Has to be enough to include the prompt as well as the response. 
        with torch.no_grad():
            print(f'{row}')
            intermediate_outputs = dummy.generate(input_ids = local_batch.input_ids.cuda(), attention_mask = local_batch.attention_mask.cuda(),  max_new_tokens = 10, pad_token_id = tokenizer.eos_token_id)
            predictions = tokenizer.batch_decode(intermediate_outputs[:, local_batch.input_ids.shape[-1]:], skip_special_tokens=True)
            print(predictions)