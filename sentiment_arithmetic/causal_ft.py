import wandb
from datasets import load_dataset
from dataclasses import dataclass, field
import torch
from transformers import (    
    HfArgumentParser,         
    TrainingArguments, 
    AutoTokenizer, 
    AutoModelForCausalLM,
    set_seed
)
from trl import SFTTrainer
from accelerate import PartialState

@dataclass 
class ModelArguments:
    model_type: str = field(default=None)
    model_name: str = field(default = None)
    cache_dir: str = field(default='/scratch')
    budget: int = field(default=9999999999999)
    train_file: str = field(default=None)
    max_source_length: str = field(default=512)   
    max_train_samples: int = field(default=None) 
    wandb_run_name: str = field(default=None)
    output_prediction_file: str = field(default='predictions.txt')
    previous_checkpoint: str = field(default=None)
    wandb_project_name: str = field(default=None)

def main():

    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()    
    wandb.init(project="Sentiment Vectors", entity="hdiddee", name = args.wandb_run_name, group=f'{args.wandb_run_name}_ddp')
    device_string = PartialState().process_index
    print(f'Device Map: {device_string}')
        
    if args.model_type == 'llama' or args.model_type=='llama2':
        model = AutoModelForCausalLM.from_pretrained(args.model_name, 
                                                     use_flash_attention_2 = True, 
                                                     device_map = {'':device_string}, 
                                                     torch_dtype=torch.bfloat16)
        model.config.use_cache = False
        model.config.pretraining_tp = 1 # Faster, more appromixate evaluation of the linear layers  
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
   

   
    data_files = {}
    if args.train_file is not None: 
        data_files["train"] = args.train_file
    raw_datasets = load_dataset('json', data_files=data_files, cache_dir=args.cache_dir)
    train_dataset = raw_datasets["train"]

    args.max_train_samples = min(args.budget, len(train_dataset))  # Setting max samples to training data budget 
    print(f'Budget set to: {args.max_train_samples}')
   
    def preprocess_train_function(examples):         
        model_inputs = {}
        inputs = [ex for ex in examples['input']]
        targets = [ex for ex in examples['target']]

        model_inputs['inputs'] = inputs
        model_inputs['labels'] = targets 
        return model_inputs
    
    
    def format_instruction(model_input):
            return f"""Given a sentence, assign an appropriate sentiment to it: {model_input['inputs']} \n 
            ### Sentiment Class: {model_input['labels']}"""
     
    if training_args.do_train:
        if args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        train_dataset = train_dataset.map(
        preprocess_train_function,
        batched=True,
        desc=f"Creating input and target training samples with budget {args.max_train_samples}",
        )
   
    set_seed(training_args.seed)
    training_args.ddp_find_unused_parameters = False
    
    trainer = SFTTrainer(
        model=model, 
        args=training_args,
        max_seq_length=args.max_source_length, 
        train_dataset=train_dataset, 
        packing = True, 
        tokenizer=tokenizer,
        formatting_func = format_instruction
    )
    

    if args.previous_checkpoint is not None: 
        print('Found a previous checkpoint - so resuming training from there ...')
        train_result = trainer.train(resume_from_checkpoint = args.previous_checkpoint)
    else: 
        train_result = trainer.train()
    trainer.save_model()  
    
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    wandb.finish()


if __name__ == "__main__":
    main()
