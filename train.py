# Import necessary libraries
import pandas as pd
import torch
from datasets import load_dataset
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,  # For generative language modeling
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from tqdm import tqdm

if __name__ == "__main__":
    # Set up data storage path
    storage = Path("data")

    # Initialize two separate models from the same base (DistilGPT2)
    model_card = "distilbert/distilgpt2"
    # Create two separate instances for positive and negative sentiment training
    model_pos = AutoModelForCausalLM.from_pretrained(model_card)
    model_neg = AutoModelForCausalLM.from_pretrained(model_card)

    # Load the Yelp review dataset
    ds = load_dataset("Yelp/yelp_review_full")
    # Initialize tokenizer for text preprocessing
    tokenizer = AutoTokenizer.from_pretrained(model_card)

    # Define preprocessing function for tokenization
    def preproc(row):
        # Tokenize text with a maximum length of 512 tokens
        return tokenizer(row["text"], max_length=512, truncation=True)

    # Filter and preprocess training data
    # Extract 4-star reviews for positive sentiment
    ds_train_pos = ds["train"].filter(lambda row: row["label"] == 4).map(
        preproc, 
        num_proc=4,  # Use 4 CPU cores for parallel processing
        remove_columns=["label"]
    )
    # Extract 0-star reviews for negative sentiment
    ds_train_neg = ds["train"].filter(lambda row: row["label"] == 0).map(
        preproc,
        num_proc=4,
        remove_columns=["label"]
    )

    print("Size of positive sentiment training dataset:", len(ds_train_pos))
    print("Size of negative sentiment training dataset:", len(ds_train_neg))

    # Set padding token same as end-of-sequence token
    tokenizer.pad_token = tokenizer.eos_token
    # Initialize data collator for batching
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Train positive sentiment model
    Trainer(
        args=TrainingArguments(
            output_dir=storage / "model_pos",  # Save model in data/model_pos
            num_train_epochs=1,  # Train for one epoch
            report_to="wandb",   # Log metrics to Weights & Biases
            run_name="positive_model_training_num_name", 
            do_train=True,
            per_device_train_batch_size=16,  # Process 16 samples per batch
        ),
        model=model_pos,
        train_dataset=ds_train_pos,
        data_collator=data_collator,
        tokenizer=tokenizer,
    ).train()

    # Train negative sentiment model
    Trainer(
        args=TrainingArguments(
            output_dir=storage / "model_neg",  # Save model in data/model_neg
            num_train_epochs=1,
            report_to="wandb",
            run_name="positive_model_training_num_name",
            do_train=True,
            per_device_train_batch_size=16,
        ),
        model=model_neg,
        train_dataset=ds_train_neg,
        data_collator=data_collator,
        tokenizer=tokenizer,
    ).train()