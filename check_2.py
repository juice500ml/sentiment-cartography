#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[21]:


#!/usr/bin/env python
# coding: utf-8

import os
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# Parse arguments
# parser = argparse.ArgumentParser(description="Save inference results and model weights")
# parser.add_argument("--dir", type=str, required=True, help="Directory to save results and model weights")
# args = parser.parse_args()

# Create the directory if it doesn't exist
output_dir = "single_v2_dir_512_large"
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
dataset = load_dataset("yelp_review_full")

# in train set, only keep those instances which have their label has 0 or 4
dataset["train"] = dataset["train"].filter(lambda example: example["label"] in [0, 4])
# map 4 to 1
dataset["train"] = dataset["train"].map(lambda example: {"label": 0 if example["label"] == 0 else 1})
# in test set, map to 0 if closer to 0 and 1 if closer to 4
dataset["test"] = dataset["test"].map(lambda example: {"label": 0 if example["label"] < 2 else 1})

# Select subsets of the data
train_subset = dataset["train"].shuffle(seed=42).select(range(100000))
# test_subset = dataset["test"].shuffle(seed=42).select(range(10))
test_subset = dataset["test"].shuffle(seed=42)

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")

# Set the padding token
tokenizer.pad_token = tokenizer.eos_token

# Define the tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# Tokenize the selected subsets
tokenized_train = train_subset.map(tokenize_function, batched=True)
tokenized_test = test_subset.map(tokenize_function, batched=True)

# Set the format of the datasets to PyTorch
tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Initialize the model
model = GPT2ForSequenceClassification.from_pretrained("distilgpt2", num_labels=2)
model.config.pad_token_id = model.config.eos_token_id

# Define the data loaders
batch_size = 32
train_dataloader = DataLoader(tokenized_train, shuffle=True, batch_size=batch_size)
test_dataloader = DataLoader(tokenized_test, batch_size=batch_size)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
NUM_EPOCHS = 1
# Define optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
total_steps = len(train_dataloader) * NUM_EPOCHS  # 3 epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


# Training function
def train(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)

# Evaluation function
def evaluate(model, dataloader, device, save_results=False):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    results = []  # List to store inference results
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Save results for each review
            if save_results:
                decoded_reviews = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                for review, pred, true_label, logit in zip(decoded_reviews, preds, labels, logits):
                    results.append({
                        "review": review,
                        "predicted_label": pred.item(),
                        "ground_truth_label": true_label.item(),
                        "logit_scores": logit.cpu().tolist()  # Add logit scores for all labels
                    })
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return total_loss / len(dataloader), accuracy, f1, results

# Training loop
num_epochs = NUM_EPOCHS
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_loss = train(model, train_dataloader, optimizer, scheduler, device)
    print(f"Training loss: {train_loss:.4f}")
    
    eval_loss, accuracy, f1, _ = evaluate(model, test_dataloader, device)
    print(f"Evaluation loss: {eval_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print()

# Final evaluation and saving results
print("Final Evaluation:")
eval_loss, accuracy, f1, results = evaluate(model, test_dataloader, device, save_results=True)
print(f"Evaluation loss: {eval_loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")




# In[22]:


# Save inference results as a JSON file
results_path = os.path.join(output_dir, "inference_results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"Inference results saved to {results_path}")

# Save the model weights
model_path = os.path.join(output_dir, "model_weights")
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

print(f"Model weights saved to {model_path}")
