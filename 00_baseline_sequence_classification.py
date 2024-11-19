import argparse
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from torch.nn import CrossEntropyLoss
import os
import json
import shutil
import numpy as np
from datetime import datetime


PREFIX_CHECKPOINT_DIR = "checkpoint"
class CustomTrainer(Trainer):
    """
    Custom trainer class that saves model weights and metrics after each epoch
    """
    def __init__(self, *args, **kwargs):
        self.weights_dir = kwargs.pop('weights_dir', None)
        self.metrics_history = {
            'train': [],
            'eval': [],
            'epoch_times': []
        }
        self.epoch_start_time = None
        super().__init__(*args, **kwargs)

    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Save checkpoint, epoch weights, and metrics
        """
        # Calculate epoch duration
        if self.epoch_start_time:
            epoch_duration = datetime.now() - self.epoch_start_time
            self.metrics_history['epoch_times'].append(str(epoch_duration))
        
        # First, save the checkpoint normally
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        
        # Save the model checkpoint
        self.save_model(output_dir)
        
        # Save epoch weights and metrics separately
        if self.weights_dir:
            epoch = int(self.state.epoch)
            epoch_dir = os.path.join(self.weights_dir, f"epoch_{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)
            
            # Save model
            self.save_model(epoch_dir)
            
            # Save optimizer and scheduler states
            torch.save(self.optimizer.state_dict(), 
                      os.path.join(epoch_dir, "optimizer.pt"))
            # if self.scheduler is not None:
            #     torch.save(self.scheduler.state_dict(), 
            #               os.path.join(epoch_dir, "scheduler.pt"))
            
            # Evaluate on train set
            train_metrics = self.evaluate(self.train_dataset)
            self.metrics_history['train'].append(train_metrics)
            
            # Evaluate on test set
            eval_metrics = self.evaluate(self.eval_dataset)
            self.metrics_history['eval'].append(eval_metrics)
            
            # Save metrics
            metrics_file = os.path.join(epoch_dir, "metrics.json")
            epoch_metrics = {
                'train': train_metrics,
                'eval': eval_metrics,
                'epoch': epoch,
                'global_step': self.state.global_step,
                'epoch_duration': str(epoch_duration) if self.epoch_start_time else None
            }
            with open(metrics_file, 'w') as f:
                json.dump(epoch_metrics, f, indent=4)
            
            # Save complete training history
            history_file = os.path.join(self.weights_dir, "training_history.json")
            with open(history_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=4)
        
        # Reset epoch start time for next epoch
        self.epoch_start_time = datetime.now()
        
        return output_dir

    def train(self, *args, **kwargs):
        """
        Override train to track epoch start time
        """
        self.epoch_start_time = datetime.now()
        return super().train(*args, **kwargs)

def create_classification_model(base_model_name, num_labels=5):
    """
    Create a sequence classification model from a base model
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=num_labels
    )
    return model

def preprocess_function(examples, tokenizer):
    """
    Tokenize the texts and prepare them for sequence classification
    Include labels in the output
    """
    # Tokenize the texts
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding='max_length',
    )
    
    # Add labels to the tokenized output
    tokenized['labels'] = examples['label']
    
    return tokenized

def compute_metrics(eval_pred):
    """
    Compute classification metrics
    """
    predictions, labels = eval_pred
    predictions = predictions.argmax(-1)
    
    # Calculate overall accuracy
    correct = (predictions == labels).sum()
    total = len(labels)
    accuracy = float(correct / total)
    
    # Calculate per-class metrics
    per_class_metrics = {}
    for class_idx in range(5):
        class_mask = (labels == class_idx)
        if class_mask.sum() > 0:
            # Accuracy
            class_correct = ((predictions == labels) & class_mask).sum()
            class_total = class_mask.sum()
            class_accuracy = float(class_correct / class_total)
            
            # Precision
            predicted_class = (predictions == class_idx)
            true_positives = ((predictions == class_idx) & (labels == class_idx)).sum()
            precision = float(true_positives / predicted_class.sum()) if predicted_class.sum() > 0 else 0
            
            # Recall
            recall = float(true_positives / class_mask.sum()) if class_mask.sum() > 0 else 0
            
            # F1 Score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class_metrics.update({
                f'class_{class_idx}_accuracy': class_accuracy,
                f'class_{class_idx}_precision': precision,
                f'class_{class_idx}_recall': recall,
                f'class_{class_idx}_f1': f1
            })
    
    metrics = {
        "accuracy": accuracy,
        **per_class_metrics
    }
    
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default="distilbert/distilgpt2")
    parser.add_argument('--output_dir', type=str, default='data/big_classification_model')
    parser.add_argument('--weights_dir', type=str, default='data/big_epoch_weights')
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.weights_dir, exist_ok=True)
    
    # Load dataset
    ds = load_dataset("Yelp/yelp_review_full")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # select only 100 examples from each train and test
    # ds["train"] = ds["train"].select(list(range(100)))
    ds["test"] = ds["test"].select(list(range(512)))
    
    # Preprocess dataset
    tokenized_ds = ds.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=ds["train"].column_names
    )
    
    # Create classification model
    model = create_classification_model(args.base_model)
    model.config.pad_token_id = tokenizer.pad_token_id


    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="wandb",
        save_total_limit=None,  # Keep all checkpoints,
    )
    
    # Initialize custom trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        compute_metrics=compute_metrics,
        weights_dir=args.weights_dir
    )
    
    # Train the model
    trainer.train()
    
    # Save the final model and tokenizer
    final_model_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)