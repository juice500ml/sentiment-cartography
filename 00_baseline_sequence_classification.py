import argparse
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from torch.nn import CrossEntropyLoss
import os
import json, sys
import shutil
import numpy as np
from datetime import datetime


PREFIX_CHECKPOINT_DIR = "checkpoint"
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.weights_dir = kwargs.pop('weights_dir', None)
        self.eval_steps = kwargs.pop('eval_steps', None)
        self.metrics_history = {
            'train': [], 'eval': [], 'step_times': []
        }
        self.step_start_time = None
        super().__init__(*args, **kwargs)

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Add the num_items_in_batch parameter with a default value of None
        if self.step_start_time is None:
            self.step_start_time = datetime.now()

        loss = super().training_step(model, inputs)

        if self.state.global_step % self.eval_steps == 0:
            self._save_checkpoint(model)

        return loss

    def _save_checkpoint(self, model, trial=None, metrics=None):
        step_duration = datetime.now() - self.step_start_time
        self.metrics_history['step_times'].append(str(step_duration))

        # Save checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
        self.save_model(output_dir)

        # Save weights and metrics
        if self.weights_dir:
            step_dir = os.path.join(self.weights_dir, f"step_{self.state.global_step}")
            os.makedirs(step_dir, exist_ok=True)
            
            # Save model
            self.save_model(step_dir)
            
            # Save optimizer state
            torch.save(self.optimizer.state_dict(), os.path.join(step_dir, "optimizer.pt"))

            # Evaluate on train and test sets
            train_metrics = self.evaluate(self.train_dataset)
            eval_metrics = self.evaluate(self.eval_dataset)
            
            self.metrics_history['train'].append(train_metrics)
            self.metrics_history['eval'].append(eval_metrics)

            # Save metrics
            step_metrics = {
                'train': train_metrics,
                'eval': eval_metrics,
                'step': self.state.global_step,
                'step_duration': str(step_duration)
            }
            with open(os.path.join(step_dir, "metrics.json"), 'w') as f:
                json.dump(step_metrics, f, indent=4)

            # Save complete training history
            with open(os.path.join(self.weights_dir, "training_history.json"), 'w') as f:
                json.dump(self.metrics_history, f, indent=4)

        self.step_start_time = datetime.now()
        return output_dir

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
    parser.add_argument('--output_dir', type=str, default='data/large_classification_model')
    parser.add_argument('--weights_dir', type=str, default='data/large_epoch_weights')
    parser.add_argument('--num_labels', type=int, default=5)
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

    # Preprocess dataset
    tokenized_ds = ds.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=ds["train"].column_names
    )
    if args.num_labels == 2:
        tokenized_ds = tokenized_ds.filter(lambda x: x["label"] in (0, 4))

    # Create classification model
    model = create_classification_model(args.base_model, num_labels=args.num_labels)
    model.config.pad_token_id = tokenizer.pad_token_id

    PER_DEVICE_BATCH_SIZE = 128
    # Calculate the number of steps per epoch
    steps_per_epoch = len(tokenized_ds["train"]) // PER_DEVICE_BATCH_SIZE
    eval_steps = steps_per_epoch // 10  # Evaluate every 1/10th of an epoch

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Evaluate every {eval_steps} steps")
    print("Len of train dataset: ", len(tokenized_ds["train"]))
    print("Len of test dataset: ", len(tokenized_ds["test"]))
    # sys.exit(0)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="wandb",
        save_total_limit=2,  # Keep all checkpoints
    )
    # Initialize custom trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        compute_metrics=compute_metrics,
        weights_dir=args.weights_dir,
        eval_steps=eval_steps
)
    # Train the model
    trainer.train()
    
    # Save the final model and tokenizer
    final_model_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)