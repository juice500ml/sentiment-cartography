import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM
import torch

def compute_delta_weights(fine_tuned_model, base_model):
    """
    Compute the difference between fine-tuned model weights and base model weights.
    This captures the changes made during fine-tuning, essentially isolating the
    "sentiment direction" in the weight space.
    
    Args:
        fine_tuned_model: Model after sentiment-specific training
        base_model: Original pre-trained model before fine-tuning
    
    Returns:
        dict: Dictionary containing weight differences for each layer
    """
    delta_weights = {}
    for name, param in fine_tuned_model.named_parameters():
        # Move base model parameters to same device as fine-tuned model
        base_param = base_model.state_dict()[name].to(param.device)
        # Calculate the difference between fine-tuned and base weights
        delta_weights[name] = param.data - base_param.data
    return delta_weights

def init_sentiment_vector(delta_weights, base_model_card, identifier):
    """
    Initialize a new model with the computed sentiment direction.
    This creates a model that represents the isolated sentiment changes.
    
    Args:
        delta_weights: Weight differences computed from compute_delta_weights
        base_model_card: Name/path of the base model architecture
        identifier: Name for saving the resulting model
    """
    # Step 1: Create a new model with the same architecture
    new_model = AutoModelForCausalLM.from_pretrained(base_model_card)

    # Set up device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    new_model.to(device)
    base_model.to(device)

    # Step 2: Add the delta weights to the base model weights
    with torch.no_grad():  # Disable gradient computation
        for name, param in new_model.named_parameters():
            base_param = base_model.state_dict()[name].to(device)
            delta = delta_weights[name].to(device)
            # New weights = base weights + delta weights
            param.data = base_param.data + delta

    # Step 3: Save the resulting model
    save_path = Path(f"data/{identifier}")
    new_model.save_pretrained(save_path)


if __name__ == '__main__':
    # Set up command line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default="distilbert/distilgpt2",
                      help="Path or name of the base model")
    parser.add_argument('--pos_finetuned_model', type=str, default='data/model_pos/',
                      help="Path to positive sentiment fine-tuned model")
    parser.add_argument('--neg_finetuned_model', type=str, default='data/neg_pos/',
                      help="Path to negative sentiment fine-tuned model")
    args = parser.parse_args()
    
    # Load the base model
    base_model_card = args.base_model 
    base_model = AutoModelForCausalLM.from_pretrained(base_model_card)
    
    # Process positive sentiment model
    # 1. Load the positive fine-tuned model
    pos_model = AutoModelForCausalLM.from_pretrained(args.pos_finetuned_model)
    # 2. Compute the weight differences (positive sentiment direction)
    delta_weights_pos = compute_delta_weights(pos_model, base_model)
    # 3. Create and save a model representing positive sentiment changes
    init_sentiment_vector(delta_weights_pos, base_model_card=args.base_model, 
                         identifier='pos_vector')

    # Process negative sentiment model
    # 1. Load the negative fine-tuned model
    neg_model = AutoModelForCausalLM.from_pretrained(args.neg_finetuned_model)
    # 2. Compute the weight differences (negative sentiment direction)
    delta_weights_neg = compute_delta_weights(neg_model, base_model)
    # 3. Create and save a model representing negative sentiment changes
    init_sentiment_vector(delta_weights_neg, base_model_card=args.base_model, 
                         identifier='neg_vector')