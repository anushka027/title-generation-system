import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Constants
MAX_TARGET_LENGTH = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def modify_model_for_better_titles():
    """
    Modify the model to generate more accurate titles by adjusting generation parameters
    """
    model_path = "./trained_model"
    if not os.path.exists(model_path):
        print("No trained model found. Please train the model first.")
        return False

    print(f"Loading model from {model_path}...")
    t5_model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    t5_tokenizer = T5Tokenizer.from_pretrained(model_path)
    
    # Create a modified generation configuration
    generation_config = t5_model.generation_config
    
    # Adjust generation parameters for better title generation
    generation_config.num_beams = 4  # Enable beam search with 4 beams
    generation_config.length_penalty = 1.5  # Values > 1.0 favor longer titles
    generation_config.repetition_penalty = 2.5  # Higher values discourage repetition
    generation_config.no_repeat_ngram_size = 3  # Prevent trigram repetition
    
    # Save the updated model
    modified_path = "./optimized_model"
    os.makedirs(modified_path, exist_ok=True)
    t5_model.save_pretrained(modified_path)
    t5_tokenizer.save_pretrained(modified_path)
    
    print(f"Optimized model saved to {modified_path}")
    print("To use this model, update the model_path in app.py to './optimized_model'")
    return True

if __name__ == "__main__":
    modify_model_for_better_titles()