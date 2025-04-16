import os
import torch
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BertTokenizer, BertModel, T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim 
from tqdm import tqdm

app = Flask(__name__)
CORS(app)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pre-trained models
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)

# Constants
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 64
BATCH_SIZE = 4  # Small batch size for systems with limited RAM

class ArgumentExtractor:
    """Extract key arguments from text using BERT embeddings and a rule-based approach"""
    
    def __init__(self, bert_model, bert_tokenizer, device):
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
        self.device = device
        
    def extract_arguments(self, text):
        # Split text into sentences (simplified approach)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if not sentences:
            return []
        
        # Get BERT embeddings for each sentence
        embeddings = []
        for sentence in sentences:
            inputs = self.bert_tokenizer(sentence, return_tensors="pt", 
                                        truncation=True, max_length=MAX_INPUT_LENGTH).to(self.device)
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            
            # Use CLS token embedding as sentence representation
            embeddings.append(outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy())
        
        embeddings = np.array(embeddings)
        
        # Find key arguments based on centrality in embedding space
        # 1. Calculate centroid
        centroid = np.mean(embeddings, axis=0)
        
        # 2. Calculate distance from centroid
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        
        # 3. Sort sentences by closeness to centroid and presence of argument markers
        sentence_scores = []
        argument_markers = ['because', 'therefore', 'thus', 'since', 'however', 
                           'although', 'despite', 'consequently', 'moreover']
        
        for i, sentence in enumerate(sentences):
            # Base score is inverse of distance to centroid
            score = 1.0 / (distances[i] + 1.0)
            
            # Boost score for sentences with argument markers
            lower_sentence = sentence.lower()
            for marker in argument_markers:
                if marker in lower_sentence:
                    score *= 1.5
                    break
                    
            # Boost score for medium-length sentences (likely to contain complete arguments)
            words = len(sentence.split())
            if 10 <= words <= 30:
                score *= 1.2
                
            sentence_scores.append((i, score))
        
        # Sort by score and take top 3 key arguments
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        key_argument_indices = [idx for idx, _ in sentence_scores[:3]]
        key_arguments = [sentences[i] for i in sorted(key_argument_indices)]
        
        return key_arguments

class TitleGenerator:
    """Generate titles using T5 model based on extracted arguments"""
    
    def __init__(self, t5_model, t5_tokenizer, device):
        self.t5_model = t5_model
        self.t5_tokenizer = t5_tokenizer
        self.device = device
    
    def generate_title(self, arguments, original_text=None):
        # Prepare input text from arguments
        if not arguments:
            if original_text:
                # Fallback to first 100 chars if no arguments extracted
                input_text = original_text[:100]
            else:
                return "Unable to generate title"
        else:
            # Combine extracted arguments
            input_text = " ".join(arguments)
        
        # Prepare input for T5 (T5 expects "summarize: " prefix for summarization)
        input_text = "summarize: " + input_text
        
        # Tokenize input
        inputs = self.t5_tokenizer(input_text, return_tensors="pt", 
                                  truncation=True, max_length=MAX_INPUT_LENGTH).to(self.device)
        
        # Generate title
        with torch.no_grad():
            outputs = self.t5_model.generate(
                inputs.input_ids,
                max_length=MAX_TARGET_LENGTH,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        # Decode and return title
        title = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up title (capitalize first letter, ensure it ends with proper punctuation)
        title = title.strip()
        if title:
            title = title[0].upper() + title[1:]
            if not any(title.endswith(p) for p in ['.', '!', '?']):
                title += '.'
                
        return title

# Initialize components
argument_extractor = ArgumentExtractor(bert_model, bert_tokenizer, device)
title_generator = TitleGenerator(t5_model, t5_tokenizer, device)

# Custom Dataset for training
class TitleGenerationDataset(Dataset):
    def __init__(self, texts, titles, bert_tokenizer, t5_tokenizer):
        self.texts = texts
        self.titles = titles
        self.bert_tokenizer = bert_tokenizer
        self.t5_tokenizer = t5_tokenizer
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        title = self.titles[idx]
        
        # Tokenize input text (with prefix for T5)
        input_text = "summarize: " + text
        input_encoding = self.t5_tokenizer(
            input_text,
            max_length=MAX_INPUT_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize target title
        target_encoding = self.t5_tokenizer(
            title,
            max_length=MAX_TARGET_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Convert label token IDs to have -100 for padding tokens
        target_ids = target_encoding["input_ids"].squeeze()
        target_ids[target_ids == self.t5_tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": target_ids
        }

# Training function
def train_model(csv_path, epochs=3, learning_rate=3e-5, save_path="./trained_model", continue_training=False):
    # Load and prepare data
    print("Loading data from CSV...")
    df = pd.read_csv(csv_path)
    
    # Ensure CSV has required columns
    required_columns = ["text", "title"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain these columns: {required_columns}")
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    # Create datasets
    train_dataset = TitleGenerationDataset(
        train_df["text"].tolist(),
        train_df["title"].tolist(),
        bert_tokenizer,
        t5_tokenizer
    )
    
    val_dataset = TitleGenerationDataset(
        val_df["text"].tolist(),
        val_df["title"].tolist(),
        bert_tokenizer,
        t5_tokenizer
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE
    )
    
    # Load base model for fine-tuning
    model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
    
    # Set up optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            train_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avg_train_loss = train_loss / len(train_dataloader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                # Move batch to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        checkpoint_dir = os.path.join(save_path, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-epoch-{epoch+1}")
        model.save_pretrained(checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            # Create directory if it doesn't exist
            os.makedirs(save_path, exist_ok=True)
            
            # Save model and tokenizer
            model.save_pretrained(save_path)
            t5_tokenizer.save_pretrained(save_path)
            
            print(f"Model saved to {save_path}")
    
    # If continuing training, use a lower learning rate
    if continue_training:
        print("Continuing training from existing model...")
        learning_rate *= 0.5  # Reduce learning rate for fine-tuning
    
    print("Training complete!")
    return model

# Load fine-tuned model if available
optimized_model_path = "./optimized_model"
trained_model_path = "./trained_model"

if os.path.exists(optimized_model_path):
    print("Loading optimized model...")
    t5_model = T5ForConditionalGeneration.from_pretrained(optimized_model_path).to(device)
    t5_tokenizer = T5Tokenizer.from_pretrained(optimized_model_path)
elif os.path.exists(trained_model_path):
    print("Loading trained model...")
    t5_model = T5ForConditionalGeneration.from_pretrained(trained_model_path).to(device)
    t5_tokenizer = T5Tokenizer.from_pretrained(trained_model_path)

title_generator.t5_model = t5_model

@app.route('/generate-title', methods=['POST'])
def generate_title_api():
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        text = data['text']
        
        # Extract arguments from text
        arguments = argument_extractor.extract_arguments(text)
        
        # Generate title
        title = title_generator.generate_title(arguments, original_text=text)
        
        return jsonify({
            'title': title,
            'arguments': arguments
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train_api():
    data = request.get_json()
    
    if not data or 'csv_path' not in data:
        return jsonify({'error': 'No CSV path provided'}), 400
    
    try:
        csv_path = data['csv_path']
        epochs = data.get('epochs', 3)
        learning_rate = data.get('learning_rate', 3e-5)
        continue_training = data.get('continue_training', False)
        
        # Train model
        model = train_model(csv_path, epochs, learning_rate, continue_training=continue_training)
        
        # Update global model
        global t5_model
        t5_model = model
        title_generator.t5_model = model
        
        return jsonify({'success': 'Model trained successfully'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)