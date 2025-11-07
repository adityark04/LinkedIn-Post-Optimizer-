"""
Quick CNN retraining script for vocabulary size mismatch
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import pickle
from tqdm import tqdm
import sys
sys.path.append('.')

from models.cnn_model import CNNSummarizer
from train_models import LinkedInDataset

print("Quick CNN Model Retraining")
print("="*60)

# Load vocabulary
with open('saved_models/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

# Load datasets
with open('data/train_data.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

with open('data/test_data.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# Create datasets for CNN (concise generation)
train_dataset_concise = LinkedInDataset(train_data, vocab, task='concise', tgt_max_len=25)
test_dataset_concise = LinkedInDataset(test_data, vocab, task='concise', tgt_max_len=25)

train_loader_concise = DataLoader(train_dataset_concise, batch_size=4, shuffle=True)
test_loader_concise = DataLoader(test_dataset_concise, batch_size=4, shuffle=False)

print(f"Training samples: {len(train_dataset_concise)}")
print(f"Test samples: {len(test_dataset_concise)}")

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = CNNSummarizer(vocab_size=vocab_size, embed_dim=128, hidden_dim=256, num_filters=128, kernel_sizes=[3, 4, 5])
model = model.to(device)

# Training setup
criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'], label_smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nTraining CNN Model (Quick - 10 epochs)...")
print("="*60)

best_val_loss = float('inf')

for epoch in range(10):
    # Training
    model.train()
    total_loss = 0
    train_bar = tqdm(train_loader_concise, desc=f"Epoch {epoch+1}/10 [Train]")
    
    # Teacher forcing schedule
    progress = epoch / 10
    tf_ratio = max(0.3, 0.9 - progress * 0.6)
    
    for batch in train_bar:
        src, tgt = batch
        src, tgt = src.to(device), tgt.to(device)
        
        optimizer.zero_grad()
        output = model(src, tgt, teacher_forcing_ratio=tf_ratio)
        
        output = output.view(-1, vocab_size)
        tgt = tgt.view(-1)
        
        loss = criterion(output, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        train_bar.set_postfix(loss=loss.item(), tf=f'{tf_ratio:.2f}')
    
    # Validation
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for batch in test_loader_concise:
            src, tgt = batch
            src, tgt = src.to(device), tgt.to(device)
            
            output = model(src, tgt, teacher_forcing_ratio=0.0)
            output = output.view(-1, vocab_size)
            tgt = tgt.view(-1)
            
            loss = criterion(output, tgt)
            val_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_loader_concise)
    avg_val_loss = val_loss / len(test_loader_concise)
    
    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'saved_models/cnn_concise_best.pth')
        print(f"✓ Saved best model with val loss: {best_val_loss:.4f}")

print("\n" + "="*60)
print(f"CNN retraining complete!")
print(f"Best validation loss: {best_val_loss:.4f}")
print("Model saved to: saved_models/cnn_concise_best.pth")
print("="*60)
