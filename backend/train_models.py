"""
Training script for all three models: CNN, Transformer, and T5.
Handles data loading, training loops, and model saving.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
from tqdm import tqdm
import pickle

from models.cnn_model import CNNSummarizer
from models.transformer_model import TransformerHookGenerator
from transformers import T5ForConditionalGeneration, AutoTokenizer
from vocabulary import Vocabulary


class LinkedInDataset(Dataset):
    """Dataset for LinkedIn posts."""
    
    def __init__(self, data, vocab, src_max_len=100, tgt_max_len=50, task='hook'):
        """
        Args:
            data: List of dictionaries with draft, engaging_hook, concise_version, rephrased
            vocab: Vocabulary object
            src_max_len: Max length for source sequences
            tgt_max_len: Max length for target sequences
            task: 'hook' for Transformer, 'concise' for CNN, 'rephrase' for T5
        """
        self.data = data
        self.vocab = vocab
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len
        self.task = task
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Source is always the draft
        src = item.get('draft', item.get('text', ''))
        
        # Target depends on task
        if self.task == 'hook':
            # Support both old and new dataset schemas
            tgt = item.get('engaging_hook', item.get('hook', ''))
        elif self.task == 'concise':
            tgt = item.get('concise_version', item.get('concise', ''))
        else:  # rephrase
            tgt = item.get('rephrased', item.get('seo_rephrased', ''))
        
        # Encode
        src_indices = [self.vocab.word2idx["<SOS>"]] + self.vocab.encode(src, self.src_max_len - 2) + [self.vocab.word2idx["<EOS>"]]
        tgt_indices = [self.vocab.word2idx["<SOS>"]] + self.vocab.encode(tgt, self.tgt_max_len - 2) + [self.vocab.word2idx["<EOS>"]]
        
        # Pad if needed
        if len(src_indices) < self.src_max_len:
            src_indices = src_indices + [self.vocab.word2idx["<PAD>"]] * (self.src_max_len - len(src_indices))
        if len(tgt_indices) < self.tgt_max_len:
            tgt_indices = tgt_indices + [self.vocab.word2idx["<PAD>"]] * (self.tgt_max_len - len(tgt_indices))
        
        return {
            'src': torch.tensor(src_indices[:self.src_max_len], dtype=torch.long),
            'tgt': torch.tensor(tgt_indices[:self.tgt_max_len], dtype=torch.long),
            'src_text': src,
            'tgt_text': tgt
        }


def train_transformer(train_loader, val_loader, vocab, num_epochs=50, device='cpu'):
    """Train the Transformer model for hook generation."""
    print("\n" + "="*60)
    print("Training Transformer Model for Hook Generation")
    print("="*60)
    
    model = TransformerHookGenerator(
        vocab_size=len(vocab),
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.1
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_loss = float('inf')
    epochs_without_improve = 0
    early_stop_patience = 5
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in train_pbar:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(src, tgt[:, :-1])  # Don't include last token in input
            
            # Calculate loss
            output = output.reshape(-1, output.shape[-1])
            tgt = tgt[:, 1:].reshape(-1)  # Don't include <SOS> in target
            
            loss = criterion(output, tgt)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                src = batch['src'].to(device)
                tgt = batch['tgt'].to(device)
                
                output = model(src, tgt[:, :-1])
                output = output.reshape(-1, output.shape[-1])
                tgt = tgt[:, 1:].reshape(-1)
                
                loss = criterion(output, tgt)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save best model + early stopping tracking
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            epochs_without_improve = 0
            os.makedirs('saved_models', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'saved_models/transformer_hook_best.pth')
            print(f"✓ Saved best model with val loss: {best_val_loss:.4f}")
        else:
            epochs_without_improve += 1
        
        scheduler.step(avg_val_loss)
        
        # Early stop around epoch 8–10 if no improvement
        if epochs_without_improve >= early_stop_patience and epoch >= 7:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    return model


def train_cnn(train_loader, val_loader, vocab, num_epochs=50, device='cpu'):
    """Train the CNN model for concise generation."""
    print("\n" + "="*60)
    print("Training CNN Model for Concise Post Generation")
    print("="*60)
    
    model = CNNSummarizer(
        vocab_size=len(vocab),
        embedding_dim=128,
        hidden_dim=256,
        num_filters=100,
        filter_sizes=[3, 4, 5],
        dropout=0.3
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<PAD>"], label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Scheduled teacher forcing: start high, decay to 0.3
        tf_ratio = max(0.3, 0.9 - (0.6 * (epoch / max(1, num_epochs - 1))))
        
        # Training
        model.train()
        train_loss = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in train_pbar:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with scheduled teacher forcing
            output = model(src, tgt, teacher_forcing_ratio=tf_ratio)
            
            # Calculate loss
            output = output[:, 1:, :].reshape(-1, output.shape[-1])  # Skip first prediction
            tgt = tgt[:, 1:].reshape(-1)  # Skip <SOS>
            
            loss = criterion(output, tgt)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                src = batch['src'].to(device)
                tgt = batch['tgt'].to(device)
                
                output = model(src, tgt, teacher_forcing_ratio=0)
                output = output[:, 1:, :].reshape(-1, output.shape[-1])
                tgt = tgt[:, 1:].reshape(-1)
                
                loss = criterion(output, tgt)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        print(f"Epoch {epoch+1}: TF={tf_ratio:.2f} Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs('saved_models', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'saved_models/cnn_concise_best.pth')
            print(f"✓ Saved best model with val loss: {best_val_loss:.4f}")
        
        scheduler.step(avg_val_loss)
    
    return model


def train_t5(train_data, val_data, num_epochs=10, device='cpu'):
    """Fine-tune T5 model for rephrasing."""
    print("\n" + "="*60)
    print("Fine-tuning T5 Model for SEO-Optimized Rephrasing")
    print("="*60)
    
    # Load pre-trained T5-small
    model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)
    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    
    optimizer = optim.AdamW(model.parameters(), lr=3e-5)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        train_pbar = tqdm(train_data, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for item in train_pbar:
            # Prepare input and target
            input_text = "rephrase for LinkedIn: " + item['draft']
            target_text = item['rephrased']
            
            # Tokenize
            inputs = tokenizer(input_text, return_tensors='pt', padding=True, 
                             truncation=True, max_length=512).to(device)
            targets = tokenizer(target_text, return_tensors='pt', padding=True,
                              truncation=True, max_length=512).to(device)
            
            # Forward pass
            outputs = model(**inputs, labels=targets.input_ids)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_data)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for item in val_data:
                input_text = "rephrase for LinkedIn: " + item['draft']
                target_text = item['rephrased']
                
                inputs = tokenizer(input_text, return_tensors='pt', padding=True,
                                 truncation=True, max_length=512).to(device)
                targets = tokenizer(target_text, return_tensors='pt', padding=True,
                                  truncation=True, max_length=512).to(device)
                
                outputs = model(**inputs, labels=targets.input_ids)
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_data) if len(val_data) > 0 else 0
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs('saved_models/t5_rephrase', exist_ok=True)
            model.save_pretrained('saved_models/t5_rephrase')
            tokenizer.save_pretrained('saved_models/t5_rephrase')
            print(f"✓ Saved best model with val loss: {best_val_loss:.4f}")
    
    return model, tokenizer


def main():
    """Main training function."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    print("\nLoading datasets...")
    with open('data/train_data.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('data/test_data.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Testing samples: {len(test_data)}")
    
    # Build vocabulary
    print("\nBuilding vocabulary...")
    all_texts = []
    for item in train_data + test_data:
        draft = item.get('draft', item.get('text', ''))
        engaging = item.get('engaging_hook', item.get('hook', ''))
        concise = item.get('concise_version', item.get('concise', ''))
        rephrased = item.get('rephrased', item.get('seo_rephrased', ''))
        all_texts.extend([draft, engaging, concise, rephrased])
    
    vocab = Vocabulary(min_freq=1)
    vocab.build_vocabulary(all_texts)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Save vocabulary
    os.makedirs('saved_models', exist_ok=True)
    with open('saved_models/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    print("✓ Vocabulary saved")
    
    # Create datasets for Transformer (hook generation)
    print("\n" + "="*60)
    print("Preparing Transformer datasets...")
    train_dataset_hook = LinkedInDataset(train_data, vocab, task='hook')
    test_dataset_hook = LinkedInDataset(test_data, vocab, task='hook')
    
    train_loader_hook = DataLoader(train_dataset_hook, batch_size=4, shuffle=True)
    test_loader_hook = DataLoader(test_dataset_hook, batch_size=4, shuffle=False)
    
    # Train Transformer
    transformer_model = train_transformer(train_loader_hook, test_loader_hook, vocab, 
                                         num_epochs=30, device=device)
    
    # Create datasets for CNN (concise generation)
    print("\n" + "="*60)
    print("Preparing CNN datasets...")
    train_dataset_concise = LinkedInDataset(train_data, vocab, task='concise', tgt_max_len=25)
    test_dataset_concise = LinkedInDataset(test_data, vocab, task='concise', tgt_max_len=25)
    
    train_loader_concise = DataLoader(train_dataset_concise, batch_size=4, shuffle=True)
    test_loader_concise = DataLoader(test_dataset_concise, batch_size=4, shuffle=False)
    
    # Train CNN
    cnn_model = train_cnn(train_loader_concise, test_loader_concise, vocab,
                         num_epochs=30, device=device)
    
    # Train T5
    t5_model, t5_tokenizer = train_t5(train_data, test_data, num_epochs=5, device=device)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("Models saved in 'saved_models/' directory:")
    print("  - transformer_hook_best.pth")
    print("  - cnn_concise_best.pth")
    print("  - t5_rephrase/")
    print("  - vocab.pkl")


if __name__ == "__main__":
    main()
