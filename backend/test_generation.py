"""
Test the ML models with sample input to verify generation quality.
"""

import torch
import pickle
import sys
sys.path.append('.')

from vocabulary import Vocabulary
from models.transformer_model import TransformerHookGenerator
from models.cnn_model import CNNSummarizer

# Load vocabulary
print("Loading vocabulary...")
vocab_path = 'saved_models/vocab.pkl'
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)
print(f"Vocabulary size: {len(vocab.word2idx)}")

# Setup device
device = torch.device('cpu')

# Load Transformer
print("\nLoading Transformer model...")
transformer_model = TransformerHookGenerator(
    vocab_size=len(vocab.word2idx),
    d_model=256,
    nhead=8,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dim_feedforward=512,
    dropout=0.1
).to(device)

transformer_checkpoint = torch.load('saved_models/transformer_hook_best.pth', map_location=device)
transformer_model.load_state_dict(transformer_checkpoint['model_state_dict'])
transformer_model.eval()
print("✓ Transformer loaded")

# Load CNN
print("\nLoading CNN model...")
cnn_model = CNNSummarizer(
    vocab_size=len(vocab.word2idx),
    embedding_dim=128,
    hidden_dim=256,
    num_filters=100,
    filter_sizes=[3, 4, 5],
    dropout=0.3
).to(device)

cnn_checkpoint = torch.load('saved_models/cnn_concise_best.pth', map_location=device)
cnn_model.load_state_dict(cnn_checkpoint['model_state_dict'])
cnn_model.eval()
print("✓ CNN loaded")

# Test with sample input
test_draft = "I won a contest at JP Morgan"
print(f"\n{'=' * 60}")
print(f"Testing with: '{test_draft}'")
print(f"{'=' * 60}")

import nltk
nltk.download('punkt', quiet=True)

def text_to_tensor(text):
    tokens = nltk.word_tokenize(text.lower())
    indices = [vocab.word2idx.get(token, vocab.word2idx["<UNK>"]) for token in tokens]
    return torch.tensor(indices).unsqueeze(0).to(device)

def tensor_to_text(tensor):
    if tensor.dim() == 2:
        tensor = tensor.squeeze(0)
    indices = tensor.tolist()
    words = []
    for idx in indices:
        if idx in [vocab.word2idx["<PAD>"], vocab.word2idx["<SOS>"], vocab.word2idx["<EOS>"]]:
            continue
        word = vocab.idx2word.get(idx, "<UNK>")
        if word != "<UNK>":
            words.append(word)
    return " ".join(words)

# Test Transformer
print("\n--- TRANSFORMER OUTPUT (Engaging Hook) ---")
src = text_to_tensor(test_draft)
with torch.no_grad():
    # Test with different temperatures
    for temp in [0.7, 0.9, 1.1]:
        generated = transformer_model.generate(
            src,
            max_length=80,
            start_token=vocab.word2idx["<SOS>"],
            end_token=vocab.word2idx["<EOS>"],
            temperature=temp,
            top_k=50
        )
        output = tensor_to_text(generated)
        print(f"\nTemperature {temp}: {output}")

# Test CNN
print("\n\n--- CNN OUTPUT (Concise Version) ---")
with torch.no_grad():
    # Test with different temperatures
    for temp in [0.7, 0.9, 1.1]:
        generated = cnn_model.generate(
            src,
            max_length=60,
            start_token=vocab.word2idx["<SOS>"],
            end_token=vocab.word2idx["<EOS>"],
            temperature=temp,
            top_k=40
        )
        output = tensor_to_text(generated)
        print(f"\nTemperature {temp}: {output}")

print(f"\n{'=' * 60}")
print("Test complete!")
print(f"{'=' * 60}")
