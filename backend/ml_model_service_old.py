"""
ML Model Service - Loads and runs trained deep learning models.
Uses CNN, Transformer, and T5 models for LinkedIn post optimization.
"""

import torch
import pickle
import os
from transformers import T5ForConditionalGeneration, AutoTokenizer
from models.cnn_model import CNNSummarizer
from models.transformer_model import TransformerHookGenerator
from vocabulary import Vocabulary
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Global variables for loaded models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab = None
transformer_model = None
cnn_model = None
t5_model = None
t5_tokenizer = None

# Paths to saved models
VOCAB_PATH = 'saved_models/vocab.pkl'
TRANSFORMER_PATH = 'saved_models/transformer_hook_best.pth'
CNN_PATH = 'saved_models/cnn_concise_best.pth'
T5_PATH = 'saved_models/t5_rephrase'


def load_models():
    """Load all trained models once when the server starts."""
    global vocab, transformer_model, cnn_model, t5_model, t5_tokenizer
    
    print("Loading trained models...")
    
    try:
        # Load vocabulary
        if os.path.exists(VOCAB_PATH):
            with open(VOCAB_PATH, 'rb') as f:
                vocab = pickle.load(f)
            print(f"✓ Vocabulary loaded (size: {len(vocab)})")
        else:
            print(f"⚠ Vocabulary not found at {VOCAB_PATH}. Please train models first.")
            return False
        
        # Load Transformer model for hook generation
        if os.path.exists(TRANSFORMER_PATH):
            transformer_model = TransformerHookGenerator(
                vocab_size=len(vocab),
                d_model=256,
                nhead=8,
                num_encoder_layers=3,
                num_decoder_layers=3,
                dim_feedforward=512,
                dropout=0.1
            ).to(device)
            
            checkpoint = torch.load(TRANSFORMER_PATH, map_location=device)
            transformer_model.load_state_dict(checkpoint['model_state_dict'])
            transformer_model.eval()
            print(f"✓ Transformer model loaded")
        else:
            print(f"⚠ Transformer model not found at {TRANSFORMER_PATH}")
            return False
        
        # Load CNN model for concise generation
        if os.path.exists(CNN_PATH):
            cnn_model = CNNSummarizer(
                vocab_size=len(vocab),
                embedding_dim=128,
                hidden_dim=256,
                num_filters=100,
                filter_sizes=[3, 4, 5],
                dropout=0.3
            ).to(device)
            
            checkpoint = torch.load(CNN_PATH, map_location=device)
            cnn_model.load_state_dict(checkpoint['model_state_dict'])
            cnn_model.eval()
            print(f"✓ CNN model loaded")
        else:
            print(f"⚠ CNN model not found at {CNN_PATH}")
            return False
        
        # Load T5 model for rephrasing (optional)
        if os.path.exists(T5_PATH):
            try:
                t5_model = T5ForConditionalGeneration.from_pretrained(T5_PATH)
                t5_tokenizer = AutoTokenizer.from_pretrained(T5_PATH)
                t5_model.eval()
                print(f"✓ T5 model loaded")
            except Exception as e:
                print(f"⚠ T5 model loading failed: {e}")
                print(f"  Continuing without T5 model...")
                t5_model = None
                t5_tokenizer = None
        else:
            print(f"⚠ T5 model not found at {T5_PATH}")
            print(f"  Continuing with Transformer and CNN only...")
            print(f"  To add T5 support, run the training script to completion.")
            t5_model = None
            t5_tokenizer = None
        
        print("Core models loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return False


def text_to_tensor(text, vocab, max_length=100):
    """Convert text to tensor using vocabulary."""
    tokens = nltk.word_tokenize(text.lower())
    indices = [vocab.word2idx.get(token, vocab.word2idx["<UNK>"]) for token in tokens]
    
    # Add SOS and EOS
    indices = [vocab.word2idx["<SOS>"]] + indices + [vocab.word2idx["<EOS>"]]
    
    # Pad or truncate
    if len(indices) < max_length:
        indices = indices + [vocab.word2idx["<PAD>"]] * (max_length - len(indices))
    else:
        indices = indices[:max_length]
    
    return torch.tensor([indices], dtype=torch.long).to(device)


def tensor_to_text(tensor, vocab):
    """Convert tensor back to text using vocabulary."""
    indices = tensor.squeeze().cpu().tolist()
    words = []
    for idx in indices:
        word = vocab.idx2word.get(idx, "<UNK>")
        if word in ["<SOS>", "<PAD>"]:
            continue
        if word == "<EOS>":
            break
        words.append(word)
    return " ".join(words)


def generate_hook_with_transformer(draft: str) -> str:
    """Generate engaging hook using Transformer model with creative sampling."""
    try:
        # Ensure models are loaded
        if vocab is None or transformer_model is None:
            load_models()
        if vocab is None or transformer_model is None:
            return f"🎉 Excited to share: {draft}\n\n#Innovation #Success"

        # Convert draft to tensor
        src = text_to_tensor(draft, vocab, max_length=100)
        
        # Generate hook with creative parameters
        with torch.no_grad():
            generated = transformer_model.generate(
                src,
                max_length=80,  # Allow longer hooks
                start_token=vocab.word2idx["<SOS>"],
                end_token=vocab.word2idx["<EOS>"],
                temperature=0.85,  # Slightly lower for more coherence
                top_k=50,
                top_p=0.92,       # Nucleus sampling threshold
                repetition_penalty=1.12  # Light penalty to reduce loops
            )
        
        # Convert back to text
        hook = tensor_to_text(generated, vocab)
        
        # Clean up and format
        hook = hook.strip()
        if not hook or len(hook) < 10:  # If generation failed, use template
            return f"🎉 Excited to share: {draft}\n\nYour thoughts? #Innovation #Success"
        
        # Add hashtags if not present
        if "#" not in hook:
            hook += "\n\n#Innovation #Success"
        
        return hook
        
    except Exception as e:
        print(f"Error in Transformer generation: {e}")
        # Fallback with template
        return f"💡 Here's something worth sharing:\n\n{draft}\n\n#Innovation #Tech"


def generate_concise_with_cnn(draft: str) -> str:
    """Generate concise, punchy version using CNN model with diverse sampling."""
    try:
        if vocab is None or cnn_model is None:
            load_models()
        if vocab is None or cnn_model is None:
            return f"{draft}\n\n#Innovation #Tech"

        # Convert draft to tensor
        src = text_to_tensor(draft, vocab, max_length=100)
        
        # Generate concise version with improved sampling
        with torch.no_grad():
            generated = cnn_model.generate(
                src,
                max_length=60,
                start_token=vocab.word2idx["<SOS>"],
                end_token=vocab.word2idx["<EOS>"],
                temperature=0.85,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.1,
                min_length=8
            )
        
        # Convert back to text
        concise = tensor_to_text(generated, vocab)
        
        # Clean up
        concise = concise.strip()
        if not concise or len(concise) < 10:  # If generation failed
            return f"{draft}\n\n#Innovation #DeepLearning"
        
        # Add hashtags if missing
        if "#" not in concise:
            concise += "\n\n#Innovation #Tech"
        
        return concise
        
    except Exception as e:
        print(f"Error in CNN generation: {e}")
        # Fallback with some enhancement
        return f"{draft}\n\nYour thoughts? #Tech #Innovation"


def generate_rephrase_with_t5(draft: str) -> str:
    """Generate SEO-optimized rephrased version using T5 model."""
    
    # Check if T5 is available
    if t5_model is None or t5_tokenizer is None:
        # Fallback: use simple text transformation
        print("T5 model not available, using fallback rephrasing")
        words = draft.split()
        # Add professional framing
        rephrased = f"Excited to share: {draft}"
        
        # Add relevant hashtags based on keywords
        hashtags = []
        keywords = {
            'ai': '#AI',
            'machine learning': '#MachineLearning',
            'deep learning': '#DeepLearning',
            'data': '#DataScience',
            'tech': '#Technology',
            'innovation': '#Innovation',
            'project': '#TechProjects',
            'development': '#SoftwareDevelopment'
        }
        
        draft_lower = draft.lower()
        for keyword, hashtag in keywords.items():
            if keyword in draft_lower and hashtag not in hashtags:
                hashtags.append(hashtag)
        
        if not hashtags:
            hashtags = ['#Innovation', '#Technology']
        
        rephrased = f"{rephrased}\n\n{' '.join(hashtags[:3])}"
        return rephrased
    
    try:
        # Prepare input
        input_text = f"rephrase for LinkedIn: {draft}"
        
        # Tokenize
        inputs = t5_tokenizer(
            input_text,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=True
        ).to(device)
        
        # Generate
        with torch.no_grad():
            outputs = t5_model.generate(
                **inputs,
                max_length=200,
                do_sample=True,
                top_p=0.92,
                temperature=0.85,
                repetition_penalty=1.08,
                early_stopping=True
            )
        
        # Decode
        rephrased = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return rephrased
        
    except Exception as e:
        print(f"Error in T5 generation: {e}")
        # Fallback
        return f"Excited to share insights on this topic:\n\n{draft}\n\n#Innovation #Technology #AI"


def generate_suggestions(draft: str) -> list[dict]:
    """
    Generate three optimized LinkedIn post suggestions using trained DL models.

    Args:
        draft: The user's input string.

    Returns:
        A list of suggestion dictionaries in the format expected by the frontend.
    """
    print(f"Generating suggestions for draft: '{draft[:50]}...'")
    
    # Check if models are loaded
    if transformer_model is None or cnn_model is None or t5_model is None:
        success = load_models()
        if not success:
            return [{
                "style": "Error - Models Not Trained",
                "post": "Please train the models first by running: python train_models.py\n\nSee TRAINING_GUIDE.md for instructions."
            }]
    
    # Generate with each model
    transformer_post = generate_hook_with_transformer(draft)
    cnn_post = generate_concise_with_cnn(draft)
    t5_post = generate_rephrase_with_t5(draft)
    
    suggestions = [
        {
            "style": "Transformer-Generated (Engaging Hook)",
            "post": transformer_post
        },
        {
            "style": "CNN-Enhanced (Concise & Punchy)",
            "post": cnn_post
        },
        {
            "style": "Hugging Face T5 (Rephrased & SEO-Optimized)",
            "post": t5_post
        }
    ]
    
    return suggestions


# Load models when module is imported
load_models()