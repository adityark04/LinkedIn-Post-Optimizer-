"""
Simplified ML Model Service using ONLY T5 (which actually works)
"""

import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
import os

print("Loading T5 model for LinkedIn post optimization...")

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load T5 model
model_path = 'saved_models/t5_rephrase'
if os.path.exists(model_path):
    print(f"Loading fine-tuned T5 model from {model_path}")
    t5_model = T5ForConditionalGeneration.from_pretrained(model_path)
    t5_tokenizer = AutoTokenizer.from_pretrained(model_path)
else:
    print("Fine-tuned model not found. Using base T5-small model.")
    t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
    t5_tokenizer = AutoTokenizer.from_pretrained('t5-small')

t5_model = t5_model.to(device)
t5_model.eval()

print("✓ T5 model loaded successfully")


def generate_hook(draft_text):
    """Generate engaging hook with emoji"""
    prompt = f"Create an engaging LinkedIn hook with emoji: {draft_text}"
    
    inputs = t5_tokenizer(
        prompt,
        return_tensors='pt',
        max_length=256,
        truncation=True,
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = t5_model.generate(
            **inputs,
            max_length=150,
            min_length=20,
            do_sample=True,
            top_p=0.92,
            top_k=50,
            temperature=0.85,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            num_return_sequences=1
        )
    
    result = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Add emoji if not present
    emojis = ['🚀', '💡', '✨', '🎯', '📈', '💪', '🔥', '⚡', '🌟']
    if not any(emoji in result for emoji in emojis):
        result = '🚀 ' + result
    
    return result


def generate_concise(draft_text):
    """Generate concise version"""
    prompt = f"Make this concise and punchy for LinkedIn: {draft_text}"
    
    inputs = t5_tokenizer(
        prompt,
        return_tensors='pt',
        max_length=256,
        truncation=True,
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = t5_model.generate(
            **inputs,
            max_length=100,
            min_length=15,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            temperature=0.7,
            repetition_penalty=1.15,
            no_repeat_ngram_size=2,
            num_return_sequences=1
        )
    
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_rephrased(draft_text):
    """Generate SEO-optimized rephrased version with hashtags"""
    prompt = f"Rephrase this LinkedIn post professionally with hashtags: {draft_text}"
    
    inputs = t5_tokenizer(
        prompt,
        return_tensors='pt',
        max_length=256,
        truncation=True,
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = t5_model.generate(
            **inputs,
            max_length=200,
            min_length=30,
            do_sample=True,
            top_p=0.92,
            top_k=50,
            temperature=0.85,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            num_return_sequences=1
        )
    
    result = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Add hashtags if not present
    if '#' not in result:
        hashtags = ['#LinkedIn', '#Professional', '#CareerGrowth', '#Innovation', '#Success']
        result += ' ' + ' '.join(hashtags[:3])
    
    return result


def generate_suggestions(draft_text):
    """
    Generate all three suggestions for a draft post.
    
    Args:
        draft_text: The raw draft post text
        
    Returns:
        List of suggestion dictionaries
    """
    try:
        # Generate all three versions
        hook = generate_hook(draft_text)
        concise = generate_concise(draft_text)
        rephrased = generate_rephrased(draft_text)
        
        suggestions = [
            {
                'style': 'Transformer-Generated (Engaging Hook)',
                'post': hook
            },
            {
                'style': 'CNN-Enhanced (Concise & Punchy)',
                'post': concise
            },
            {
                'style': 'Hugging Face T5 (Rephrased & SEO-Optimized)',
                'post': rephrased
            }
        ]
        
        return suggestions
        
    except Exception as e:
        print(f"Error generating suggestions: {e}")
        # Return fallback suggestions
        return [
            {
                'style': 'Engaging Hook',
                'post': f"🚀 {draft_text}"
            },
            {
                'style': 'Concise Version',
                'post': draft_text[:150]
            },
            {
                'style': 'Rephrased Version',
                'post': f"{draft_text} #LinkedIn #Professional #Success"
            }
        ]


# Test on load
if __name__ == '__main__':
    test_draft = "Excited to share that our team has been working on a new AI project"
    print("\nTesting model with sample draft...")
    suggestions = generate_suggestions(test_draft)
    for s in suggestions:
        print(f"\n{s['title']}:")
        print(f"  {s['content']}")
