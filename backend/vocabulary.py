"""
Vocabulary class for tokenization and encoding/decoding text.
"""

import nltk
from collections import Counter

class Vocabulary:
    """
    A vocabulary object that maps tokens to indices and vice versa.
    Includes special tokens for padding, unknown words, start, and end of sequence.
    """
    
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        
        # Add special tokens
        self.add_word('<pad>')
        self.add_word('<unk>')
        self.add_word('<sos>')
        self.add_word('<eos>')
    
    def add_word(self, word):
        """Add a word to the vocabulary if it doesn't exist"""
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def __len__(self):
        """Return the size of the vocabulary"""
        return len(self.word2idx)
    
    def __call__(self, word):
        """Get index of a word, return <unk> index if not found"""
        return self.word2idx.get(word, self.word2idx['<unk>'])
    
    def encode(self, text, max_length=None):
        """
        Convert text to list of indices
        Args:
            text: Input text string
            max_length: Maximum sequence length (optional)
        Returns:
            List of token indices
        """
        # Tokenize
        tokens = nltk.word_tokenize(text.lower())
        
        # Convert to indices
        indices = [self(token) for token in tokens]
        
        # Add EOS token
        indices.append(self.word2idx['<eos>'])
        
        # Truncate or pad if max_length is specified
        if max_length is not None:
            if len(indices) > max_length:
                indices = indices[:max_length-1] + [self.word2idx['<eos>']]
            else:
                indices = indices + [self.word2idx['<pad>']] * (max_length - len(indices))
        
        return indices
    
    def decode(self, indices, skip_special=True):
        """
        Convert list of indices back to text
        Args:
            indices: List of token indices
            skip_special: Whether to skip special tokens in output
        Returns:
            Decoded text string
        """
        special_tokens = {'<pad>', '<unk>', '<sos>', '<eos>'}
        tokens = []
        
        for idx in indices:
            if isinstance(idx, int):
                word = self.idx2word.get(idx, '<unk>')
            else:
                word = self.idx2word.get(idx.item(), '<unk>')
            
            if skip_special and word in special_tokens:
                if word == '<eos>':  # Stop at end of sequence
                    break
                continue
            
            tokens.append(word)
        
        return ' '.join(tokens)
    
    @staticmethod
    def build_from_texts(texts, min_freq=1):
        """
        Build vocabulary from a list of texts
        Args:
            texts: List of text strings
            min_freq: Minimum frequency for a word to be included
        Returns:
            Vocabulary object
        """
        vocab = Vocabulary()
        
        # Count word frequencies
        word_freq = Counter()
        for text in texts:
            tokens = nltk.word_tokenize(text.lower())
            word_freq.update(tokens)
        
        # Add words that meet minimum frequency
        for word, freq in word_freq.items():
            if freq >= min_freq:
                vocab.add_word(word)
        
        return vocab
