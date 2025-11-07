"""
Custom Transformer Model for Engaging Hook Generation.
Generates attention-grabbing opening lines for LinkedIn posts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model."""
    
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding to input."""
        pe_buf = getattr(self, 'pe')  # type: ignore[attr-defined]
        x = x + pe_buf[:, :x.size(1), :]
        return self.dropout(x)


class TransformerHookGenerator(nn.Module):
    """
    Transformer model for generating engaging hooks from draft posts.
    Uses encoder-decoder architecture with multi-head attention.
    """
    
    def __init__(self, vocab_size, d_model=256, nhead=8, num_encoder_layers=3,
                 num_decoder_layers=3, dim_feedforward=512, dropout=0.1, max_len=512):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Dimension of model embeddings
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            max_len: Maximum sequence length
        """
        super(TransformerHookGenerator, self).__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.src_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layer
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.tgt_embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)
    
    def generate_square_subsequent_mask(self, sz):
        """Generate mask for decoder to prevent attention to future tokens."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask
    
    def create_padding_mask(self, seq, pad_idx=0):
        """Create mask for padding tokens."""
        return (seq == pad_idx)
    
    def forward(self, src, tgt):
        """
        Forward pass for training.
        
        Args:
            src: Source sequence (batch_size, src_len)
            tgt: Target sequence (batch_size, tgt_len)
            
        Returns:
            Output logits (batch_size, tgt_len, vocab_size)
        """
        # Create masks
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(src.device)
        src_padding_mask = self.create_padding_mask(src)
        tgt_padding_mask = self.create_padding_mask(tgt)
        
        # Embeddings with positional encoding
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        
        # Transformer forward pass
        output = self.transformer(
            src_emb, 
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        # Project to vocabulary
        logits = self.fc_out(output)
        
        return logits
    
    def generate(self, src, max_length=60, start_token=1, end_token=2,
                 temperature=0.9, top_k=50, top_p=0.92, repetition_penalty=1.12):
        """
        Generate hook using improved sampling for creativity.
        
        Args:
            src: Source sequence (batch_size, src_len)
            max_length: Maximum length of generated sequence
            start_token: Start token ID
            end_token: End token ID
            temperature: Sampling temperature (higher = more creative)
            top_k: Number of top tokens to sample from
            
        Returns:
            Generated sequence (batch_size, generated_len)
        """
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)
            device = src.device
            
            # Encode source
            src_padding_mask = self.create_padding_mask(src)
            src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
            memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)
            
            # Initialize decoder input with start token
            generated = torch.full((batch_size, 1), start_token, dtype=torch.long).to(device)
            
            generated_tokens = set()
            for _ in range(max_length):
                # Create target mask
                tgt_mask = self.generate_square_subsequent_mask(generated.size(1)).to(device)
                
                # Embed target
                tgt_emb = self.pos_encoder(self.tgt_embedding(generated) * math.sqrt(self.d_model))
                
                # Decode
                output = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                
                # Get next token logits and apply temperature
                logits = self.fc_out(output[:, -1, :]) / temperature
                
                # Repetition penalty
                if generated.shape[1] > 1 and repetition_penalty != 1.0:
                    for tkn in generated[0].tolist():
                        if tkn not in (start_token, end_token, 0):
                            logits[0, tkn] /= repetition_penalty

                # Sort logits for nucleus sampling
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumulative = torch.cumsum(probs, dim=-1)

                # Apply top-p filter
                nucleus_mask = cumulative <= top_p
                # Ensure at least one token kept
                nucleus_mask[..., 0] = True
                filtered_logits = sorted_logits.clone()
                filtered_logits[~nucleus_mask] = -float('inf')

                # Optionally also apply top-k restriction inside nucleus
                if top_k and top_k < filtered_logits.shape[-1]:
                    filtered_logits[..., top_k:] = -float('inf')

                final_probs = F.softmax(filtered_logits, dim=-1)
                sampled = torch.multinomial(final_probs, 1)  # index within sorted list
                next_token = sorted_indices.gather(1, sampled)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if all sequences generated end token
                if (next_token == end_token).all():
                    break
            
            return generated
    
    def generate_with_beam_search(self, src, beam_size=3, max_length=50, 
                                   start_token=1, end_token=2):
        """
        Generate hook using beam search for better quality.
        
        Args:
            src: Source sequence (1, src_len) - single example
            beam_size: Number of beams
            max_length: Maximum length
            start_token: Start token ID
            end_token: End token ID
            
        Returns:
            Best generated sequence
        """
        self.eval()
        with torch.no_grad():
            device = src.device
            
            # Encode source
            src_padding_mask = self.create_padding_mask(src)
            src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
            memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)
            
            # Initialize beams
            beams = [(torch.tensor([[start_token]]).to(device), 0.0)]  # (sequence, score)
            
            for _ in range(max_length):
                new_beams = []
                
                for seq, score in beams:
                    if seq[0, -1].item() == end_token:
                        new_beams.append((seq, score))
                        continue
                    
                    # Get predictions for current sequence
                    tgt_mask = self.generate_square_subsequent_mask(seq.size(1)).to(device)
                    tgt_emb = self.pos_encoder(self.tgt_embedding(seq) * math.sqrt(self.d_model))
                    output = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                    logits = self.fc_out(output[:, -1, :])
                    log_probs = torch.log_softmax(logits, dim=-1)
                    
                    # Get top k tokens
                    top_k_probs, top_k_indices = torch.topk(log_probs, beam_size)
                    
                    for i in range(beam_size):
                        next_token = top_k_indices[0, i].unsqueeze(0).unsqueeze(0)
                        next_score = score + top_k_probs[0, i].item()
                        next_seq = torch.cat([seq, next_token], dim=1)
                        new_beams.append((next_seq, next_score))
                
                # Keep top beam_size beams
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
                
                # Stop if all beams ended
                if all(seq[0, -1].item() == end_token for seq, _ in beams):
                    break
            
            # Return best beam
            best_seq, _ = beams[0]
            return best_seq


if __name__ == "__main__":
    # Test the model
    vocab_size = 5000
    batch_size = 4
    src_len = 50
    tgt_len = 20
    
    model = TransformerHookGenerator(vocab_size=vocab_size)
    
    # Dummy data
    src = torch.randint(3, vocab_size, (batch_size, src_len))
    tgt = torch.randint(3, vocab_size, (batch_size, tgt_len))
    
    # Forward pass
    output = model(src, tgt)
    print(f"Output shape: {output.shape}")
    
    # Generation
    generated = model.generate(src, max_length=20)
    print(f"Generated shape: {generated.shape}")
    
    # Beam search generation (single example)
    generated_beam = model.generate_with_beam_search(src[:1], beam_size=3, max_length=20)
    print(f"Beam search generated shape: {generated_beam.shape}")
    
    print("Transformer model test passed!")
