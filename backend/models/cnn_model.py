"""
CNN Model for LinkedIn Post Summarization/Concise Generation.
Uses 1D convolutions to extract key features from text.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """
    1D CNN for text processing and summarization.
    Uses multiple filter sizes to capture different n-gram patterns.
    """
    
    def __init__(self, vocab_size, embedding_dim=128, num_filters=100, 
                 filter_sizes=[3, 4, 5], output_dim=128, dropout=0.5):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            num_filters: Number of filters for each filter size
            filter_sizes: List of filter sizes (kernel sizes)
            output_dim: Dimension of output representation
            dropout: Dropout probability
        """
        super(TextCNN, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Convolutional layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, 
                     out_channels=num_filters, 
                     kernel_size=fs)
            for fs in filter_sizes
        ])
        
        # Fully connected layers
        self.fc1 = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass through the CNN.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Embedding: (batch_size, seq_length) -> (batch_size, seq_length, embedding_dim)
        embedded = self.embedding(x)
        
        # Transpose for Conv1d: (batch_size, embedding_dim, seq_length)
        embedded = embedded.permute(0, 2, 1)
        
        # Apply convolutions and max pooling
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        # Concatenate all pooled features
        cat = torch.cat(pooled, dim=1)
        cat = self.dropout(cat)
        
        # Fully connected layers
        out = F.relu(self.fc1(cat))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class CNNSummarizer(nn.Module):
    """
    CNN-based sequence-to-sequence model for text summarization.
    Encodes input text and generates concise version.
    """
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, 
                 num_filters=100, filter_sizes=[3, 4, 5], dropout=0.5):
        super(CNNSummarizer, self).__init__()
        
        # Encoder CNN
        self.encoder_cnn = TextCNN(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_filters=num_filters,
            filter_sizes=filter_sizes,
            output_dim=hidden_dim,
            dropout=dropout
        )
        
        # Decoder LSTM (simpler decoder for generation)
        self.decoder_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.decoder_lstm = nn.LSTM(
            input_size=embedding_dim + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=dropout,
            batch_first=True
        )
        
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Forward pass for training.
        
        Args:
            src: Source sequence (batch_size, src_len)
            trg: Target sequence (batch_size, trg_len)
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            Output predictions (batch_size, trg_len, vocab_size)
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        vocab_size = self.output_layer.out_features
        
        # Encode source with CNN
        encoder_output = self.encoder_cnn(src)  # (batch_size, hidden_dim)
        
        # Prepare decoder
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(src.device)
        
        # First input to decoder is <sos> token
        decoder_input = trg[:, 0].unsqueeze(1)  # (batch_size, 1)
        
        # Initialize hidden state
        hidden = None
        
        for t in range(1, trg_len):
            # Embed decoder input
            embedded = self.decoder_embedding(decoder_input)  # (batch_size, 1, embedding_dim)
            
            # Concatenate with encoder output
            encoder_out_expanded = encoder_output.unsqueeze(1)  # (batch_size, 1, hidden_dim)
            lstm_input = torch.cat([embedded, encoder_out_expanded], dim=2)
            
            # Pass through LSTM
            if hidden is None:
                output, hidden = self.decoder_lstm(lstm_input)
            else:
                output, hidden = self.decoder_lstm(lstm_input, hidden)
            
            # Predict next token
            prediction = self.output_layer(output.squeeze(1))
            outputs[:, t, :] = prediction
            
            # Teacher forcing
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            if use_teacher_forcing:
                decoder_input = trg[:, t].unsqueeze(1)
            else:
                decoder_input = prediction.argmax(1).unsqueeze(1)
        
        return outputs
    
    def generate(self, src, max_length=48, start_token=1, end_token=2,
                 temperature=0.9, top_k=40, top_p=0.9, repetition_penalty=1.1,
                 min_length=8):
        """
        Generate a concise summary using nucleus (top-p) + optional top-k sampling
        and a light repetition penalty to reduce loops, with a minimum-length heuristic.

        Args:
            src: Tensor (batch_size, src_len)
            max_length: Max number of generated tokens (excluding <sos>)
            start_token: ID for <sos>
            end_token: ID for <eos>
            temperature: Softmax temperature (higher = more diverse)
            top_k: Optional cap on nucleus candidate count
            top_p: Cumulative probability threshold for nucleus sampling
            repetition_penalty: >1.0 reduces probability of already sampled tokens
            min_length: Enforce at least this many tokens (excludes <sos>) before allowing <eos>

        Returns:
            Tensor of shape (batch_size, generated_len)
        """
        self.eval()
        with torch.no_grad():
            batch_size = src.shape[0]
            # Encode source once
            encoder_output = self.encoder_cnn(src)  # (batch_size, hidden_dim)

            # Initialize decoder with <sos>
            decoder_input = torch.full((batch_size, 1), start_token, device=src.device, dtype=torch.long)
            generated = decoder_input.clone()
            hidden = None

            for step in range(max_length):
                embedded = self.decoder_embedding(decoder_input)  # (batch, 1, emb_dim)
                encoder_expanded = encoder_output.unsqueeze(1)    # (batch, 1, hidden_dim)
                lstm_input = torch.cat([embedded, encoder_expanded], dim=2)

                if hidden is None:
                    output, hidden = self.decoder_lstm(lstm_input)
                else:
                    output, hidden = self.decoder_lstm(lstm_input, hidden)

                logits = self.output_layer(output.squeeze(1))  # (batch, vocab)
                logits = logits / max(temperature, 1e-6)

                # Repetition penalty (unique tokens only for efficiency)
                if repetition_penalty != 1.0 and generated.shape[1] > 0:
                    used_tokens = set(generated[0].tolist())
                    for tkn in used_tokens:
                        if tkn not in (start_token, end_token, 0) and tkn < logits.shape[1]:
                            logits[:, tkn] /= repetition_penalty

                # Nucleus (top-p) sampling
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumulative = torch.cumsum(probs, dim=-1)
                nucleus_mask = cumulative <= top_p
                nucleus_mask[:, 0] = True  # always keep top token
                filtered_logits = sorted_logits.clone()
                filtered_logits[~nucleus_mask] = -float('inf')

                # Optional top-k refinement inside nucleus
                if top_k and top_k < filtered_logits.shape[1]:
                    filtered_logits[:, top_k:] = -float('inf')

                final_probs = F.softmax(filtered_logits, dim=-1)
                sampled_rel = torch.multinomial(final_probs, 1)  # (batch,1) index in sorted list
                next_token = sorted_indices.gather(1, sampled_rel)  # map back to vocab ids

                # Enforce minimum content length: if EOS too early, pick the next-best token
                if (next_token == end_token).all() and (generated.shape[1] - 1) < min_length:
                    # choose second-best token in sorted indices
                    if sorted_indices.shape[1] > 1:
                        next_token = sorted_indices[:, 1:2]

                generated = torch.cat([generated, next_token], dim=1)
                decoder_input = next_token

                if (next_token == end_token).all():
                    break

            # If still too short, pad out by taking top-1 (excluding special tokens)
            while (generated.shape[1] - 1) < min_length and (generated.shape[1] - 1) < max_length:
                embedded = self.decoder_embedding(decoder_input)
                encoder_expanded = encoder_output.unsqueeze(1)
                lstm_input = torch.cat([embedded, encoder_expanded], dim=2)
                output, hidden = self.decoder_lstm(lstm_input, hidden)
                logits = self.output_layer(output.squeeze(1))
                # Avoid sampling <eos>, <sos>, <pad>
                logits[:, start_token] = -float('inf')
                logits[:, end_token] = -float('inf')
                logits[:, 0] = -float('inf')
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                decoder_input = next_token

            return generated


if __name__ == "__main__":
    # Test the model
    vocab_size = 5000
    batch_size = 4
    src_len = 50
    trg_len = 30
    
    model = CNNSummarizer(vocab_size=vocab_size)
    
    # Dummy data
    src = torch.randint(0, vocab_size, (batch_size, src_len))
    trg = torch.randint(0, vocab_size, (batch_size, trg_len))
    
    # Forward pass
    output = model(src, trg)
    print(f"Output shape: {output.shape}")
    
    # Generation
    generated = model.generate(src, max_length=30)
    print(f"Generated shape: {generated.shape}")
    print("CNN model test passed!")
