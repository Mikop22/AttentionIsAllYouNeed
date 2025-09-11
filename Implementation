import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism as described in the paper.
    
    This module implements the scaled dot-product attention with multiple heads,
    allowing the model to attend to information from different representation
    subspaces at different positions.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model: Dimensionality of the model (embedding size)
            n_heads: Number of attention heads
            dropout: Dropout probability for regularization
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply multi-head attention mechanism.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Optional mask tensor for preventing attention to certain positions
            
        Returns:
            Output tensor after attention [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = query.shape
        
        # Linear transformations and reshape to multiple heads
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads and apply output projection
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.w_o(context)
        
        return output


class PositionalEncoding(nn.Module):
    """
    Positional Encoding using sine and cosine functions.
    
    Since the Transformer contains no recurrence and no convolution,
    we inject information about positions using positional encodings.
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Create div_term for the sinusoidal pattern
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    Implements FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimensionality
            d_ff: Hidden layer dimensionality (typically 4 * d_model)
            dropout: Dropout probability
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward network."""
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    """
    Single encoder layer consisting of multi-head attention and feed-forward network.
    
    Each layer has two sub-layers with residual connections and layer normalization.
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process input through encoder layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Processed tensor [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """
    Single decoder layer with masked self-attention, encoder-decoder attention,
    and feed-forward network.
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process input through decoder layer.
        
        Args:
            x: Input tensor from previous decoder layer
            encoder_output: Output from encoder stack
            src_mask: Mask for encoder-decoder attention
            tgt_mask: Mask for masked self-attention
            
        Returns:
            Processed tensor
        """
        # Masked self-attention
        self_attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Encoder-decoder attention
        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward network
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class TransformerEncoder(nn.Module):
    """
    Stack of N encoder layers forming the encoder component of the Transformer.
    """
    
    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int,
                 vocab_size: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode input sequence.
        
        Args:
            x: Input token indices [batch_size, seq_len]
            mask: Optional padding mask
            
        Returns:
            Encoded representation [batch_size, seq_len, d_model]
        """
        # Token embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)


class TransformerDecoder(nn.Module):
    """
    Stack of N decoder layers forming the decoder component of the Transformer.
    """
    
    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int,
                 vocab_size: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode target sequence given encoder output.
        
        Args:
            x: Target token indices [batch_size, tgt_seq_len]
            encoder_output: Encoder output [batch_size, src_seq_len, d_model]
            src_mask: Source sequence mask
            tgt_mask: Target sequence mask (causal)
            
        Returns:
            Logits over vocabulary [batch_size, tgt_seq_len, vocab_size]
        """
        # Token embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        x = self.norm(x)
        return self.output_projection(x)


class Transformer(nn.Module):
    """
    Complete Transformer model for sequence-to-sequence tasks.
    
    This implementation follows the architecture described in "Attention is All You Need",
    featuring multi-head attention, positional encoding, and layer normalization.
    """
    
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 d_model: int = 512, n_heads: int = 8, n_layers: int = 6,
                 d_ff: int = 2048, max_seq_len: int = 5000, dropout: float = 0.1):
        """
        Initialize Transformer model.
        
        Args:
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            d_model: Model dimension (default: 512)
            n_heads: Number of attention heads (default: 8)
            n_layers: Number of encoder/decoder layers (default: 6)
            d_ff: Feed-forward network dimension (default: 2048)
            max_seq_len: Maximum sequence length for positional encoding
            dropout: Dropout probability for regularization
        """
        super().__init__()
        
        self.encoder = TransformerEncoder(
            n_layers, d_model, n_heads, d_ff, src_vocab_size, max_seq_len, dropout
        )
        self.decoder = TransformerDecoder(
            n_layers, d_model, n_heads, d_ff, tgt_vocab_size, max_seq_len, dropout
        )
        
        # Initialize parameters with Xavier uniform
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize model parameters using Xavier uniform distribution."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def generate_square_subsequent_mask(self, size: int) -> torch.Tensor:
        """
        Generate causal mask for decoder self-attention.
        
        Args:
            size: Sequence length
            
        Returns:
            Upper triangular matrix of shape [size, size]
        """
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))
    
    def create_padding_mask(self, x: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """
        Create padding mask for attention mechanisms.
        
        Args:
            x: Input tensor with padding
            pad_idx: Index used for padding tokens
            
        Returns:
            Boolean mask where True indicates valid positions
        """
        return (x != pad_idx).unsqueeze(1).unsqueeze(2)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the Transformer.
        
        Args:
            src: Source sequence [batch_size, src_seq_len]
            tgt: Target sequence [batch_size, tgt_seq_len]
            src_mask: Optional source mask
            tgt_mask: Optional target mask
            
        Returns:
            Output logits [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        # Encode source sequence
        encoder_output = self.encoder(src, src_mask)
        
        # Decode target sequence
        output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        
        return output
    
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode source sequence only."""
        return self.encoder(src, src_mask)
    
    def decode(self, tgt: torch.Tensor, encoder_output: torch.Tensor,
               src_mask: Optional[torch.Tensor] = None,
               tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode target sequence given encoder output."""
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)


class TransformerTrainer:
    """
    Training utilities for the Transformer model.
    
    Implements training loop with label smoothing, learning rate scheduling,
    and gradient clipping as described in the paper.
    """
    
    def __init__(self, model: Transformer, learning_rate: float = 0.0001,
                 warmup_steps: int = 4000, label_smoothing: float = 0.1):
        """
        Initialize trainer with model and hyperparameters.
        
        Args:
            model: Transformer model to train
            learning_rate: Base learning rate
            warmup_steps: Number of warmup steps for learning rate schedule
            label_smoothing: Label smoothing factor for regularization
        """
        self.model = model
        self.base_lr = learning_rate
        self.warmup_steps = warmup_steps
        self.label_smoothing = label_smoothing
        
        # Initialize optimizer (Adam with β1=0.9, β2=0.98, ε=10^-9 as in paper)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        self.step_num = 0
        
    def get_learning_rate(self) -> float:
        """
        Calculate learning rate with warmup schedule as described in the paper.
        
        lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
        """
        self.step_num += 1
        d_model = self.model.encoder.d_model
        
        # Implement the learning rate schedule from the paper
        arg1 = self.step_num ** (-0.5)
        arg2 = self.step_num * (self.warmup_steps ** (-1.5))
        
        return (d_model ** (-0.5)) * min(arg1, arg2)
    
    def update_learning_rate(self):
        """Update learning rate according to schedule."""
        lr = self.get_learning_rate()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                    pad_idx: int = 0) -> torch.Tensor:
        """
        Compute cross-entropy loss with label smoothing.
        
        Args:
            predictions: Model predictions [batch_size, seq_len, vocab_size]
            targets: Target tokens [batch_size, seq_len]
            pad_idx: Padding token index to ignore
            
        Returns:
            Scalar loss tensor
        """
        vocab_size = predictions.size(-1)
        
        # Reshape for loss computation
        predictions = predictions.reshape(-1, vocab_size)
        targets = targets.reshape(-1)
        
        # Create smoothed target distribution
        if self.label_smoothing > 0:
            smoothed_targets = torch.zeros_like(predictions).scatter_(
                1, targets.unsqueeze(1), 1 - self.label_smoothing
            )
            smoothed_targets += self.label_smoothing / vocab_size
            
            # Compute KL divergence loss
            log_probs = F.log_softmax(predictions, dim=-1)
            loss = F.kl_div(log_probs, smoothed_targets, reduction='none').sum(dim=-1)
        else:
            loss = F.cross_entropy(predictions, targets, reduction='none')
        
        # Mask padding tokens
        mask = targets != pad_idx
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
    
    def train_step(self, src: torch.Tensor, tgt: torch.Tensor,
                  pad_idx: int = 0) -> float:
        """
        Perform single training step.
        
        Args:
            src: Source sequence batch
            tgt: Target sequence batch
            pad_idx: Padding token index
            
        Returns:
            Loss value for this step
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Create masks
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        src_mask = self.model.create_padding_mask(src, pad_idx)
        tgt_mask = self.model.generate_square_subsequent_mask(tgt_input.size(1)).to(src.device)
        
        # Forward pass
        predictions = self.model(src, tgt_input, src_mask, tgt_mask)
        
        # Compute loss
        loss = self.compute_loss(predictions, tgt_output, pad_idx)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update weights and learning rate
        self.optimizer.step()
        self.update_learning_rate()
        
        return loss.item()


def create_model_for_translation(src_vocab_size: int = 10000,
                                tgt_vocab_size: int = 10000) -> Transformer:
    """
    Create a Transformer model configured for translation tasks.
    
    Uses the base model configuration from the paper:
    - 6 layers
    - 8 attention heads
    - 512 model dimension
    - 2048 feed-forward dimension
    
    Args:
        src_vocab_size: Size of source vocabulary
        tgt_vocab_size: Size of target vocabulary
        
    Returns:
        Initialized Transformer model
    """
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        max_seq_len=100,
        dropout=0.1
    )
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


# Example usage and testing
if __name__ == "__main__":
    # Create model instance
    model = create_model_for_translation(src_vocab_size=10000, tgt_vocab_size=10000)
    
    # Example input tensors
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 12
    
    # Random token indices for demonstration
    src = torch.randint(1, 10000, (batch_size, src_seq_len))
    tgt = torch.randint(1, 10000, (batch_size, tgt_seq_len))
    
    # Forward pass
    output = model(src, tgt[:, :-1])
    print(f"Output shape: {output.shape}")  # [batch_size, tgt_seq_len-1, tgt_vocab_size]
    
    # Initialize trainer
    trainer = TransformerTrainer(model, learning_rate=0.0001, warmup_steps=4000)
    
    # Perform one training step
    loss = trainer.train_step(src, tgt)
    print(f"Training loss: {loss:.4f}")
    
    print("\nModel ready for training on translation tasks!")
