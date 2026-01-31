# Attention Is All You Need - Transformer Implementation

A PyTorch implementation of the Transformer architecture from the groundbreaking paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. (2017).

## 📋 Overview

This repository contains a complete, from-scratch implementation of the Transformer model, which revolutionized natural language processing by introducing the multi-head self-attention mechanism. The implementation includes all core components of the architecture along with training utilities and example usage.

## ✨ Features

- **Complete Transformer Architecture**
  - Multi-Head Attention mechanism with scaled dot-product attention
  - Positional Encoding using sinusoidal functions
  - Position-wise Feed-Forward Networks
  - Encoder and Decoder stacks with residual connections and layer normalization
  
- **Training Utilities**
  - Label smoothing for regularization
  - Learning rate scheduling with warmup (as described in the paper)
  - Gradient clipping
  - Customizable hyperparameters

- **Well-Documented Code**
  - Comprehensive docstrings for all classes and methods
  - Type hints for better code clarity
  - Clear comments explaining key concepts

## 🚀 Installation

### Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy

### Setup

```bash
# Clone the repository
git clone https://github.com/Mikop22/AttentionIsAllYouNeed.git
cd AttentionIsAllYouNeed

# Install dependencies
pip install torch numpy
```

## 💻 Usage

### Basic Example

```python
import torch
from Implementation import Transformer, TransformerTrainer

# Create a Transformer model
model = Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    n_heads=8,
    n_layers=6,
    d_ff=2048,
    max_seq_len=100,
    dropout=0.1
)

# Example input tensors (batch_size=2, seq_len=10)
src = torch.randint(1, 10000, (2, 10))
tgt = torch.randint(1, 10000, (2, 12))

# Forward pass
output = model(src, tgt[:, :-1])
print(f"Output shape: {output.shape}")  # [2, 11, 10000]

# Training
trainer = TransformerTrainer(model, learning_rate=0.0001, warmup_steps=4000)
loss = trainer.train_step(src, tgt)
print(f"Training loss: {loss:.4f}")
```

### Creating Masks

```python
# Padding mask for source sequence
src_mask = model.create_padding_mask(src, pad_idx=0)

# Causal mask for decoder (prevents attending to future tokens)
tgt_mask = model.generate_square_subsequent_mask(tgt_seq_len)
```

### Using Encoder and Decoder Separately

```python
# Encode source sequence
encoder_output = model.encode(src, src_mask)

# Decode with encoder output
decoder_output = model.decode(tgt, encoder_output, src_mask, tgt_mask)
```

## 🏗️ Architecture

### Model Components

1. **MultiHeadAttention**: Implements the scaled dot-product attention with multiple heads
   - Allows the model to jointly attend to information from different representation subspaces
   - Formula: `Attention(Q, K, V) = softmax(QK^T / √d_k)V`

2. **PositionalEncoding**: Adds positional information to token embeddings
   - Uses sine and cosine functions of different frequencies
   - Enables the model to understand sequence order

3. **FeedForwardNetwork**: Position-wise fully connected feed-forward network
   - Consists of two linear transformations with ReLU activation
   - Applied identically to each position

4. **EncoderLayer**: Single encoder layer with self-attention and feed-forward sublayers
   - Each sublayer has residual connection followed by layer normalization

5. **DecoderLayer**: Single decoder layer with masked self-attention, cross-attention, and feed-forward sublayers

6. **TransformerEncoder**: Stack of N encoder layers
   - Includes embedding layer and positional encoding

7. **TransformerDecoder**: Stack of N decoder layers
   - Includes embedding layer, positional encoding, and output projection

8. **Transformer**: Complete encoder-decoder model
   - Combines encoder and decoder stacks
   - Provides convenient methods for encoding, decoding, and mask generation

### Hyperparameters (Base Model)

| Parameter | Value | Description |
|-----------|-------|-------------|
| d_model | 512 | Model dimension |
| n_heads | 8 | Number of attention heads |
| n_layers | 6 | Number of encoder/decoder layers |
| d_ff | 2048 | Feed-forward dimension |
| dropout | 0.1 | Dropout rate |

## 📁 Repository Structure

```
AttentionIsAllYouNeed/
├── Implementation.py           # Complete Transformer implementation
├── SENTIMENT ANALYSIS NLP.ipynb # Jupyter notebook with sentiment analysis example
└── README.md                   # This file
```

## 🔑 Key Concepts

### Self-Attention Mechanism
The attention mechanism allows each position in the sequence to attend to all positions in the previous layer, enabling the model to capture long-range dependencies.

### Multi-Head Attention
Instead of performing a single attention function, multi-head attention linearly projects the queries, keys, and values h times with different learned projections, allowing the model to jointly attend to information from different representation subspaces.

### Positional Encoding
Since the Transformer contains no recurrence or convolution, positional encodings are added to give the model information about the relative or absolute position of tokens in the sequence.

## 🎯 Applications

This implementation can be used for various sequence-to-sequence tasks:
- Machine Translation
- Text Summarization
- Question Answering
- Text Generation
- Sentiment Analysis (see included notebook)

## 📚 References

- **Paper**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
  - Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
  - Published: NeurIPS 2017

- **Additional Resources**:
  - [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) by Jay Alammar
  - [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) by Harvard NLP

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📝 License

This project is available for educational and research purposes. Please cite the original paper if you use this implementation in your work.

## ⚡ Quick Start

Run the example to verify everything is working:

```bash
python Implementation.py
```

This will:
1. Initialize a Transformer model with the base configuration
2. Perform a forward pass with random input
3. Execute a single training step
4. Display model statistics and output shapes

Expected output:
```
Model initialized with X,XXX,XXX parameters
Output shape: torch.Size([2, 11, 10000])
Training loss: X.XXXX

Model ready for training on translation tasks!
```

## 🙏 Acknowledgments

This implementation is based on the original Transformer architecture from "Attention is All You Need". Special thanks to the authors for their groundbreaking work that has shaped modern NLP.

---

**Note**: This is an educational implementation. For production use, consider using established libraries like [Hugging Face Transformers](https://github.com/huggingface/transformers) or PyTorch's built-in [nn.Transformer](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html).
