# Transformer from Scratch

Simple implementation of decoder-only Transformer (GPT-style) using only NumPy.

## Requirements

- NumPy

```bash
pip install numpy
```

## Usage

```python
import numpy as np
from transformer import GPTModel

# model config
vocab_size = 1000
d_model = 256
num_heads = 8
d_ff = 1024
num_layers = 6

# create model
model = GPTModel(vocab_size, d_model, num_heads, d_ff, num_layers)

# input tokens
x = np.array([[1, 2, 3, 4, 5]])  # (batch, seq_len)

# forward pass
logits, probs = model.forward(x)
# logits: (batch, seq_len, vocab_size)
# probs: (batch, vocab_size) - next token distribution
```

## Run Tests

```bash
python3 test.py
```

## Components

- Token Embedding with scaling
- Sinusoidal Positional Encoding
- Scaled Dot-Product Attention
- Multi-Head Attention
- Feed-Forward Network (GELU activation)
- Layer Normalization (pre-norm)
- Causal Masking
- Full GPT decoder model

## Architecture Details

- Uses pre-norm architecture (LayerNorm before attention/FFN)
- Causal masking prevents attending to future tokens
- Returns both full logits and next-token probability distribution
