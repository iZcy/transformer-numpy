import numpy as np

# basic transformer implementation

class TokenEmbedding:
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = np.random.randn(vocab_size, d_model) * 0.01

    def forward(self, x):
        # x: (batch, seq_len)
        return self.embedding[x]  # (batch, seq_len, d_model)
