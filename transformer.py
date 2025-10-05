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


class PositionalEncoding:
    def __init__(self, d_model, max_len=5000):
        self.d_model = d_model
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = pe

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.shape[1]
        return x + self.pe[:seq_len]
