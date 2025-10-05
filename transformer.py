import numpy as np

# basic transformer implementation

class TokenEmbedding:
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = np.random.randn(vocab_size, d_model) * 0.01

    def forward(self, x):
        # x: (batch, seq_len)
        return self.embedding[x] * np.sqrt(self.d_model)  # scale embedding


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


def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


class LayerNorm:
    def __init__(self, d_model, eps=1e-5):
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class ScaledDotProductAttention:
    def __init__(self):
        pass

    def forward(self, Q, K, V, mask=None):
        # Q, K, V: (batch, seq_len, d_k)
        d_k = Q.shape[-1]
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)

        if mask is not None:
            scores = scores + mask

        attn_weights = softmax(scores, axis=-1)
        output = np.matmul(attn_weights, V)
        return output, attn_weights
