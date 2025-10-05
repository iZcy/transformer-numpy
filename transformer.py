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
        # Q, K, V: (batch, seq_len, d_k) or (batch, num_heads, seq_len, d_k)
        d_k = Q.shape[-1]

        # handle both 3D and 4D inputs
        if len(Q.shape) == 4:
            scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
        else:
            scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)

        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)

        attn_weights = softmax(scores, axis=-1)
        output = np.matmul(attn_weights, V)
        return output, attn_weights


class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        self.W_o = np.random.randn(d_model, d_model) * 0.01

        self.attention = ScaledDotProductAttention()

    def split_heads(self, x, batch_size, seq_len):
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        Q = self.split_heads(np.matmul(x, self.W_q), batch_size, seq_len)
        K = self.split_heads(np.matmul(x, self.W_k), batch_size, seq_len)
        V = self.split_heads(np.matmul(x, self.W_v), batch_size, seq_len)

        attn_output, attn_weights = self.attention.forward(Q, K, V, mask)

        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        output = np.matmul(attn_output, self.W_o)

        return output
