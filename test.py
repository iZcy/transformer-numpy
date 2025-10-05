import numpy as np
from transformer import *

# test transformer components

def test_embeddings():
    vocab_size = 100
    d_model = 64
    batch_size = 2
    seq_len = 10

    embed = TokenEmbedding(vocab_size, d_model)
    x = np.random.randint(0, vocab_size, (batch_size, seq_len))
    output = embed.forward(x)

    assert output.shape == (batch_size, seq_len, d_model)
    print(f"✓ Token embedding: {output.shape}")


def test_positional_encoding():
    d_model = 64
    batch_size = 2
    seq_len = 10

    pos_enc = PositionalEncoding(d_model)
    x = np.random.randn(batch_size, seq_len, d_model)
    output = pos_enc.forward(x)

    assert output.shape == (batch_size, seq_len, d_model)
    print(f"✓ Positional encoding: {output.shape}")


def test_attention():
    d_model = 64
    num_heads = 8
    batch_size = 2
    seq_len = 10

    attn = MultiHeadAttention(d_model, num_heads)
    x = np.random.randn(batch_size, seq_len, d_model)
    output = attn.forward(x)

    assert output.shape == (batch_size, seq_len, d_model)
    print(f"✓ Multi-head attention: {output.shape}")


def test_mask():
    seq_len = 5
    mask = create_causal_mask(seq_len)
    print(f"✓ Causal mask shape: {mask.shape}")
    print(mask)


def test_full_model():
    vocab_size = 100
    d_model = 64
    num_heads = 8
    d_ff = 256
    num_layers = 2
    batch_size = 2
    seq_len = 10

    model = GPTModel(vocab_size, d_model, num_heads, d_ff, num_layers)
    x = np.random.randint(0, vocab_size, (batch_size, seq_len))

    logits, probs = model.forward(x)

    assert logits.shape == (batch_size, seq_len, vocab_size), f"Expected {(batch_size, seq_len, vocab_size)}, got {logits.shape}"
    assert probs.shape == (batch_size, vocab_size), f"Expected {(batch_size, vocab_size)}, got {probs.shape}"
    assert np.allclose(np.sum(probs, axis=-1), 1.0), "Probabilities should sum to 1"
    print(f"✓ Full model logits: {logits.shape}")
    print(f"✓ Next token probs: {probs.shape}")
    print(f"✓ Probability sum check passed")


if __name__ == "__main__":
    print("Testing transformer components...")
    test_embeddings()
    test_positional_encoding()
    test_attention()
    test_mask()
    test_full_model()
    print("\nAll tests passed!")
