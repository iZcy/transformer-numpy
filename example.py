import numpy as np
from transformer import GPTModel

# simple example
vocab_size = 50
d_model = 128
num_heads = 4
d_ff = 512
num_layers = 4

print("Creating GPT model...")
model = GPTModel(vocab_size, d_model, num_heads, d_ff, num_layers)

# simulate input sequence
batch_size = 1
seq_len = 8
x = np.random.randint(0, vocab_size, (batch_size, seq_len))

print(f"Input shape: {x.shape}")
print(f"Input tokens: {x}")

# forward pass
logits, probs = model.forward(x)

print(f"\nOutput logits shape: {logits.shape}")
print(f"Next token probabilities shape: {probs.shape}")

# get top 5 predicted tokens
top_5 = np.argsort(probs[0])[-5:][::-1]
print(f"\nTop 5 predicted next tokens:")
for i, token_id in enumerate(top_5):
    print(f"  {i+1}. Token {token_id}: {probs[0][token_id]:.4f}")
