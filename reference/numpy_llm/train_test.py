import numpy as np
from core import *
from full_model import *
from tokenizer import train_bpe, encode

#tokenize a tiny repeated corpus using YOUR tokenizer
text = "the cat sat on the mat the cat sat on the mat the cat sat on the mat"
merges, token_to_id, id_to_token = train_bpe(text, target_vocab_size=20, verbose=False)
tokens, ids = encode(text, merges, token_to_id)
vocab_size = len(token_to_id)
print(f"Vocab size: {vocab_size}")
print(f"Encoded length: {len(ids)} tokens")

#build (input, target) pairs: predict next token at every position
seq_len = 8
X_list, Y_list = [], []
for i in range(0, len(ids) - seq_len - 1, 1):
    X_list.append(ids[i:i+seq_len])
    Y_list.append(ids[i+1:i+seq_len+1])
X = np.array(X_list)
Y = np.array(Y_list)
print(f"Training examples: {X.shape[0]}, each seq_len={seq_len}")

#shape testacross a couple configs first
print("\n" + "="*60, "\nSHAPE TEST\n" + "="*60)
for embed_dim, num_heads, num_blocks in [(8,2,1), (16,4,2)]:
    p = init_transformer(vocab_size, embed_dim, num_heads, num_blocks, ffn_hidden=embed_dim*2, seed=0)
    logits, cache = model_forward(X[:2], p, num_heads, vocab_size, embed_dim)
    ok = logits.shape == (2, seq_len, vocab_size)
    print(f"  embed_dim={embed_dim} heads={num_heads} blocks={num_blocks} -> logits shape {logits.shape}  OK={ok}")

#tiny overfit test
print("\n" + "="*60, "\nOVERFIT TEST: can it learn this tiny repeated sentence?\n" + "="*60)
embed_dim, num_heads, num_blocks, ffn_hidden = 16, 2, 2, 32
params = init_transformer(vocab_size, embed_dim, num_heads, num_blocks, ffn_hidden, seed=42)
flat_params = flatten_params(params, num_blocks)
m_state = {k: None for k in flat_params}
v_state = {k: None for k in flat_params}

EPOCHS, lr = 400, 0.01
for epoch in range(1, EPOCHS + 1):
    logits, cache = model_forward(X, params, num_heads, vocab_size, embed_dim)
    loss, dLogits = softmax_cross_entropy(logits, Y)
    grads = model_backward(dLogits, cache, params, vocab_size, embed_dim)

    flat_grads = flatten_grads(grads, num_blocks)
    flat_params, m_state, v_state = apply_adam_to_all(flat_params, flat_grads, m_state, v_state, epoch, lr)
    params = unflatten_into_params(flat_params, params, num_blocks)

    if epoch == 1 or epoch % 50 == 0:
        preds = np.argmax(logits, axis=-1)
        acc = np.mean(preds == Y)
        print(f"Epoch {epoch:4d} | loss: {loss:.4f} | next-token acc: {acc:.3f}")

#generate some text from a prompt
print("\n" + "="*60, "\nGENERATION TEST\n" + "="*60)
def generate(prompt_ids, params, num_heads, vocab_size, embed_dim, max_new_tokens=10):
    ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        context = np.array([ids[-seq_len:]]) if len(ids) >= seq_len else np.array([ids])
        logits, _ = model_forward(context, params, num_heads, vocab_size, embed_dim)
        next_id = np.argmax(logits[0, -1])
        ids.append(int(next_id))
    return ids

prompt = ids[:4]
generated = generate(prompt, params, num_heads, vocab_size, embed_dim, max_new_tokens=10)
decoded = "".join(id_to_token[i] for i in generated)
print(f"Prompt tokens: {prompt} -> {''.join(id_to_token[i] for i in prompt)!r}")
print(f"Generated:     {decoded!r}")
