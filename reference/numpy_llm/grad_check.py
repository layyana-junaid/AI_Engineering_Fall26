import numpy as np
from core import *

eps = 1e-5

def numeric_grad(f, X, dOut_used):
    grad = np.zeros_like(X)
    it = np.nditer(X, flags=['multi_index'])
    for _ in it:
        idx = it.multi_index
        orig = X[idx]
        X[idx] = orig + eps
        out_plus = f(X)
        X[idx] = orig - eps
        out_minus = f(X)
        X[idx] = orig
        grad[idx] = np.sum((out_plus - out_minus) * dOut_used) / (2 * eps)
    return grad

# ---- TEST: linear_forward/backward ----
print("="*60, "\nGRAD CHECK 1: linear_forward/backward\n" + "="*60)
np.random.seed(0)
X = np.random.randn(2, 3, 4)
W, b = init_linear(4, 5, seed=1)
out, cache = linear_forward(X, W, b)
dOut = np.random.randn(*out.shape)
dX, dW, db = linear_backward(dOut, cache, W)

dX_num = numeric_grad(lambda x: linear_forward(x, W, b)[0], X.copy(), dOut)
print("dX max abs diff:", np.max(np.abs(dX - dX_num)))

dW_num = numeric_grad(lambda w: linear_forward(X, w, b)[0], W.copy(), dOut)
print("dW max abs diff:", np.max(np.abs(dW - dW_num)))

# ---- TEST: layer_norm_forward/backward ----
print("\n" + "="*60, "\nGRAD CHECK 2: layer_norm_forward/backward\n" + "="*60)
X = np.random.randn(2, 3, 6)
gamma = np.ones((1,1,6)) * 1.3
beta = np.zeros((1,1,6))
out, cache = layer_norm_forward(X, gamma, beta)
dOut = np.random.randn(*out.shape)
dX, dgamma, dbeta = layer_norm_backward(dOut, cache)

dX_num = numeric_grad(lambda x: layer_norm_forward(x, gamma, beta)[0], X.copy(), dOut)
print("dX max abs diff:", np.max(np.abs(dX - dX_num)))
dgamma_num = numeric_grad(lambda g: layer_norm_forward(X, g, beta)[0], gamma.copy(), dOut)
print("dgamma max abs diff:", np.max(np.abs(dgamma - dgamma_num)))

# ---- TEST: attention_forward/backward (the big one) ----
print("\n" + "="*60, "\nGRAD CHECK 3: attention_forward/backward (causal, multi-head)\n" + "="*60)
np.random.seed(2)
batch, seq_len, embed_dim, num_heads = 2, 4, 8, 2
X = np.random.randn(batch, seq_len, embed_dim) * 0.5
Wq, bq = init_linear(embed_dim, embed_dim, seed=10)
Wk, bk = init_linear(embed_dim, embed_dim, seed=11)
Wv, bv = init_linear(embed_dim, embed_dim, seed=12)
Wo, bo = init_linear(embed_dim, embed_dim, seed=13)
params = dict(Wq=Wq, bq=bq, Wk=Wk, bk=bk, Wv=Wv, bv=bv, Wo=Wo, bo=bo)
mask = causal_mask(seq_len)

out, cache = attention_forward(X, params, num_heads, mask)
dOut = np.random.randn(*out.shape)
dX, grads = attention_backward(dOut, cache, params)

dX_num = numeric_grad(lambda x: attention_forward(x, params, num_heads, mask)[0], X.copy(), dOut)
print("dX max abs diff:", np.max(np.abs(dX - dX_num)))

dWq_num = numeric_grad(lambda w: attention_forward(X, {**params, "Wq": w}, num_heads, mask)[0], Wq.copy(), dOut)
print("dWq max abs diff:", np.max(np.abs(grads["dWq"] - dWq_num)))

dWo_num = numeric_grad(lambda w: attention_forward(X, {**params, "Wo": w}, num_heads, mask)[0], Wo.copy(), dOut)
print("dWo max abs diff:", np.max(np.abs(grads["dWo"] - dWo_num)))

# ---- Causal mask actually blocks future positions? ----
print("\n" + "="*60, "\nCHECK 4: causal mask correctness\n" + "="*60)
weights = cache["weights"]  # (batch, heads, seq, seq)
future_weights = np.triu(weights[0,0], k=1)
print("Max attention weight on any FUTURE position:", future_weights.max())
print("PASS:", np.isclose(future_weights.max(), 0.0))
