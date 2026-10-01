import numpy as np

##each token id gets its own row of numbers, which is the embedding vector for that token
def init_embedding(vocab_size, embedding_dim, scale=0.01, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.randn(vocab_size, embedding_dim) * scale

def embedding_forward(indices, E):
    return E[indices], indices

def embedding_backward(dout, cache, vocab_size, embedding_dim):
    indices = cache
    dE = np.zeros((vocab_size, embedding_dim))
    np.add.at(dE, indices, dout)
    return dE

def adam(params, gradients, momentum, velocity, t, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    momentum = beta1 * momentum + (1 - beta1) * gradients
    velocity = beta2 * velocity + (1 - beta2) * gradients ** 2
    momentum_corrected = momentum / (1 - beta1 ** t)
    velocity_corrected = velocity / (1 - beta2 ** t)
    params = params - learning_rate * momentum_corrected / (np.sqrt(velocity_corrected) + epsilon)
    return params, momentum, velocity

#basically currently the model have no sense of order, so this function adds a positional stamp
#to each tokens vector, so the token 1 and token 5 will numerically look different even if they are the same word
def positional_encoding(seq_len, embed_dim):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(embed_dim)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / embed_dim)
    angles = pos * angle_rates
    pe = np.zeros((seq_len, embed_dim))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe

def init_linear(n_in, n_out, seed=None):
    if seed is not None:
        np.random.seed(seed)
    W = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
    b = np.zeros((1, n_out))
    return W, b

def linear_forward(X, W, b):
    out = X @ W + b
    return out, X

def linear_backward(dOut, cache, W):
    X = cache
    X_flat = X.reshape(-1, X.shape[-1])          # (batch*seq, n_in)
    dOut_flat = dOut.reshape(-1, dOut.shape[-1])  # (batch*seq, n_out)
    dW = X_flat.T @ dOut_flat                     # sums over batch AND seq dims
    db = dOut_flat.sum(axis=0, keepdims=True)
    dX = dOut @ W.T
    return dX, dW, db


def softmax(Z):
    Z = Z - np.max(Z, axis=-1, keepdims=True)
    e = np.exp(Z)
    return e / np.sum(e, axis=-1, keepdims=True)

def softmax_backward(dOut, out):
    # out: the softmax output. Standard Jacobian-vector product, vectorized.
    dot = np.sum(dOut * out, axis=-1, keepdims=True)
    return out * (dOut - dot)

##basically stating that the mode is only allowed to look at the tokens that came before it,
###never the ones that come after it, this is important for training the model to predict the next token in 
### a sequence, because it should not be able to cheat and look at the future tokens
def causal_mask(seq_len):
    mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
    return mask  # 0 where allowed, -1e9 where disallowed (future positions)

def split_heads(X, num_heads):
    batch, seq_len, embed_dim = X.shape
    head_dim = embed_dim // num_heads
    X = X.reshape(batch, seq_len, num_heads, head_dim)
    return X.transpose(0, 2, 1, 3)  # (batch, num_heads, seq_len, head_dim)

def merge_heads(X):
    batch, num_heads, seq_len, head_dim = X.shape
    X = X.transpose(0, 2, 1, 3)
    return X.reshape(batch, seq_len, num_heads * head_dim)

def attention_forward(X, params, num_heads, mask=None):
    Q, cache_q = linear_forward(X, params["Wq"], params["bq"])
    K, cache_k = linear_forward(X, params["Wk"], params["bk"])
    V, cache_v = linear_forward(X, params["Wv"], params["bv"])

    Qh, Kh, Vh = split_heads(Q, num_heads), split_heads(K, num_heads), split_heads(V, num_heads)
    head_dim = Qh.shape[-1]

    scores = Qh @ Kh.transpose(0, 1, 3, 2) / np.sqrt(head_dim)
    if mask is not None:
        scores = scores + mask  # broadcasts (seq,seq) onto (batch,heads,seq,seq)

    weights = softmax(scores)
    attn_out = weights @ Vh                      # (batch, heads, seq, head_dim)
    merged = merge_heads(attn_out)                # (batch, seq, embed_dim)

    out, cache_o = linear_forward(merged, params["Wo"], params["bo"])

    cache = dict(cache_q=cache_q, cache_k=cache_k, cache_v=cache_v, cache_o=cache_o,
                 Qh=Qh, Kh=Kh, Vh=Vh, weights=weights, head_dim=head_dim, num_heads=num_heads)
    return out, cache

def attention_backward(dOut, cache, params):
    dMerged, dWo, dbo = linear_backward(dOut, cache["cache_o"], params["Wo"])
    num_heads = cache["num_heads"]
    dAttnOut = split_heads(dMerged, num_heads)               # (batch,heads,seq,head_dim)

    Vh, weights, Qh, Kh = cache["Vh"], cache["weights"], cache["Qh"], cache["Kh"]
    head_dim = cache["head_dim"]

    dWeights = dAttnOut @ Vh.transpose(0, 1, 3, 2)
    dVh = weights.transpose(0, 1, 3, 2) @ dAttnOut

    dScores = softmax_backward(dWeights, weights) / np.sqrt(head_dim)
    dQh = dScores @ Kh
    dKh = dScores.transpose(0, 1, 3, 2) @ Qh

    dQ, dK, dV = merge_heads(dQh), merge_heads(dKh), merge_heads(dVh)

    dX_q, dWq, dbq = linear_backward(dQ, cache["cache_q"], params["Wq"])
    dX_k, dWk, dbk = linear_backward(dK, cache["cache_k"], params["Wk"])
    dX_v, dWv, dbv = linear_backward(dV, cache["cache_v"], params["Wv"])

    dX = dX_q + dX_k + dX_v  # X was reused three times (branches into Q,K,V) -> gradients sum

    grads = dict(dWq=dWq, dbq=dbq, dWk=dWk, dbk=dbk, dWv=dWv, dbv=dbv, dWo=dWo, dbo=dbo)
    return dX, grads


#layer normalization
## keep the numbers flowing through the network from wildly exploding or vanishing, by normalizing the numbers in each layer
def layer_norm_forward(X, gamma, beta, eps=1e-5):
    mu = X.mean(axis=-1, keepdims=True)
    var = X.var(axis=-1, keepdims=True)
    X_norm = (X - mu) / np.sqrt(var + eps)
    out = gamma * X_norm + beta
    cache = (X, X_norm, mu, var, gamma, eps)
    return out, cache

def layer_norm_backward(dOut, cache):
    X, X_norm, mu, var, gamma, eps = cache
    N = X.shape[-1]
    std_inv = 1 / np.sqrt(var + eps)

    dX_norm = dOut * gamma
    dvar = np.sum(dX_norm * (X - mu) * -0.5 * std_inv**3, axis=-1, keepdims=True)
    dmu = np.sum(dX_norm * -std_inv, axis=-1, keepdims=True) + dvar * np.mean(-2 * (X - mu), axis=-1, keepdims=True)
    dX = dX_norm * std_inv + dvar * 2 * (X - mu) / N + dmu / N

    dgamma = np.sum(dOut * X_norm, axis=tuple(range(dOut.ndim - 1)), keepdims=True)
    dbeta = np.sum(dOut, axis=tuple(range(dOut.ndim - 1)), keepdims=True)
    return dX, dgamma, dbeta

#feed forward network
def ffn_forward(X, params):
    H, cache_h = linear_forward(X, params["W1"], params["b1"])
    H_relu = np.maximum(0, H)
    out, cache_o = linear_forward(H_relu, params["W2"], params["b2"])
    cache = (cache_h, H, H_relu, cache_o)
    return out, cache

def ffn_backward(dOut, cache, params):
    cache_h, H, H_relu, cache_o = cache
    dH_relu, dW2, db2 = linear_backward(dOut, cache_o, params["W2"])
    dH = dH_relu * (H > 0).astype(float)
    dX, dW1, db1 = linear_backward(dH, cache_h, params["W1"])
    grads = dict(dW1=dW1, db1=db1, dW2=dW2, db2=db2)
    return dX, grads

#transformer block forward and backward
def transformer_block_forward(X, params, num_heads, mask):
    attn_out, attn_cache = attention_forward(X, params["attn"], num_heads, mask)
    X1 = X + attn_out                                         # residual
    X1_norm, norm1_cache = layer_norm_forward(X1, params["gamma1"], params["beta1"])

    ffn_out, ffn_cache = ffn_forward(X1_norm, params["ffn"])
    X2 = X1_norm + ffn_out                                    # residual
    X2_norm, norm2_cache = layer_norm_forward(X2, params["gamma2"], params["beta2"])

    cache = dict(attn_cache=attn_cache, norm1_cache=norm1_cache,
                 ffn_cache=ffn_cache, norm2_cache=norm2_cache)
    return X2_norm, cache

def transformer_block_backward(dOut, cache, params):
    dX2, dgamma2, dbeta2 = layer_norm_backward(dOut, cache["norm2_cache"])
    dX1_norm_from_ffn, ffn_grads = ffn_backward(dX2, cache["ffn_cache"], params["ffn"])
    dX1_norm = dX1_norm_from_ffn + dX2   # residual: dX2 flows both through ffn AND straight through

    dX1, dgamma1, dbeta1 = layer_norm_backward(dX1_norm, cache["norm1_cache"])
    dX_from_attn, attn_grads = attention_backward(dX1, cache["attn_cache"], params["attn"])
    dX = dX_from_attn + dX1   # residual: dX1 flows both through attention AND straight through

    grads = dict(attn=attn_grads, ffn=ffn_grads,
                 dgamma1=dgamma1, dbeta1=dbeta1, dgamma2=dgamma2, dbeta2=dbeta2)
    return dX, grads


def output_projection_forward(X, W_out):
    logits = X @ W_out   # (batch, seq_len, vocab_size), no bias for simplicity
    return logits, X

def output_projection_backward(dLogits, cache, W_out):
    X = cache
    X_flat = X.reshape(-1, X.shape[-1])
    dLogits_flat = dLogits.reshape(-1, dLogits.shape[-1])
    dW_out = X_flat.T @ dLogits_flat
    dX = dLogits @ W_out.T
    return dX, dW_out

def softmax_cross_entropy(logits, targets):
    # logits: (batch, seq_len, vocab_size)   targets: (batch, seq_len) int token ids
    probs = softmax(logits)
    batch, seq_len, vocab_size = logits.shape
    idx_b = np.arange(batch)[:, None]
    idx_s = np.arange(seq_len)[None, :]
    correct_probs = probs[idx_b, idx_s, targets]
    loss = -np.mean(np.log(correct_probs + 1e-9))

    dLogits = probs.copy()
    dLogits[idx_b, idx_s, targets] -= 1
    dLogits /= (batch * seq_len)
    return loss, dLogits
