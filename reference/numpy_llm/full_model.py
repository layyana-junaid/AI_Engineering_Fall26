import numpy as np
from core import *

## implementing the core model that is going to work all tgt
##basically similar to the transformer

def init_transformer(vocab_size, embed_dim, num_heads, num_blocks, ffn_hidden, seed=42):
    np.random.seed(seed)
    params = {}
    params["E"] = init_embedding(vocab_size, embed_dim, scale=0.1)
    params["blocks"] = []
    for _ in range(num_blocks):
        Wq, bq = init_linear(embed_dim, embed_dim)
        Wk, bk = init_linear(embed_dim, embed_dim)
        Wv, bv = init_linear(embed_dim, embed_dim)
        Wo, bo = init_linear(embed_dim, embed_dim)
        W1, b1 = init_linear(embed_dim, ffn_hidden)
        W2, b2 = init_linear(ffn_hidden, embed_dim)
        block = dict(
            attn=dict(Wq=Wq, bq=bq, Wk=Wk, bk=bk, Wv=Wv, bv=bv, Wo=Wo, bo=bo),
            ffn=dict(W1=W1, b1=b1, W2=W2, b2=b2),
            gamma1=np.ones((1, 1, embed_dim)), beta1=np.zeros((1, 1, embed_dim)),
            gamma2=np.ones((1, 1, embed_dim)), beta2=np.zeros((1, 1, embed_dim)),
        )
        params["blocks"].append(block)
    params["W_out"], _ = init_linear(embed_dim, vocab_size)
    return params

def model_forward(token_ids, params, num_heads, vocab_size, embed_dim):
    batch, seq_len = token_ids.shape
    embedded, embed_cache = embedding_forward(token_ids, params["E"])
    pos = positional_encoding(seq_len, embed_dim)
    X = embedded + pos   # broadcasts (seq_len, embed_dim) onto (batch, seq_len, embed_dim)

    mask = causal_mask(seq_len)
    block_caches = []
    for block in params["blocks"]:
        X, cache = transformer_block_forward(X, block, num_heads, mask)
        block_caches.append(cache)

    logits, out_cache = output_projection_forward(X, params["W_out"])
    full_cache = dict(embed_cache=embed_cache, block_caches=block_caches, out_cache=out_cache)
    return logits, full_cache

def model_backward(dLogits, cache, params, vocab_size, embed_dim):
    dX, dW_out = output_projection_backward(dLogits, cache["out_cache"], params["W_out"])

    block_grads = []
    for block, block_cache in zip(reversed(params["blocks"]), reversed(cache["block_caches"])):
        dX, grads = transformer_block_backward(dX, block_cache, block)
        block_grads.append(grads)
    block_grads.reverse()

    dE = embedding_backward(dX, cache["embed_cache"], params["E"].shape[0], embed_dim)
    return dict(dE=dE, dW_out=dW_out, block_grads=block_grads)


def flatten_params(params, num_blocks):
    flat = {"E": params["E"], "W_out": params["W_out"]}
    for i, block in enumerate(params["blocks"]):
        for k, v in block["attn"].items():
            flat[f"b{i}_attn_{k}"] = v
        for k, v in block["ffn"].items():
            flat[f"b{i}_ffn_{k}"] = v
        flat[f"b{i}_gamma1"] = block["gamma1"]; flat[f"b{i}_beta1"] = block["beta1"]
        flat[f"b{i}_gamma2"] = block["gamma2"]; flat[f"b{i}_beta2"] = block["beta2"]
    return flat

def flatten_grads(grads, num_blocks):
    flat = {"dE": grads["dE"], "dW_out": grads["dW_out"]}
    for i, bg in enumerate(grads["block_grads"]):
        for k, v in bg["attn"].items():
            flat[f"db{i}_attn_{k[1:]}"] = v
        for k, v in bg["ffn"].items():
            flat[f"db{i}_ffn_{k[1:]}"] = v
        flat[f"db{i}_gamma1"] = bg["dgamma1"]; flat[f"db{i}_beta1"] = bg["dbeta1"]
        flat[f"db{i}_gamma2"] = bg["dgamma2"]; flat[f"db{i}_beta2"] = bg["dbeta2"]
    return flat

def apply_adam_to_all(flat_params, flat_grads, m_state, v_state, t, lr):
    for key in flat_params:
        gkey = "d" + key
        if m_state[key] is None:
            m_state[key] = np.zeros_like(flat_params[key])
            v_state[key] = np.zeros_like(flat_params[key])
        flat_params[key], m_state[key], v_state[key] = adam(
            flat_params[key], flat_grads[gkey], m_state[key], v_state[key], t, learning_rate=lr)
    return flat_params, m_state, v_state

def unflatten_into_params(flat_params, params, num_blocks):
    params["E"] = flat_params["E"]
    params["W_out"] = flat_params["W_out"]
    for i, block in enumerate(params["blocks"]):
        for k in block["attn"]:
            block["attn"][k] = flat_params[f"b{i}_attn_{k}"]
        for k in block["ffn"]:
            block["ffn"][k] = flat_params[f"b{i}_ffn_{k}"]
        block["gamma1"] = flat_params[f"b{i}_gamma1"]; block["beta1"] = flat_params[f"b{i}_beta1"]
        block["gamma2"] = flat_params[f"b{i}_gamma2"]; block["beta2"] = flat_params[f"b{i}_beta2"]
    return params
