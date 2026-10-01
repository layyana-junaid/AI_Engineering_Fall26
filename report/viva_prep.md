# Viva prep - be able to answer these without looking

1. **Why divide by sqrt(d_k)?** Dot products of d_k-dim unit-variance vectors have variance d_k; large scores saturate softmax -> tiny gradients. (tests/test_attention.py::test_scaling_by_sqrt_dk_matters)
2. **How does the causal mask work and why -inf, not 0?** Added before softmax; exp(-inf)=0 so the weight is exactly 0 and rows still sum to 1. A 0 would still leave weight on the future.
3. **Walk through multi-head shapes.** (B,T,D) -> Q,K,V -> (B,H,T,d_k) -> weights (B,H,T,T) -> (B,H,T,d_k) -> merge (B,T,D) -> W_o. Heads see different learned subspaces; merge = concatenation.
4. **Why positional encoding?** Attention is permutation-equivariant. Sinusoids: unique per position, smooth, bounded; relative offsets are linear functions of the encodings.
5. **Pre-LN vs post-LN?** Pre-LN keeps a clean identity residual path -> stable without long warm-up (GPT-2). Mine is pre-LN. (Your old NumPy model was post-LN - know the difference.)
6. **Why residual connections / LayerNorm?** Gradient highway (your neuron-chain notebook: sigmoid derivative <= 0.25 per layer); normalisation keeps activation scale stable.
7. **Why GELU, why 4x FFN?** Smooth nonlinearity; FFN holds most parameters (8d^2 of 12d^2 per block) and acts as per-token memory.
8. **How are targets built?** Window of T+1 tokens; x = w[:-1], y = w[1:]; loss at every position. tests/test_overfit.py::test_targets_are_inputs_shifted_by_one
9. **BERT vs GPT?** Mask (bidirectional vs causal) and objective (masked tokens vs next token). Show the leak experiment: loss 0.678 but 5.69 under causal mask.
10. **Where do the 810,240 parameters live?** 198,272 per block x4 + 2x8,448 embedding/head + 256 final LN. 12d^2+13d per block.
11. **What is perplexity?** exp(cross-entropy). 5.18 means the model is as uncertain as a fair 5-way choice per character. Uniform would be 66.
12. **Why does greedy loop?** Self-reinforcing context; argmax has no escape. 83% repeated 4-grams.
13. **What did the ablations NOT show?** Heads: within noise (seed spread 0.018). Context-length result is confounded with batch size and early training.
14. **Why is your NumPy version not the submission?** ~3.5 s/step; PyTorch for speed; NumPy kept as reference with gradient cross-check.
15. **What would you do next?** LR sweep, multiple seeds, width-matched depth control, subword tokenizer, longer context at convergence.
