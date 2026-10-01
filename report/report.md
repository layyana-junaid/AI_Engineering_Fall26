# Building a Small GPT From Scratch: What a 0.8 M-Parameter Transformer Learns From Shakespeare

*Technical report - AI Engineering, Transformer Deep Dive. Code: `<YOUR_REPO_URL>`. Figures are in `outputs/figures/`; every number below is produced by a command in the README and logged in `outputs/logs/`.*

**Summary.** I built a decoder-only Transformer (4 layers, 4 heads, d_model 128, 810,240 parameters) in PyTorch with every Transformer component written by hand, trained it from random initialisation on Tiny Shakespeare at character level, and packaged it behind a CLI and a FastAPI service. It reaches **validation loss 1.645 nats/char (perplexity 5.18, 2.37 bits/char)**, against 4.19 for a uniform guess. It produces correctly formatted plays with real words but also invented ones. Controlled ablations show that depth helps, head count does not matter measurably at this scale, shorter context trained faster in a short budget, and fixed sinusoidal positions beat learned ones. I also report what does *not* work and where my evidence is weak.

---

## 1. Problem and scenario

A small team has a private corpus too small to justify a large foundation model, but large enough to teach a compact model the style and vocabulary of a narrow domain. The question to answer: **can a small Transformer trained from scratch learn the next-token structure of a domain corpus well enough to generate coherent text?** I use Tiny Shakespeare as the stand-in corpus and build the full pipeline: tokenizer -> batches -> embedding + position -> causal Transformer blocks -> vocabulary logits -> next-token prediction -> generated text.

Success criteria I set (and report against): (a) the architecture is demonstrably correct (hand calculation, unit tests, causality test); (b) training works (tiny-overfit gate, then loss curves); (c) conclusions are supported by controlled experiments; (d) a stranger can reproduce everything from a clean checkout.

## 2. Transformer foundations

### 2.1 Why Transformers (sequential vs. parallel depth)

A recurrent layer computes `h_t = tanh(W x_t + U h_{t-1})`: step *t* needs step *t-1*, so the number of **sequentially dependent operations is O(T)** and the path between token 1 and token T is T steps long, which is where vanishing gradients come from. Self-attention connects every pair of positions directly: **O(1) sequential operations per layer and a path length of 1** between any two tokens. The price is arithmetic and memory: attention costs O(T^2 d) compute and stores a T x T weight matrix per head (O(T^2) memory), where a recurrent layer costs O(T d^2) and stores O(T d).

*(My own earlier neuron-chain experiment, `llm.ipynb`, showed the gradient problem in miniature: with sigmoid activations the gradient reaching the first neuron is multiplied by sigma'(z) <= 0.25 at every layer, so the first layer barely learns. Residual connections and LayerNorm are the Transformer's answer.)*

**Measurement** (`experiments/seq_vs_parallel.py`; one RNN layer vs. one Transformer block, d = 128, batch 16, forward + backward, 2-core CPU, best of 3):

| T | RNN layer (ms) | Transformer block (ms) | RNN sequential steps | attention weights (MB) | RNN hidden states (MB) |
|---:|---:|---:|---:|---:|---:|
| 32 | 10.1 | 7.1 | 32 | 0.25 | 0.25 |
| 64 | 12.8 | 16.8 | 64 | 1.0 | 0.5 |
| 128 | 41.0 | 26.4 | 128 | 4.0 | 1.0 |
| 256 | 84.8 | 84.9 | 256 | 16.0 | 2.0 |
| 512 | 317.0 | 347.1 | 512 | 64.0 | 4.0 |

*Observation:* on this machine the wall-clock times are roughly equal (the Transformer is faster at T = 128 and slower at T = 512), while attention memory grows quadratically (64 MB vs. 4 MB at T = 512). *Interpretation (reasoned, not measured):* two CPU cores cannot exploit T-way parallelism, so the headline advantage does not appear here; but the Transformer block performs about 12 d^2 multiply-adds per token (4 projections + a 4x FFN) versus 2 d^2 for the RNN, so it does roughly 6x the arithmetic in about the same time, which is the practical meaning of better parallel utilisation. On a GPU the sequential-depth difference is what dominates. The quadratic memory is why context length is a real design constraint (Section 7, context ablation).

### 2.2 Scaled dot-product attention

With queries Q, keys K, values V of shape (T, d_k):

```
scores  = Q K^T / sqrt(d_k)            (T, T)
weights = softmax(scores + mask, dim=-1)
output  = weights V                    (T, d_v)
```

Dividing by sqrt(d_k) keeps the variance of the scores near 1: if q and k have unit-variance components, q.k has variance d_k, and large scores push the softmax into a near one-hot regime with vanishing gradients (tested in `test_scaling_by_sqrt_dk_matters`).

### 2.3 Hand-calculated example (required artifact, Figure 1)

Three tokens, d_k = d_v = 2: Q = K = [[1,0],[0,1],[1,1]], V = [[1,0],[0,2],[3,1]].

* Q K^T = [[1,0,1],[0,1,1],[1,1,2]]; divided by sqrt(2) = 1.4142 gives [[.707, 0, .707],[0, .707, .707],[.707, .707, 1.414]].
* Row-wise softmax: row 1 = [e^.707, e^0, e^.707] / 5.056 = **[0.4011, 0.1978, 0.4011]**; rows 2, 3 = [0.1978, 0.4011, 0.4011] and [0.2483, 0.2483, 0.5035].
* Output = weights V: row 1 = 0.4011 [1,0] + 0.1978 [0,2] + 0.4011 [3,1] = **[1.6044, 0.7967]**; rows 2, 3 = [1.4011, 1.2033] and [1.7587, 1.0000].
* **Causal mask** (Figure 2): add -inf above the diagonal -> masked scores [[.707, -inf, -inf],[0, .707, -inf],[.707, .707, 1.414]] -> weights [[1, 0, 0],[0.3302, 0.6698, 0],[0.2483, 0.2483, 0.5035]] -> output rows [1, 0], [0.3302, 1.3395], [1.7587, 1.0].
* **What changes:** unmasked, each token mixes information from all positions, including the future. Masked, token *i* receives a convex combination of tokens 1..*i* only. Token 1 can only copy its own value (output = v_1 exactly); token 3, which may already see everyone, is unchanged. Information flows strictly left to right, which is what makes next-token training honest.

`experiments/hand_calculation.py` prints every step and asserts that the library function reproduces these numbers; `tests/test_attention.py` pins them.

### 2.4 Multi-head attention and tensor shapes (Figure 3)

With H heads and d_k = d_model / H: project x (B,T,D) with W_q, W_k, W_v (each D x D), **split** to (B,H,T,d_k) via `view` + `transpose`, attend independently per head, **merge** back to (B,T,D) (this is the concatenation), and apply W_o. The shape trace for d_model 128, 4 heads, B = 2, T = 16 is: input (2,16,128) -> Q split (2,4,16,32) -> weights (2,4,16,16) -> per-head output (2,4,16,32) -> concatenated (2,16,128) -> W_o (2,16,128). Test `test_split_heads_puts_contiguous_feature_chunks_in_each_head` checks that head 0 receives features 0-3 and head 1 features 4-7, and that `merge(split(x)) == x`.

### 2.5 Positional encoding (Figure 4)

Attention is permutation-equivariant, so order must be injected. I add `PE[pos, 2i] = sin(pos / 10000^(2i/d))`, `PE[pos, 2i+1] = cos(...)`. Low dimensions oscillate quickly (wavelength about 6 positions), high dimensions slowly, so every position gets a unique signature, and the dot product of PE[40] with other positions peaks at 40 and decays smoothly with distance. Adding the same token embedding at positions 0-5 gives pairwise L2 distances 1.95 (neighbours), 3.64, 4.86, 5.55, 5.80, so identical tokens become distinguishable only through position, and distance grows with offset.

### 2.6 Full block and architecture (Figure 5, architecture diagram)

Pre-LN decoder block: `x = x + Dropout(MHA(LN1(x), causal_mask)); x = x + FFN(LN2(x))`, with FFN = Linear(d, 4d) -> GELU -> Linear(4d, d). The model is `token embedding (x sqrt(d)) + PE -> N blocks -> final LayerNorm -> Linear(d, V)`. Parameters per block = 12 d^2 + 13 d = 198,272 at d = 128; total = V d + N (12 d^2 + 13 d) + 2 d + d V = 8,448 + 793,088 + 256 + 8,448 = **810,240** (asserted in `test_parameter_count_formula`).

### 2.7 BERT-style masked LM vs. GPT causal LM

Both use the same Transformer; they differ in the attention mask and the objective. **GPT** factorises P(x) = prod_t P(x_t | x_<t) and uses a lower-triangular mask so each position sees only its past; training on every position at once is cheap and generation is just repeated sampling. **BERT** corrupts about 15 % of the *input* tokens with `[MASK]` and predicts only those from **both** sides with a full (all-ones) mask; this yields better representations for understanding tasks but there is no left-to-right factorisation to sample from. Mask matrices are in `fig_mlm_vs_clm_masks.png`.

**Experiment** (`experiments/mlm_vs_clm.py`; 2-layer, d = 64 model, 300 steps, identical data/objective/seed; only the mask differs):

| Model | final train loss | val loss, mask as trained | val loss with causal mask (what generation sees) |
|---|---:|---:|---:|
| causal (GPT) | 2.203 | 2.255 | 2.255 |
| bidirectional | 0.674 | 0.678 | **5.692** |

*Observation:* the two loss curves are identical until about step 200; then the bidirectional model's loss collapses while the causal model's keeps its slow decline. Evaluated the way generation works (causal mask), the bidirectional model is far *worse* than uniform guessing (5.69 > ln 66 = 4.19). *Interpretation:* with a full mask, position t can attend to token t+1, which is exactly its target, so it learns to read the answer instead of predicting it. The low loss is a leak, not language modelling. This is why a GPT must be causal, and `test_full_model_is_causal` / `test_bidirectional_model_is_NOT_causal` verify the property directly (changing future tokens does not change earlier logits).

## 3. Implementation

| Component | File | Notes |
|---|---|---|
| Tokenizer | `src/tokenizer.py` | char-level, ids by sorted code point (deterministic); id 0 = `<unk>` decoded as `?`; no BOS/EOS/PAD because training windows are fixed-length slices of one token stream. Vocab 66 = 65 chars + `<unk>`. |
| Dataset | `src/dataset.py` | windows of T+1 tokens -> `x = w[:-1]`, `y = w[1:]`; target at position t is the token after input t. |
| Attention | `src/attention.py` | own `scaled_dot_product_attention`, `MultiHeadAttention` with explicit `split_heads` / `merge_heads`; boolean strictly-upper-triangular mask filled with `-inf` before softmax. |
| Position | `src/positional_encoding.py` | sinusoidal (fixed buffer); learned table for the ablation. |
| Block / model | `src/transformer_block.py`, `src/gpt.py` | pre-LN, GELU FFN, residuals, final LN, untied LM head; GPT-2 style init. |
| Loss | `src/gpt.py` | `cross_entropy(logits.view(-1,V), targets.view(-1))`. |
| Training | `src/train.py` | AdamW, linear warm-up + cosine decay, grad clipping, seeded validation batches, JSONL logs, best-val checkpoint. |
| Generation | `src/generate.py` | autoregressive loop, sliding window of `context_length`, temperature, top-k, top-p, greedy. |

Design choices worth defending: pre-LN keeps an unnormalised identity path through the residual stream (GPT-2); the embedding is multiplied by sqrt(d_model) after N(0, 1/d) initialisation so token identity is not drowned out by position codes of magnitude about 1; residual output projections are initialised smaller by 1/sqrt(2N) so the residual variance does not grow with depth; weight decay is applied to matrices only. The untrained model's loss is 4.22 against ln(66) = 4.19, i.e. initialisation is neutral.

**Tests** (`python -m pytest -q`, 36 tests): attention vs. the hand example and vs. `F.scaled_dot_product_attention` used only as an independent oracle; future positions receive exactly zero weight; the whole model is causal; split/merge and shapes; positional-encoding values; parameter-count formula; targets are shifted by exactly one; tiny-overfit; checkpoint reload in a **fresh process** gives identical text; CLI prints the required fields. In addition, my original NumPy implementation with hand-derived backward passes (`reference/numpy_llm/`) is cross-checked against PyTorch autograd for attention, LayerNorm and softmax cross-entropy gradients (agreement to 1e-6).

## 4. Dataset and preprocessing

Tiny Shakespeare (Hugging Face `karpathy/tiny_shakespeare`; mirrored in `karpathy/char-rnn`): 1,115,394 characters, about 40,000 lines, 65 distinct characters. Split: the **last 10 %** (contiguous, no shuffling) is validation: 1,003,855 train / 111,539 validation tokens. No filtering. Because the split is contiguous, validation text comes from different plays than most training text, so the train-validation gap mixes overfitting with a small distribution shift. Training windows are 128 characters sampled uniformly at random; validation batches use a fixed seed so every run is scored on the same windows.

## 5. Training setup

| Item | Value |
|---|---|
| Hardware | 2 vCPU cloud container, no GPU (peak process memory 1.44 GB) |
| Model | 4 layers, 4 heads, d_model 128, FFN 512, context 128, dropout 0.1, sinusoidal; **810,240 params** |
| Optimiser | AdamW, betas (0.9, 0.95), weight decay 0.1 (matrices only), grad-clip 1.0 |
| Schedule | lr 1e-3; 100 warm-up steps; cosine decay to 1e-4 |
| Batch / steps | 64 x 128 tokens, 3,000 steps = 24.6 M tokens (about 24.5 passes over the training text) |
| Wall-clock | 2,310 s (38.5 min), 10,637 tokens/s |
| Evaluation | every 250 steps, 50 fixed-seed batches, dropout off; checkpoint on best validation loss |
| Seed | 1337 |

The learning rate (handout suggests about 3e-4) was **not swept**; 1e-3 is a standard choice for small character-level models and training was stable with no loss spikes. A sweep would be the first thing I would add. Debug gate before the long run: training on one fixed batch drives the loss from 4.20 to **0.069** in 300 steps (`overfit` run, also `test_tiny_overfit`), which shows that gradients flow and that input/target alignment is correct.

## 6. Results

**Loss and perplexity (Figure 6).** Validation loss falls from 4.19 (uniform) to **1.647 at step 3000** (full-split evaluation: **1.645**, perplexity **5.18**, 2.37 bits/char; train 1.450, perplexity 4.26). The curves have not flattened completely: validation improved from 1.693 (step 2000) to 1.647 (step 3000) while the learning rate was already decaying, and the train-validation gap is about 0.2 nats, so the model is capacity- and budget-limited rather than badly overfit.

**Generation (Figure 8, `outputs/samples/eval_samples.txt`; temperature 0.8, top-k 40, fixed seeds).** Example from the prompt `KING HENRY:` :

```
KING HENRY:
Ah, you are by have speak come many good from thou have
In ever him to thirs in the grief be to being
Pempt was you bark, my countly give majesty,
Thou hast the wat we cault with us dear our forth.

ROMEO:
Sir, forth, and good-morrow, good sorrow pro
```

*Observation:* the model has learned the format (speaker tags in capitals, colons, line breaks, short verse lines, punctuation), Elizabethan vocabulary and common phrases. *Observation:* it also produces non-words ("Pempt", "cault", "countly") and the dialogue is not semantically coherent; speakers change arbitrarily.

**Attention inspection (Figure 7).** For the sentence "ROMEO:\nO, speak again, bright angel!", the trained heads differ qualitatively (`outputs/logs/attention_stats.txt`): layer 0 head 1 is a sharp "previous character" head (50.3 % of its mass on the preceding character, 17.7 % on itself); layers 1-2 contain several heads attending to the previous few characters or to word beginnings; and **layer 3 head 2 places 61 % of its average attention (for queries after position 8) on the newline that ends the speaker tag and another 13 % on the colon**. *Interpretation (hypothesis, not tested):* that head acts as a delimiter / "attention sink" that lets the model remember "I am inside a speech by this speaker" or simply park unneeded attention. In all heads the upper triangle is exactly zero, confirming causality in the trained model.

**Inference.** CLI and API (Figure 10): `python -m src.generate --prompt "The future of artificial intelligence" --max-new-tokens 80` prints the prompt, the generation, `model parameters: 0.81M`, `context length: 128`, `temperature: 0.8`. The FastAPI endpoint returned 120 characters in 918 ms on the CPU container, and a request above the 1,000-token limit is rejected with HTTP 422.

## 7. Ablation study

Protocol: change **one** factor at a time from the baseline (4 layers, 4 heads, context 128, sinusoidal positions) and keep everything else fixed. Because 3,000-step runs for eight variants were too expensive on a CPU, every ablation uses **800 steps at batch 32 x 128 tokens (about 3.3 M tokens)**, a reduced budget, so absolute losses (about 2.1-2.2) are higher than the main run (1.65), and conclusions apply to this *early-training regime*. Context-length runs keep **tokens per step constant** (batch 64 / 32 / 16 for context 64 / 128 / 256), so they see the same amount of data. All runs are single-seed; to estimate noise I re-ran the baseline with a second seed: **2.138 vs. 2.157 (spread 0.018)**, so differences smaller than about 0.02 are not interpretable. Reproduce at full budget on Colab with `python -m experiments.ablation --steps 3000 --batch-size 64`.

| Run | Setting | Params | Val loss | Val ppl | Train loss | Time (s) | Tokens/s | Peak RSS (MB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| base | 4 layers / 4 heads / ctx 128 | 810,240 | 2.157 | 8.64 | 2.105 | 297 | 11,042 | 1,087 |
| base, seed 7 | (noise estimate) | 810,240 | 2.138 | 8.48 | 2.089 | 316 | - | - |
| depth2 | 2 layers | 413,696 | 2.226 | 9.26 | 2.190 | 150 | 21,847 | 889 |
| depth6 | 6 layers | 1,206,784 | 2.108 | 8.23 | 2.048 | 437 | 7,495 | 1,222 |
| heads2 | 2 heads | 810,240 | 2.161 | 8.68 | 2.113 | 257 | 12,767 | 989 |
| heads8 | 8 heads | 810,240 | 2.150 | 8.58 | 2.098 | 402 | 8,150 | 1,281 |
| ctx64 | context 64 | 810,240 | 2.055 | 7.81 | 1.997 | 249 | 13,144 | 987 |
| ctx256 | context 256 | 810,240 | 2.245 | 9.44 | 2.214 | 409 | 8,002 | 1,233 |
| learnedpos | learned positions | 826,624 | 2.260 | 9.58 | 2.228 | 301 | 10,878 | 1,064 |

(Peak RSS is process-wide on CPU and includes about 0.9 GB of PyTorch runtime, so it is a coarse proxy for memory; the analytic attention memory in Section 2.1 is the cleaner quantity.) Figure 9 plots the validation losses with the noise band.

**Depth.** *Observation:* validation loss 2.226 -> 2.157 -> 2.108 for 2 -> 4 -> 6 layers (improvements of 0.069 and 0.049, both larger than the 0.018 noise), at the cost of training time 150 -> 297 -> 437 s and parameters 0.41 -> 0.81 -> 1.21 M. *Interpretation:* extra layers let attention compose: the attention maps show early heads building simple local features (previous character) that later layers can use. Depth is confounded with parameter count here (I did not run a width-matched control), and the returns diminish (-0.069 then -0.049 for equal increments in layers and increasing cost).

**Heads.** *Observation:* 2.161 / 2.157 / 2.150 for 2 / 4 / 8 heads with identical parameter counts; all within the noise band; time rises with head count (257 -> 297 -> 402 s). *Interpretation:* I found **no measurable benefit** from more heads at this scale; I would not claim that 8 heads is better. A plausible reason for the cost is that more heads means more, smaller matrix multiplies and a larger attention tensor (B x H x T x T) on CPU, with d_k shrinking to 16.

**Context length.** *Observation:* with equal tokens per step, validation loss is **lowest for the shortest context** (2.055 at 64, 2.157 at 128, 2.245 at 256), the opposite of the naive expectation that more context is always better; throughput falls from 13.1k to 11.0k to 8.0k tokens/s. *Interpretation (hypotheses, not tested):* (i) in an 800-step budget the model is still learning local character statistics, which need little context, and attention over 256 positions is harder to train than over 64; (ii) the batch contains more distinct windows when the context is shorter (64 vs. 16), lowering gradient noise, which confounds the comparison. This result should **not** be read as "long context is harmful": it says that in a very short budget longer windows were slower to exploit. A converged comparison, or one with batch size held constant, would be needed.

**Position method.** *Observation:* sinusoidal 2.157 vs. learned 2.260 (+0.103, far above noise), with learned positions adding 16,384 parameters. *Interpretation:* the fixed sinusoidal code supplies a smooth, distance-aware structure for free, while a learned table must discover it from data in few steps; with longer training the gap might close (not tested).

**Generation strategy** (optional experiment; `experiments/generation_strategy.py`, 5 prompts x 400 characters; "valid-word rate" = share of generated words that occur in the training corpus):

| Strategy | valid-word rate | distinct-2 (word bigrams) | repeated 4-gram share |
|---|---:|---:|---:|
| greedy (T = 0) | 0.992 | 0.117 | 0.830 |
| T = 0.5 | 0.958 | 0.878 | 0.221 |
| T = 0.8, top-k 40 | 0.893 | 0.975 | 0.120 |
| T = 0.8, top-p 0.9 | 0.941 | 0.938 | 0.153 |
| T = 1.2 | 0.718 | 0.989 | 0.052 |

Lower temperature trades diversity for correctness of words; there is no setting that is best on all three metrics. T = 0.8 with top-p 0.9 is a good compromise on this model.

## 8. Failure analysis

1. **Greedy decoding collapses into loops.** Sample (prompt `ROMEO:`): "I will not the see to the see the see ... That the seen of the seen of the seen of the seen". 83 % of its character 4-grams are repeats and only 11.7 % of word bigrams are distinct, versus 12 % and 97.5 % under T = 0.8 top-k 40. *Likely cause:* argmax always follows the single most probable continuation; once a short cycle appears in the context, the model's own output makes the cycle more likely still, and a small model with a 128-character window has no mechanism to notice the loop. Sampling breaks the cycle. The valid-word rate of 99.2 % shows that the loop is made of real words, i.e. fluent but empty.
2. **Invented words and no long-range coherence.** At T = 0.8 / top-k 40 about 11 % of generated words do not exist in the corpus (28 % at T = 1.2), and scenes have no thread: speakers change arbitrarily. *Likely causes:* a character-level model has to spell every word, so one wrong character makes a non-word; 0.8 M parameters and 2.37 bits/char leave substantial uncertainty per character; and a 128-character window is about 20 words, far shorter than a scene. Larger models, subword tokenization and longer context are the obvious remedies; I did not test them.
3. **Information leak under a bidirectional mask** (Section 2.7): loss 0.678 looks excellent but is 5.69 under the causal mask. A reminder that low training loss is meaningless unless the architecture forbids access to the target.
4. **Limits of my evidence.** Ablations are single-seed, at a reduced budget, with a noise estimate from only one extra seed; the learning rate was not tuned; attention-head interpretations are hypotheses.

## 9. Engineering packaging

* **Reproducibility:** `README.md` lists exact commands; all hyper-parameters live in `configs/tiny_gpt.yaml` and can be overridden with `--set key=value`; seeds are fixed; data is downloaded by `python -m src.dataset --download`; checkpoint (3 MB) is in the repository.
* **CLI:** `python -m src.generate --prompt ... --max-new-tokens ... --temperature ...`.
* **API:** `uvicorn src.api:app` serves `GET /health` and `POST /generate` (validated inputs; seed, temperature, top-k, top-p); `render.yaml` and `Dockerfile` deploy it (CPU-only PyTorch via `requirements-api.txt`). No secrets are used or committed.
* **Colab:** `notebooks/colab_minigpt.ipynb` runs tests, training, evaluation, attention maps and ablations end to end.
* **Acceptance tests (handout Section 16)** - run in a fresh clone: install; 36 unit tests pass; tiny-overfit 4.20 -> 0.069; checkpoint reload in a new process; three fixed-prompt generations; causal-correctness test; README commands verified.

## 10. Conclusion

A 0.8 M-parameter, 4-layer GPT trained from scratch for 38 minutes on a CPU learns the *surface structure* of Shakespeare - format, spelling patterns, vocabulary - to 2.37 bits per character, but not its meaning. The evidence for each architectural idea: the hand calculation and tests show attention and masking behave as derived; the mask experiment shows that causality, not capacity, is what makes next-token training honest; the ablations show that depth (+0.05 to +0.07 loss improvement per step up in the 2-4-6 series) and fixed sinusoidal positions (+0.10 over learned) matter in a short budget, while head count does not measurably at this size; and the failure analysis shows that decoding strategy matters as much as the weights (greedy loops vs. sampled diversity). The main limitations are scale and evidence quality (single seeds, reduced ablation budget); the next steps I would take are a learning-rate sweep, multiple seeds, a width-matched depth control, and a subword tokenizer.

---

### Sources and attribution

Vaswani et al. 2017, *Attention Is All You Need* (attention, multi-head, sinusoidal encoding). Radford et al. 2019, *Language Models are Unsupervised Multitask Learners* (pre-LN block, initialisation, GPT recipe). Devlin et al. 2018, *BERT* (masked LM). Karpathy, char-rnn / Tiny Shakespeare. PyTorch is used for tensors, autograd, `nn.Linear`, `nn.LayerNorm`, AdamW; `F.scaled_dot_product_attention` appears only in unit tests as an independent oracle. The model code is my own; no pretrained weights are loaded.
