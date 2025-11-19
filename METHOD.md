# Method Overview

We tackle next-item prediction using a sampled-softmax objective, drawing negative examples from both in-batch positives and a global random pool. Training is further enhanced through in-batch hard top-k negative mining and a curriculum schedule that adapts difficulty over time.

## Loss Function

Given query **Q**, positive **K₊**, negatives **N**, and temperature **τ**,
(similarity is dot-product after L2-normalization → cosine):

$$
\mathcal{L} = -\log
\frac{\exp(s_+/\tau - \log q(i_+))}{
\exp(s_+/\tau - \log q(i_+)) + \sum_{j\in N}\exp(s_j/\tau - \log q(i_j))}
,\quad s_\bullet=\langle Q,K_\bullet\rangle
$$

All logits subtract **log q(·)** for **importance sampling correction**, making the result **unbiased** to full-softmax.

---

## Preparation

### 1. Frequency Statistics (`log q`)

Use `polars` to count item frequencies and build
$$
q(i)\propto \mathrm{freq}(i)^{0.75}
$$

### 2. Global Negative Pool

Keep items whose **last appearance > May 29** (i.e., still active).
Exclude items already shown in the current sequence when sampling.

---

## Mixed Negative Sampling

### A. In-Batch

* Negatives = other positives in the same batch
* Sampling bias ≈ item popularity → correct by subtracting **log q**
* Exclude:

  * Same-sequence items
  * Items with similarity > 0.99
* Positive samples also subtract **log q** to emphasize **rare / cold items**

### B. Global Random

* Randomly sample from the active item pool (excluding current sequence)

---

## In-Batch Hard Negative Mining (Top-k)

**Motivation:** late training = most negatives easy → dilute gradient.
Keep only **top-k hardest** negatives per batch.

To balance sources, apply top-k **separately** for in-batch and global negatives.

---

## Curriculum Learning

Gradually reduce `hard_topk` as training progresses:

---

## Feature Modeling

### 1. Item Tower (End-to-End)

Item ID and sparse features → embeddings → `itemdnn` (MLP).
No handcrafted features.

### 2. User as BOS + FiLM Conditioning

User features = **first token (mask = 2)**, `pos_emb = 0`.
Aggregated user vector → `(γ, β)` → FiLM modulation:
$$
x_t' = x_t \cdot (1+\gamma) + \beta
$$

### 3. Time & Action

* **Cyclic time**: sin/cos(hour, weekday)
* **Relative Attention Bias (RAB)**: log-bucketed time difference as additive bias
* **Action type**: added via embedding to input (applied across all layers)

---

## Model Architecture: HSTU

**Single Layer Steps**

1. Linear → **U/V/Q/K** (multi-head split)
2. Compute logits:
   $$
   \frac{QK^\top}{\sqrt{d_k}} + \mathrm{RAB}
   $$
   → apply **SiLU** instead of softmax → weights A
3. Aggregate Y = A @ V
4. Normalize Y and **gate** with U
5. Linear projection + residual connection

---

## Inference

* Read `user_action_type.json` to set each user’s next-action condition
* Exclude items already seen in user history during sampling

---


