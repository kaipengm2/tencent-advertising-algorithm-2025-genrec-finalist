# 🏆 A Simple Ticket to the Finals — [2025 Tencent Advertising Algorithm Competition](https://algo.qq.com/)

### *Generative Recommendation Challenge*

This repository presents a **simple yet effective generative next-item recommendation framework**.
Despite using **no multimodal inputs**, the model achieves:

* **Score:** 0.129843
* **NDCG@10:** 0.104463
* **HitRate@10:** 0.186332

and successfully advances to the **finals**.

### ⭐ Highlights

* Mixed negative sampling (in-batch + global)
* Log-q correction
* Hard negative mining & curriculum learning
* Action-conditioned generation
* A purely end-to-end, single-stage training pipeline, free of handcrafted feature engineering or multi-stage complexity.

For more details, see [METHOD.md](METHOD.md).

---

## 📦 Requirements

* polars
* numpy, tqdm, tensorboard

Tested on:
- Python 3.10
- PyTorch 2.7.1 + CUDA 12.3
- 1 NVIDIA H20 (96GB)

Install example:

```bash
pip install "torch==2.7.1" polars numpy tqdm tensorboard
```

---

## 🌱 Environment Variables

All I/O directories are configured via environment variables.
`run.sh` sets these by default, but you may override them manually or via CLI:

* `TRAIN_DATA_PATH` — dataset directory
* `TRAIN_CKPT_PATH` — checkpoint directory
* `TRAIN_LOG_PATH` — log directory (`train.log`)
* `TRAIN_TF_EVENTS_PATH` — TensorBoard events directory
* `USER_CACHE_PATH` — cache directory for precomputed data

Example override:

```bash
export TRAIN_DATA_PATH=/path/to/data
export TRAIN_CKPT_PATH=/path/to/ckpts
export TRAIN_LOG_PATH=/path/to/logs
export TRAIN_TF_EVENTS_PATH=/path/to/tfevents
export USER_CACHE_PATH=/path/to/usrcache
```

---

## 📁 Data

The dataset structure follows the competition’s fully anonymized format.

### **Training Input**

**`seq.jsonl`**: Each line contains the full sequence of events associated with a single user.
Every event is represented in the following format:

* **User re-index ID**
* **Item re-index ID**
* **User features** — a dictionary, e.g. `{'feat_id': feat_val, ...}`
* **Item features** — a dictionary, e.g. `{'feat_id': feat_val, ...}`
* **Action type**

  * `0` = *impression*
  * `1` = *click*
  * `2` = *conversion*
* **Timestamp** (in seconds)

```text
[user_id, item_id, user_feat, item_feat, action, ts]
```

In addition to `seq.jsonl`, the following files are involved:

* **`data/seq_offsets.pkl`**
  Byte offsets for random-access reads in multi-worker DataLoaders.
  Each entry maps a user index to its starting byte in `seq.jsonl`.

* **`data/indexer.pkl`**
  Centralized ID vocabularies:

  * `indexer["i"]`: raw `creative_id` → re-indexed item id `[1, itemnum]`
  * `indexer["u"]`: raw user id → re-indexed user id
  * `indexer["f"][fid]`: vocabulary for feature `fid`
    Used to derive `itemnum`, `usernum`, and the vocabulary sizes for all features.

* **`data/item_feat_dict.json`**
  Provides per-item non-embedding side features, consistent with those recorded in `seq.jsonl`:

  ```json
  {
    "1": { "100": 12, "117": 3, ... },
    "2": { ... }
  }
  ```

For evaluation, the following additional files are used:

* `eval/predict_seq.jsonl` — user histories (same format as `seq.jsonl`)
* `eval/predict_seq_offsets.pkl` — byte offsets for the file above
* `eval/user_action_type.json` — target action type for each user:

  ```json
  { "user_123": 3, "user_456": 1, ... }
  ```

---

**Optional Dense Multimodal Embeddings (memmap):**

While multimodal inputs are currently *unused*, the implementation supports dense multimodal item embeddings.

* Raw embeddings are stored under:

  * `data/creative_emb/emb_{fid}_{D}.pkl` (for `fid=81`), or
  * `data/creative_emb/emb_{fid}_{D}/*.json` (other `fid` values)

* On first access, they are converted into memory-mapped tables:

  * Path: `$USER_CACHE_PATH/emb_table_{fid}_{D}.mmap`
  * Shape: `(itemnum + 1, D)` with row `0` as a zero/padding vector

To disable multimodal embeddings, omit `creative_emb` or skip setting `mm_emb_id` in `args`.

---

## 🚀 Training Workflow

### **Preprocess + train from scratch**

Simply run:

```bash
./run.sh
```

This will execute:

1. `get_stat.py`
2. `main.py` with all paths sourced from `run.sh`

**Note:** The `run.sh` script automatically sets the project directory and required I/O environment variables. If this has already been done, all you need to do is run:

```bash
python -u get_stat.py
python -u main.py
```

---

In addition, you can also resume training from an existing checkpoint.

```bash
python -u train.py --resume checkpoints/global_stepXXXX/ckpt.pt
```

---

## 🔍 Inference — Top-K Retrieval

### **1. Prepare evaluation data**

* Place evaluation files in `eval/` (or override `EVAL_DATA_PATH`)

---

### **2. Run inference**

Basic usage:

```bash
./infer.sh MODEL_DIR [TOPK]
```

* `MODEL_DIR` — directory containing a `.pt` checkpoint (e.g. `checkpoints/global_stepXXXX`)
* `TOPK` — number of retrieved items (default: 10)

Example:

```bash
./infer.sh checkpoints/global_step136000 10
```

This will:

1. Set `MODEL_OUTPUT_PATH=MODEL_DIR` (can also be configured via an environment variable)
2. Set `EVAL_DATA_PATH` (default: `./eval`, also overridable through an environment variable)
you can also set these environment var yourself
3. Run:

   ```bash
   python -u infer.py --topk 10
   ```

### **Reproducibility**

**The model at `global_step136000` has been verified to achieve a score of **0.129843**, with the result being fully reproducible across runs.**

---

## 📄 License

SPDX-License-Identifier: Apache-2.0

This project is licensed under the Apache License 2.0.

---

## 🙏 Acknowledgements

Developed upon the baseline of the 2025 Tencent Advertising Algorithm Competition  


References the official HSTU implementation (“Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations”).
