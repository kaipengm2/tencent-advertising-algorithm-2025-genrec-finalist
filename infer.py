"""
Top-K retrieval inference.

- Load BaselineModel checkpoint from $MODEL_OUTPUT_PATH
- Build L2-normalized candidate embeddings from $EVAL_DATA_PATH/predict_set.jsonl
- Encode users and run chunked GPU similarity (mask seen items)
- Output per-user Top-K creative_ids and user_ids

Usage: python infer.py --topk 10
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MyTestDataset
from model import BaselineModel


def get_ckpt_path():
    """Locate a .pt checkpoint file under MODEL_OUTPUT_PATH env var."""
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if ckpt_path is None:
        raise ValueError("MODEL_OUTPUT_PATH is not set")
    for item in os.listdir(ckpt_path):
        if item.endswith("model.pt"):
            return os.path.join(ckpt_path, item)
    raise FileNotFoundError("No .pt checkpoint under MODEL_OUTPUT_PATH")


def get_args():
    """Parse CLI arguments used for data, model, and inference."""
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--maxlen', default=101, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--state_dict_path', default=None, type=str)

    # Model (keep consistent with training)
    parser.add_argument('--hidden_units', default=128, type=int)
    parser.add_argument('--num_blocks', default=16, type=int)
    parser.add_argument('--num_epochs', default=2, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--drop_path_rate', default=0.0, type=float)
    parser.add_argument('--emb_dropout', default=0.2, type=float)
    parser.add_argument('--ffn_dropout', default=0.2, type=float)
    parser.add_argument('--attn_out_dropout', default=0.0, type=float)
    parser.add_argument('--sdpa_dropout', default=0.0, type=float)
    parser.add_argument('--norm_first', action='store_true')

    # Inference config
    parser.add_argument('--topk', default=10, type=int)
    parser.add_argument('--cand_encode_bs', default=8192, type=int,
                        help='Batch size for candidate encoding')
    parser.add_argument('--chunk_items', default=65534, type=int,
                        help='Candidate chunk size for GPU similarity')

    # Multimodal embedding IDs (same as training)
    parser.add_argument('--mm_emb_id', nargs='+', default=[], type=str, choices=[str(s) for s in range(81, 87)])
    return parser.parse_args()


def process_cold_start_feat(feat):
    """Set unseen string values to 0; for lists, process each element."""
    processed_feat = {}
    for feat_id, feat_value in feat.items():
        if isinstance(feat_value, list):
            processed_feat[feat_id] = [0 if isinstance(v, str) else v for v in feat_value]
        elif isinstance(feat_value, str):
            processed_feat[feat_id] = 0
        else:
            processed_feat[feat_id] = feat_value
    return processed_feat


@torch.no_grad()
def build_candidate_matrix_from_predict_set(indexer_i, feat_types, feat_default_value, mm_tables,
                                             model, device, batch_size=4096):
    """
    Vectorize only the candidates in predict_set.jsonl (aligned with eval path).
    Returns:
      cand_embs_cpu: (C, D) float32, L2-normalized on CPU
      creative_ids:  list of length C, aligned with rows
      cand_item_ids: (C,) torch.long, 1-based item_id aligned with rows (0=padding/missing)
    """
    candidate_path = Path(os.environ.get('EVAL_DATA_PATH'), 'predict_set.jsonl')
    item_ids, creative_ids, features = [], [], []

    # Load candidate features and align with item_id
    with open(candidate_path, 'r') as f:
        for line in f:
            rec = json.loads(line)
            feat = rec['features']
            creative_id = rec['creative_id']
            item_id = indexer_i.get(creative_id, 0)  # 0 means padding

            # Fill missing fields + cold start handling
            missing = set(feat_types['item_sparse'] + feat_types['item_array'] + feat_types['item_continual']) - set(feat.keys())
            feat = process_cold_start_feat(feat)
            for fid in missing:
                feat[fid] = feat_default_value[fid]

            # Multimodal embeddings: memmap lookup by item_id; item_id=0 -> zero vector
            for fid in feat_types['item_emb']:
                D = mm_tables[fid].shape[1]
                if item_id > 0:
                    # Note: memmap slicing is a view; convert to ndarray with correct dtype
                    feat[fid] = np.asarray(mm_tables[fid][item_id], dtype=np.float32)
                else:
                    feat[fid] = np.zeros(D, dtype=np.float32)

            item_ids.append(item_id)
            creative_ids.append(creative_id)
            features.append(feat)

    # Encode in blocks to control memory
    all_blocks = []
    for start in tqdm(range(0, len(item_ids), batch_size), desc="Encode candidates"):
        end = min(start + batch_size, len(item_ids))
        ids_np = np.asarray(item_ids[start:end], dtype=np.int64)
        item_seq = torch.from_numpy(ids_np).to(device).unsqueeze(0)  # (1, M)
        feats_block = np.asarray(features[start:end], dtype=object)
        emb = model.feat2emb(item_seq, [feats_block], include_user=False).squeeze(0)  # (M, D)
        emb = F.normalize(emb, dim=-1)
        all_blocks.append(emb.cpu())

    cand_embs_cpu = torch.cat(all_blocks, dim=0) if all_blocks else torch.zeros(0, model.repr_dim).cpu()
    cand_item_ids = torch.from_numpy(np.asarray(item_ids, dtype=np.int64))  # (C,)
    return cand_embs_cpu, creative_ids, cand_item_ids


@torch.no_grad()
def batched_topk_indices(q_vecs: torch.Tensor,
                         item_embs_cpu: torch.Tensor,
                         device: str,
                         k: int = 10,
                         chunk_items: int = 65536,
                         # Flattened (row_idx, global_col_idx) pairs to ban
                         banned_row_index: torch.Tensor = None,
                         banned_global_col_index: torch.Tensor = None):
    """
    Compute similarity on GPU by chunking candidates; return 0-based top-k indices (B, k).
    If banned_* is provided, those (row, col) pairs are set to -inf so they won't appear in Top-K.
    """
    if item_embs_cpu.numel() == 0:
        return torch.empty((q_vecs.size(0), 0), dtype=torch.long)

    q = F.normalize(q_vecs, dim=-1).to(device, non_blocking=True)  # (B, D)
    B = q.shape[0]
    I = item_embs_cpu.shape[0]

    # Move banned indices to device if provided
    if banned_row_index is not None and banned_global_col_index is not None:
        banned_row_index = banned_row_index.to(device, non_blocking=True).long()
        banned_global_col_index = banned_global_col_index.to(device, non_blocking=True).long()
    else:
        banned_row_index = banned_global_col_index = None

    top_scores = torch.full((B, 0), float("-inf"), device=device)
    top_indices = torch.empty((B, 0), dtype=torch.long, device=device)

    for start in range(0, I, chunk_items):
        end = min(start + chunk_items, I)
        cand = item_embs_cpu[start:end].to(device, non_blocking=True)  # (m, D)
        scores = q @ cand.t()  # (B, m)

        # Batch mask the (row, col) pairs falling into the current chunk (no Python loops)
        if banned_row_index is not None:
            in_chunk = (banned_global_col_index >= start) & (banned_global_col_index < end)
            if in_chunk.any():
                ri = banned_row_index[in_chunk]
                ci = (banned_global_col_index[in_chunk] - start).long()
                neg_val = torch.finfo(scores.dtype).min
                scores[ri, ci] = neg_val  # advanced indexing (avoid index_put_ dtype issues)

        # Merge with running top-k
        if top_indices.numel() == 0:
            vals, idx = torch.topk(scores, k=min(k, scores.size(1)), dim=1)
            top_scores = vals
            top_indices = idx + start
        else:
            comb_scores = torch.cat([top_scores, scores], dim=1)  # (B, k+m)
            chunk_idx = torch.arange(start, end, device=device).view(1, -1).expand(B, -1)
            comb_indices = torch.cat([top_indices, chunk_idx], dim=1)  # (B, k+m)
            vals, new_pos = torch.topk(comb_scores, k=min(k, comb_scores.size(1)), dim=1)
            row_ids = torch.arange(B, device=device).unsqueeze(1)
            new_idx = comb_indices[row_ids, new_pos]
            top_scores, top_indices = vals, new_idx

    return top_indices.to('cpu')  # 0-based


def infer():
    """Main inference: encode candidates, compute user queries, top-k search, map to creative_ids."""
    args = get_args()
    data_path = os.environ.get('EVAL_DATA_PATH')

    # --- DataLoader ---
    test_dataset = MyTestDataset(data_path, args)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        collate_fn=test_dataset.collate_fn,
        persistent_workers=False
    )

    # --- Model ---
    usernum, itemnum = test_dataset.usernum, test_dataset.itemnum
    feat_statistics, feat_types = test_dataset.feat_statistics, test_dataset.feature_types
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
    model.eval()

    # Load checkpoint
    ckpt_path = get_ckpt_path()
    state = torch.load(ckpt_path, map_location=torch.device(args.device))
    model.load_state_dict(state)

    # Match training/eval behavior: zero-out padding indices in embeddings
    with torch.no_grad():
        emb = getattr(model, "item_emb", None)
        if emb is not None and isinstance(emb, nn.Embedding):
            emb.weight[0].zero_()
        if hasattr(model, 'pos_emb'):
            model.pos_emb.weight[0].zero_()
        if hasattr(model, 'user_emb'):
            model.user_emb.weight[0].zero_()
        if hasattr(model, 'sparse_emb'):
            for k in model.sparse_emb:
                model.sparse_emb[k].weight[0].zero_()

    # 1) Candidate encoding + aligned item_id
    cand_embs_cpu, creative_ids, cand_item_ids = build_candidate_matrix_from_predict_set(
        test_dataset.indexer['i'],
        test_dataset.feature_types,
        test_dataset.feature_default_value,
        test_dataset.mm_tables,
        model,
        args.device,
        batch_size=args.cand_encode_bs
    )
    
    C = cand_embs_cpu.size(0)

    # 2) Build id -> candidate column lookup on GPU (1-based ids; 0/missing -> -1)
    id2col = torch.full((itemnum + 1,), -1, dtype=torch.long, device=args.device)
    cand_item_ids_dev = cand_item_ids.to(args.device, non_blocking=True).long()
    cols = torch.arange(C, device=args.device, dtype=torch.long)
    mask = cand_item_ids_dev > 0
    id2col.scatter_(0, cand_item_ids_dev[mask], cols[mask])  # id2col[item_id] = col

    # 3) Batch inference + chunked Top-K (mask all items seen in the sequence)
    all_user_ids = []
    all_topk_creatives = []

    with torch.inference_mode():
        for _, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc="Infer & search"):
            seq, token_type, next_action_type, seq_time, next_time, seq_feat, user_id = batch

            seq = seq.to(args.device, non_blocking=True)  # (B, L)
            # User representation
            q = model.predict(seq, seq_feat, token_type, seq_time, next_action_type)  # (B, D)

            # Map tokens to global candidate column indices; -1 if not a candidate
            seq_cols = id2col.index_select(0, seq.view(-1)).view_as(seq)  # (B, L)
            valid = (seq > 0) & (seq_cols >= 0)                           # (B, L)

            B = seq.size(0)
            row_grid = torch.arange(B, device=args.device).unsqueeze(1).expand_as(seq)  # (B, L)

            # Flattened banned positions for masking during scoring
            banned_row_index = row_grid[valid].long()          # (T_total,)
            banned_global_col_index = seq_cols[valid].long()   # (T_total,)

            # Top-K over candidates with chunked matrix-mul and masking
            idx = batched_topk_indices(
                q_vecs=q,
                item_embs_cpu=cand_embs_cpu,
                device=args.device,
                k=args.topk,
                chunk_items=args.chunk_items,
                banned_row_index=banned_row_index,
                banned_global_col_index=banned_global_col_index
            )  # 0-based (B, k)

            # 4) Map indices to creative_id (small per-row loop for string mapping)
            for row in idx.tolist():
                all_topk_creatives.append([creative_ids[i] for i in row])

            all_user_ids.extend(user_id)

    return all_topk_creatives, all_user_ids

if __name__ == "__main__":
    infer()

