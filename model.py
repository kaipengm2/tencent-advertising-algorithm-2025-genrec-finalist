# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025, Zheng Kaipeng
#
# Generative Sequential Recommendation (HSTU-based) implemented in PyTorch.
# Features an optimized sampled softmax with log q correction, in-batch hard negative mining
# and curriculum learning for action-conditioned generation.
# Developed upon the baseline of the 2025 Tencent Advertising Algorithm Competition.
# References the official HSTU implementation (“Actions Speak Louder than Words”).


from __future__ import annotations

import math
import os
from collections import deque
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------
# Utils
# ------------------------------

def get_activation(name: str):
    """Return a torch.nn.functional activation by name."""
    name = name.lower()
    if name in ("silu", "swish"):
        return F.silu
    if name in ("gelu",):
        return F.gelu
    if name in ("relu",):
        return F.relu
    if name in ("identity", "none"):
        return lambda x: x
    raise ValueError(f"Unknown activation: {name}")


# ------------------------------
# Relative Attention Bias (time-only)
# ------------------------------

class RelativeAttentionBiasModule(nn.Module):
    """
    Bucketed relative time bias (and optional position buckets) with per-head or shared scalars.

    Output: additive bias of shape [B, H, Lq, Lk].
    """

    def __init__(
        self,
        num_heads: int,
        pos_num_buckets: int = 64,
        time_num_buckets: int = 64,
        max_pos_distance: int = 2048,
        max_time_span: int = 86400 * 90,
        share_across_heads: bool = False,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.pos_num_buckets = pos_num_buckets
        self.time_num_buckets = time_num_buckets
        self.max_pos_distance = max_pos_distance
        self.max_time_span = max_time_span
        self.share_across_heads = share_across_heads

        emb_dim = 1 if share_across_heads else num_heads
        # Position bias is kept for extension; not added in forward for now.
        self.pos_bias = nn.Embedding(pos_num_buckets, emb_dim)
        self.time_bias = nn.Embedding(time_num_buckets, emb_dim)

        nn.init.trunc_normal_(self.pos_bias.weight, std=init_std)
        nn.init.trunc_normal_(self.time_bias.weight, std=init_std)

        self.register_buffer(
            "_pos_boundaries",
            self._build_boundaries_linearlog(pos_num_buckets, max_pos_distance),
            persistent=False,
        )
        self.register_buffer(
            "_time_boundaries",
            self._build_boundaries(time_num_buckets, max_time_span),
            persistent=False,
        )

    @staticmethod
    def _build_boundaries_linearlog(num_buckets: int, max_val: int) -> torch.Tensor:
        """First half linear, second half logarithmic bucket boundaries."""
        max_exact = max(1, num_buckets // 2)
        linear = torch.arange(1, max_exact + 1, dtype=torch.float32)
        if num_buckets - 1 == max_exact:
            return linear
        steps = num_buckets - 1 - max_exact
        start = math.log(max_exact + 1.0)
        end = math.log(max(max_val, max_exact + 1))
        log_part = torch.exp(torch.linspace(start, end, steps=steps))
        b = torch.unique(torch.floor(torch.cat([linear, log_part]).clamp(min=1)).to(torch.int64))
        return b.to(torch.float32)

    @staticmethod
    def _build_boundaries(num_buckets: int, max_val: int) -> torch.Tensor:
        """Pure logarithmic bucket boundaries."""
        max_exact = 0
        steps = max(1, num_buckets - 1 - max_exact)
        start = math.log(max_exact + 1.0)
        end = math.log(max(max_val, max_exact + 1))
        log_part = torch.exp(torch.linspace(start, end, steps=steps))
        b = torch.unique(torch.floor(log_part.clamp(min=1)).to(torch.int64))
        return b.to(torch.float32)

    def _bucketize_time_pair(self, ts_q: torch.Tensor, ts_k: torch.Tensor) -> torch.Tensor:
        """
        ts_q: [B, Lq] query timestamps (seconds)
        ts_k: [B, Lk] key   timestamps (seconds)
        Return: [B, Lq, Lk] bucket indices
        """
        tdiff = (ts_q.unsqueeze(2) - ts_k.unsqueeze(1)).abs().clamp_min(1).to(torch.float32)
        return torch.bucketize(tdiff, self._time_boundaries.to(ts_q.device))

    def forward(
        self,
        L: int,  # kept for API compatibility (position buckets could use it)
        timestamps: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        device = device or self.time_bias.weight.device

        # Support single tensor or (ts_q, ts_k)
        if isinstance(timestamps, tuple):
            ts_q, ts_k = timestamps
        else:
            ts_q = ts_k = timestamps
        assert ts_q is not None and ts_k is not None, "timestamps are required for time bias."

        # Time bias only
        time_bk = self._bucketize_time_pair(ts_q, ts_k)  # [B, Lq, Lk]
        time_bias = self.time_bias(time_bk)              # [B, Lq, Lk, emb]
        if self.share_across_heads:
            time_bias = time_bias.squeeze(-1).unsqueeze(1)  # [B, 1, Lq, Lk]
        else:
            time_bias = time_bias.permute(0, 3, 1, 2)       # [B, H, Lq, Lk]
        return time_bias.to(device)


# ------------------------------
# Sequential Transduction Unit (HSTU)
# ------------------------------

class SequentialTransductionUnitJagged(nn.Module):
    """
    Single HSTU layer:
      1) One linear produces U, V, Q, K (layout 'uvqk')
      2) Pointwise attention: A = phi2(QK^T + RAB)  (no softmax)
      3) Aggregate A@V per head, merge heads
      4) (Optional norm) then elementwise gate with U; project to embedding_dim
    """

    def __init__(
        self,
        embedding_dim: int,
        linear_hidden_dim: int,
        attention_dim: int,
        num_heads: int,
        dropout_ratio: float = 0.0,
        attn_dropout_ratio: float = 0.0,
        linear_activation: str = "silu",
        relative_attention_bias_module: Optional[RelativeAttentionBiasModule] = None,
        normalization: str = "rmsnorm",  # "layernorm" | "rmsnorm" | "none" (rmsnorm falls back to Identity here)
        linear_config: str = "uvqk",
        concat_ua: bool = False,
        eps: float = 1e-6,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        assert linear_config == "uvqk", "This reference supports 'uvqk' only."
        assert attention_dim % num_heads == 0, "attention_dim must be divisible by num_heads"
        assert linear_hidden_dim % num_heads == 0, "linear_hidden_dim must be divisible by num_heads"

        self.embedding_dim = embedding_dim
        self.linear_hidden_dim = linear_hidden_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.d_qk = attention_dim // num_heads
        self.d_uv = linear_hidden_dim // num_heads
        self.dropout_ratio = dropout_ratio
        self.attn_dropout_ratio = attn_dropout_ratio
        self.act1 = get_activation(linear_activation)
        self.act2 = get_activation("silu")
        self.rab = relative_attention_bias_module
        self.concat_ua = concat_ua
        self.scale = 1.0 / math.sqrt(self.d_qk)

        # One shot linear -> [U, V, Q, K]
        out_dim = num_heads * (self.d_uv * 2 + self.d_qk * 2)
        self.uvqk = nn.Linear(embedding_dim, out_dim, bias=True)
        nn.init.trunc_normal_(self.uvqk.weight, std=init_std)
        nn.init.zeros_(self.uvqk.bias)

        # Normalization after pointwise pooling
        if normalization.lower() == "layernorm":
            self.norm_after = nn.LayerNorm(self.linear_hidden_dim, eps=eps)
        elif normalization.lower() == "rmsnorm":
            self.norm_after = nn.RMSNorm(self.linear_hidden_dim, eps=eps)
        else:
            self.norm_after = nn.Identity()

        out_in_dim = linear_hidden_dim * (3 if concat_ua else 1)
        self.o = nn.Linear(out_in_dim, embedding_dim, bias=True)
        nn.init.trunc_normal_(self.o.weight, std=init_std)
        nn.init.zeros_(self.o.bias)

        self.dropout = nn.Dropout(dropout_ratio)
        self.attn_drop = nn.Dropout(attn_dropout_ratio)

    def _split_heads(self, t: torch.Tensor, head_dim: int) -> torch.Tensor:
        # [B, L, H*D] -> [B, H, L, D]
        B, L, _ = t.shape
        return t.view(B, L, self.num_heads, head_dim).permute(0, 2, 1, 3).contiguous()

    def forward(
        self,
        x: torch.Tensor,                     # [B, L, D]
        timestamps: Optional[torch.Tensor],  # [B, L] or tuple([B,L],[B,L])
        attn_mask: Optional[torch.Tensor],   # [B, L, L] bool, True=visible (column mask)
        x_offsets: Optional[torch.Tensor] = None,  # kept for API compatibility
    ) -> torch.Tensor:
        B, L, _ = x.shape
        h = self.act1(self.uvqk(x))
        u, v, q, k = torch.split(
            h,
            [self.num_heads * self.d_uv,
             self.num_heads * self.d_uv,
             self.num_heads * self.d_qk,
             self.num_heads * self.d_qk],
            dim=-1,
        )
        u = self._split_heads(u, self.d_uv)
        v = self._split_heads(v, self.d_uv)
        q = self._split_heads(q, self.d_qk) * self.scale
        k = self._split_heads(k, self.d_qk)

        # attention logits + relative bias (time)
        attn_logits = torch.einsum("bhid,bhjd->bhij", q.to(torch.float32), k.to(torch.float32))
        if self.rab is not None:
            rab = self.rab(L, timestamps, device=x.device)  # [B,H,L,L]
            if rab.size(0) == 1:
                rab = rab.expand(attn_logits.size(0), -1, -1, -1)
            attn_logits = attn_logits + rab

        # Pointwise nonlinearity (no softmax)
        A_full = self.act2(attn_logits)
        if attn_mask is not None:
            A = A_full * attn_mask.unsqueeze(1).to(A_full.dtype)  # mask columns only
        else:
            A = A_full
        A = self.attn_drop(A)

        # Aggregate A@V
        y_h = torch.einsum("bhij,bhjd->bhid", A, v)
        y = y_h.permute(0, 2, 1, 3).contiguous().view(B, L, self.linear_hidden_dim)

        # Gate with U, optional concat
        u_merge = u.permute(0, 2, 1, 3).contiguous().view(B, L, self.linear_hidden_dim)
        y_point = self.norm_after(y).type_as(x) * u_merge
        y_out = torch.cat([y, y_point, y - y_point], dim=-1) if self.concat_ua else y_point

        y_out = self.o(y_out)
        y_out = self.dropout(y_out)
        return y_out


class HSTULayerCompat(nn.Module):
    """Thin wrapper to keep a Transformer-like API for stacking layers."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rab: RelativeAttentionBiasModule,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        normalization: str = "layernorm",
    ) -> None:
        super().__init__()
        self.stu = SequentialTransductionUnitJagged(
            embedding_dim=d_model,
            linear_hidden_dim=d_model,
            attention_dim=d_model,
            num_heads=n_heads,
            dropout_ratio=dropout,
            attn_dropout_ratio=attn_dropout,
            linear_activation="silu",
            relative_attention_bias_module=rab,
            normalization=normalization,
            linear_config="uvqk",
            concat_ua=False,
        )

    def forward(self, x, attn_mask=None, seq_time=None):
        return self.stu(x, timestamps=seq_time, attn_mask=attn_mask)


# ------------------------------
# Importance weighting from item frequency
# ------------------------------

def make_log_q_table_from_df(
    df: pd.DataFrame,
    item_id_col: str = "item_id",
    count_col: str = "occ_total",
    *,
    vocab_size: Optional[int] = None,
    power: float = 1.0,
    eps: float = 1e-8,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Union[torch.Tensor, Dict[int, float]]:
    """
    Return a Tensor of shape (vocab_size,) with logQ[i] = log P(i), if vocab_size is given.
    Otherwise, return a dict {item_id -> logQ}.
    """
    ids = df[item_id_col].to_numpy()
    cnts = df[count_col].to_numpy().astype(np.float64)
    cnts = np.maximum(cnts, eps) ** float(power)

    Z = cnts.sum()
    logq = np.log(cnts) - np.log(Z)

    if vocab_size is None:
        return {int(i): float(l) for i, l in zip(ids, logq)}

    table = torch.full((vocab_size,), float("-inf"), dtype=dtype, device=device)
    table[torch.as_tensor(ids, dtype=torch.long, device=device)] = torch.as_tensor(logq, dtype=dtype, device=device)

    # Optional tiny mass for missing ids
    miss = (table == float("-inf"))
    if miss.any():
        tiny = float(np.log(1.0 / (vocab_size * 1e3)))  # ~1e-9 / V
        table[miss] = tiny
    return table


# ------------------------------
# Calibrated Dual-Negative Sampled Softmax
# ------------------------------

def CDN_Softmax(
    Q: torch.Tensor,                    # (B, N, C) queries
    Pos: torch.Tensor,                  # (B, N, C) positives aligned with Q
    pos_item_ids: torch.LongTensor,     # (B, N) item ids for positives

    *,
    neg_global: Optional[torch.Tensor] = None,              # (B, N, C) global negatives (same shape as Pos)
    neg_global_item_ids: Optional[torch.LongTensor] = None, # (B, N) ids for global negatives
    log_q_table: torch.Tensor,                               # (V,) log probability table for importance correction
    sim_threshold: float = 0.99,                             # drop negatives too similar to positives (by cosine)
    temperature: float = 0.07,                               # temperature
    normalize: bool = True,                                  # L2-normalize embeddings before similarity
    use_inbatch_neg: bool = True,                            # treat other batch positives as negatives
    apply_log_q_to_pos: bool = True,                         # subtract logQ for positives too (sampled-softmax style)
    return_pos_logits: bool = True,                          # return positive logits along with negatives
) -> Union[
    Tuple[torch.Tensor, list[torch.Tensor]],
    list[torch.Tensor]
]:
    """Prepare logits for contrastive training (no reduction here).

    Shapes
    ------
    Q, Pos: (B, N, C)  — per-token embeddings in a sequence (N can be sequence length)
    pos_item_ids: (B, N)
    neg_global, neg_global_item_ids: optional (B, N, C) and (B, N)
    """
    assert Q.shape == Pos.shape and Q.dim() == 3
    B, N, C = Q.shape
    BN = B * N
    device = Q.device
    dtype = Q.dtype

    def _gather_logq(ids_2d: torch.LongTensor) -> torch.Tensor:
        return log_q_table.to(device=device, dtype=dtype)[ids_2d]

    if normalize:
        Q = F.normalize(Q, dim=-1, eps=1e-8)
        Pos = F.normalize(Pos, dim=-1, eps=1e-8)
        if neg_global is not None:
            neg_global = F.normalize(neg_global, dim=-1, eps=1e-8)

    Qf = Q.reshape(BN, C)
    Pf = Pos.reshape(BN, C)
    pos_ids_flat = pos_item_ids.reshape(BN)

    all_neg_logits: list[torch.Tensor] = []

    # A) Global negatives
    if (neg_global is not None) and (neg_global_item_ids is not None):
        Nf = neg_global.reshape(BN, C)
        neg_ids_flat = neg_global_item_ids.reshape(BN)

        # Drop overly similar negatives w.r.t positives (per-seq)
        pos2neg_cos = Pf @ Nf.t()                           # (BN, BN)
        pos2neg_cos_seq = pos2neg_cos.view(B, N, BN)        # (B, N, BN)
        max_cos_per_seq = pos2neg_cos_seq.max(dim=1).values # (B, BN)
        neg_mask_seq_glob = max_cos_per_seq <= sim_threshold
        row_seq_idx = torch.arange(BN, device=device) // N
        neg_col_ok_glob = neg_mask_seq_glob[row_seq_idx]    # (BN, BN)

        sim_neg_glob = (Qf @ Nf.t()) / temperature          # (BN, BN)
        neg_logq_cols = _gather_logq(neg_ids_flat)          # (BN,)
        logits_neg_glob = sim_neg_glob - neg_logq_cols.unsqueeze(0)
        logits_neg_glob = logits_neg_glob.masked_fill(~neg_col_ok_glob, float("-inf"))
        all_neg_logits.append(logits_neg_glob)

    # B) In-batch negatives
    if use_inbatch_neg:
        pos2pos_cos = Pf @ Pf.t()                           # (BN, BN)
        pos2pos_cos_seq = pos2pos_cos.view(B, N, BN)        # (B, N, BN)
        max_cos_per_seq = pos2pos_cos_seq.max(dim=1).values # (B, BN)
        neg_mask_seq_ibn = max_cos_per_seq <= sim_threshold

        row_seq_idx = torch.arange(BN, device=device) // N
        col_seq_idx = torch.arange(BN, device=device) // N
        ibn_seq_diff = (row_seq_idx[:, None] != col_seq_idx[None, :])  # don't use same sequence as negative
        neg_col_ok_ibn = neg_mask_seq_ibn[row_seq_idx] & ibn_seq_diff

        sim_neg_ibn = (Qf @ Pf.t()) / temperature           # (BN, BN)
        ibn_logq_cols = _gather_logq(pos_ids_flat)          # (BN,)
        logits_neg_ibn = sim_neg_ibn - ibn_logq_cols.unsqueeze(0)
        logits_neg_ibn = logits_neg_ibn.masked_fill(~neg_col_ok_ibn, float("-inf"))
        all_neg_logits.append(logits_neg_ibn)

    # Positive logits
    if return_pos_logits:
        pos_logits = ((Q * Pos).sum(dim=-1) / temperature).reshape(BN, 1)
        if apply_log_q_to_pos:
            pos_logits = pos_logits - _gather_logq(pos_ids_flat).unsqueeze(1)
        return pos_logits, all_neg_logits

    return all_neg_logits


# ------------------------------
# Baseline model tying everything together
# ------------------------------

class BaselineModel(nn.Module):
    def __init__(self, user_num, item_num, feat_statistics, feat_types, args) -> None:
        super().__init__()
        # ---- Configuration ----
        self.use_item_id_emb = getattr(args, "use_item_id_emb", True)
        self.repr_dim = 512
        self.useremb_dim = 32
        self.userfeat_dim = 128
        self.user_num = user_num
        self.item_num = item_num
        self.dev = torch.device(getattr(args, "device", "cuda"))
        self.norm_first = getattr(args, "norm_first", False)
        self.maxlen = getattr(args, "maxlen", 100)
        self.array_token_dropout = getattr(args, "array_token_dropout", 0.0)

        # ---- Feature schemas ----
        self._init_feat_info(feat_statistics, feat_types)

        # ---- Embeddings ----
        if self.use_item_id_emb:
            self.item_emb = nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        else:
            self.item_emb = None

        self.user_emb = nn.Embedding(self.user_num + 1, self.useremb_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(2 * self.maxlen + 1, self.repr_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(p=getattr(args, "emb_dropout", 0.0))
        self.emb_dropout_u = nn.Dropout(p=getattr(args, "emb_dropout", 0.0))
        self.user_post_dropout = nn.Dropout(p=getattr(args, "emb_dropout", 0.0))

        self.sparse_emb = nn.ModuleDict()
        self.emb_transform = nn.ModuleDict()
        self.emb_norm = nn.ModuleDict()

        self.num_actions = getattr(args, "num_actions", 3)  # e.g., 0=expose, 1=click, 2=conversion
        self.action_emb = nn.Embedding(self.num_actions, self.repr_dim)
        nn.init.normal_(self.action_emb.weight, mean=0.0, std=0.02)

        self.time_feat_proj = nn.Linear(4, self.repr_dim, bias=False)
        nn.init.normal_(self.time_feat_proj.weight, mean=0.0, std=0.02)

        self.attention_layers = nn.ModuleList()

        # ---- Importance weights (logQ) ----
        self.log_q_table = self._load_or_build_logq_table()

        self.neg_pool = deque(maxlen=1)  # reserved for future hard-negative cache

        # ---- User & Item towers ----
        userdim = (
            self.userfeat_dim * (len(self.USER_SPARSE_FEAT) + len(self.USER_ARRAY_FEAT))
            + len(self.USER_CONTINUAL_FEAT)
            + self.useremb_dim
        )

        # Per-feature dims for items (sparse dims may vary by field id)
        def _item_sparse_dim_for(k: str) -> int:
            # Keep the original special-casing but compute the sum exactly
            if k in ["100", "101", "112", "114", "116"]:
                return args.hidden_units // 2
            else:
                return args.hidden_units

        item_sparse_dim = sum(_item_sparse_dim_for(k) for k in self.ITEM_SPARSE_FEAT)
        item_array_dim = len(self.ITEM_ARRAY_FEAT) * args.hidden_units
        item_cont_dim = len(self.ITEM_CONTINUAL_FEAT)  # scalar features
        item_id_dim = args.hidden_units if self.use_item_id_emb else 0
        item_mm_dim = 128 * len(self.ITEM_EMB_FEAT)   # each projected to 128

        itemdim_total = item_sparse_dim + item_array_dim + item_cont_dim + item_id_dim + item_mm_dim

        self.userdnn = nn.Linear(userdim, self.repr_dim)
        self.itemdnn = nn.Sequential(
            nn.Linear(itemdim_total, self.repr_dim * 2),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.repr_dim * 2, self.repr_dim),
        )

        self.cond = nn.Linear(self.repr_dim, 2 * self.repr_dim)  # FiLM params for user conditioning

        pos_num_buckets = getattr(args, "pos_num_buckets", 64)
        time_num_buckets = getattr(args, "time_num_buckets", 100)
        max_pos_distance = getattr(args, "max_pos_distance", self.maxlen)
        max_time_span = getattr(args, "max_time_span", 86400 * 30)

        # Shared RAB across layers
        self._hstu_rab = RelativeAttentionBiasModule(
            num_heads=args.num_heads,
            pos_num_buckets=pos_num_buckets,
            time_num_buckets=time_num_buckets,
            max_pos_distance=max_pos_distance,
            max_time_span=max_time_span,
            share_across_heads=True,
        )

        for _ in range(args.num_blocks):
            self.attention_layers.append(
                HSTULayerCompat(
                    d_model=self.repr_dim,
                    n_heads=args.num_heads,
                    rab=self._hstu_rab,
                    dropout=getattr(args, "attn_out_dropout", 0.0),
                    attn_dropout=getattr(args, "sdpa_dropout", 0.0),
                    normalization="layernorm",
                )
            )

        # Embedding tables: user
        for k in self.USER_SPARSE_FEAT:
            self.sparse_emb[k] = nn.Embedding(self.USER_SPARSE_FEAT[k] + 1, self.userfeat_dim, padding_idx=0)
        for k in self.USER_ARRAY_FEAT:
            self.sparse_emb[k] = nn.EmbeddingBag(
                self.USER_ARRAY_FEAT[k] + 1, self.userfeat_dim, mode="mean", include_last_offset=True
            )

        # Embedding tables: item (sparse/array)
        for k in self.ITEM_SPARSE_FEAT:
            dim = _item_sparse_dim_for(k)
            self.sparse_emb[k] = nn.Embedding(self.ITEM_SPARSE_FEAT[k] + 1, dim, padding_idx=0)

        for k in self.ITEM_ARRAY_FEAT:
            self.sparse_emb[k] = nn.Embedding(self.ITEM_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)

        # Multimodal embedding projections
        for k in self.ITEM_EMB_FEAT:
            self.emb_transform[k] = nn.Linear(self.ITEM_EMB_FEAT[k], 128)
            self.emb_norm[k] = nn.LayerNorm(128)

        # Zero padding rows
        with torch.no_grad():
            if hasattr(self, "pos_emb"):
                self.pos_emb.weight[0].zero_()
            if self.item_emb is not None:
                self.item_emb.weight[0].zero_()
            if hasattr(self, "user_emb"):
                self.user_emb.weight[0].zero_()
            for k in self.sparse_emb:
                if isinstance(self.sparse_emb[k], nn.Embedding):
                    self.sparse_emb[k].weight[0].zero_()

    # ---- Feature schema helpers ----
    def _init_feat_info(self, feat_statistics, feat_types) -> None:
        self.USER_SPARSE_FEAT = {k: int(feat_statistics[k]) for k in feat_types["user_sparse"]}
        self.USER_CONTINUAL_FEAT = list(feat_types["user_continual"])
        self.ITEM_SPARSE_FEAT = {k: int(feat_statistics[k]) for k in feat_types["item_sparse"]}
        self.ITEM_CONTINUAL_FEAT = list(feat_types["item_continual"])
        self.USER_ARRAY_FEAT = {k: int(feat_statistics[k]) for k in feat_types["user_array"]}
        self.ITEM_ARRAY_FEAT = {k: int(feat_statistics[k]) for k in feat_types["item_array"]}
        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in feat_types["item_emb"]}

    def _load_or_build_logq_table(self) -> torch.Tensor:
        """Load item frequency CSV"""
        vocab_size = self.item_num + 1
        cache_root = os.environ.get("USER_CACHE_PATH")
        if not cache_root:
            raise EnvironmentError("USER_CACHE_PATH is not set")
        csv_path = os.path.join(cache_root, "item_freq.csv")
        df = pd.read_csv(csv_path).dropna()
        return make_log_q_table_from_df(
                    df,
                    item_id_col="item_id",
                    count_col="occ_total",
                    vocab_size=vocab_size,
                    power=0.75,
                    device=self.dev,
                    dtype=torch.float32,
                )

    def _apply_item_id_dropout(self, ids: torch.Tensor, p: float) -> torch.Tensor:
        if (not self.training) or p <= 0:
            return ids
        drop = (torch.rand_like(ids.float()) < p)
        return ids.masked_fill(drop, 0)  # padding_idx=0

    # ---- Feature tensorization ----
    def feat2tensor(self, seq_feature, k: str) -> torch.Tensor:
        """Convert list-of-dict sequence features into padded tensors (legacy path)."""
        batch_size = len(seq_feature)
        if k in self.ITEM_ARRAY_FEAT or k in self.USER_ARRAY_FEAT:
            max_array_len = 0
            max_seq_len = 0
            for i in range(batch_size):
                seq_data = [item.get(k, []) for item in seq_feature[i]]
                max_seq_len = max(max_seq_len, len(seq_data))
                for item_data in seq_data:
                    max_array_len = max(max_array_len, len(item_data))
            max_array_len = max(1, max_array_len)
            max_seq_len = max(1, max_seq_len)
            batch_data = np.zeros((batch_size, max_seq_len, max_array_len), dtype=np.int64)
            for i in range(batch_size):
                seq_data = [item.get(k, []) for item in seq_feature[i]]
                for j, item_data in enumerate(seq_data):
                    actual_len = min(len(item_data), max_array_len)
                    if actual_len > 0:
                        batch_data[i, j, :actual_len] = np.asarray(item_data[:actual_len], dtype=np.int64)
            return torch.from_numpy(batch_data).to(self.dev)
        else:
            max_seq_len = max(len(seq_feature[i]) for i in range(batch_size))
            max_seq_len = max(1, max_seq_len)
            batch_data = np.zeros((batch_size, max_seq_len), dtype=np.int64)
            for i in range(batch_size):
                seq_data = [item.get(k, 0) for item in seq_feature[i]]
                L = min(len(seq_data), max_seq_len)
                if L > 0:
                    batch_data[i, :L] = np.asarray(seq_data[:L], dtype=np.int64)
            return torch.from_numpy(batch_data).to(self.dev)

    def _make_time_feats(self, ts: torch.Tensor) -> torch.Tensor:
        """
        ts: [B, L] in seconds (local timezone preferred).
        Return: [B, L, 4] -> hour_sin, hour_cos, wday_sin, wday_cos
        """
        TWO_PI = 2.0 * math.pi
        ts = ts.to(self.dev)
        hour_f = (torch.remainder(ts, 86400).to(torch.float32) / 3600.0)
        ang_h = hour_f * (TWO_PI / 24.0)
        h_sin, h_cos = torch.sin(ang_h), torch.cos(ang_h)
        day_idx = torch.div(ts, 86400, rounding_mode="floor")
        wday = torch.remainder(day_idx + 4, 7).to(torch.float32)
        ang_w = wday * (TWO_PI / 7.0)
        w_sin, w_cos = torch.sin(ang_w), torch.cos(ang_w)
        return torch.stack([h_sin, h_cos, w_sin, w_cos], dim=-1)

    # ---- Feature -> embeddings ----
    def feat2emb(
        self,
        seq: torch.Tensor,
        feature_array,
        mask: Optional[torch.Tensor] = None,
        include_user: bool = False
    ):
        """
        `feature_array` can be a dict of pre-packed tensors or list-of-dict (legacy).
        When dict, tensors are expected on CPU; moved to device with non_blocking=True.
        """
        seq = seq.to(self.dev, non_blocking=True)
        feature_is_packed = isinstance(feature_array, dict)

        # item/user id embeddings
        item_feat_list: list[torch.Tensor] = []
        if include_user:
            assert mask is not None, "mask is required when include_user=True"
            user_mask = (mask == 2).to(self.dev, non_blocking=True)
            item_mask = (mask == 1).to(self.dev, non_blocking=True)
            user_embedding = self.user_emb(user_mask * seq)
            user_feat_list = [user_embedding]
            if self.use_item_id_emb:
                item_embedding = self.item_emb(item_mask * seq)
                item_feat_list.append(item_embedding)
        else:
            if self.use_item_id_emb:
                seq_for_item = self._apply_item_id_dropout(seq, p=0.0)
                item_embedding = self.item_emb(seq_for_item)
                item_feat_list.append(item_embedding)

        # Batch per feature type
        all_feat_types: list[Tuple[Dict[str, int], str, list[torch.Tensor]]] = [
            (self.ITEM_SPARSE_FEAT, "item_sparse", item_feat_list),
            (self.ITEM_ARRAY_FEAT, "item_array", item_feat_list),
            (self.ITEM_CONTINUAL_FEAT, "item_continual", item_feat_list),
        ]
        if include_user:
            all_feat_types.extend([
                (self.USER_SPARSE_FEAT, "user_sparse", user_feat_list),
                (self.USER_ARRAY_FEAT, "user_array", user_feat_list),
                (self.USER_CONTINUAL_FEAT, "user_continual", user_feat_list),
            ])

        for feat_dict, feat_type, feat_list in all_feat_types:
            if not feat_dict:
                continue
            for k in feat_dict:
                if feature_is_packed:
                    tensor_feature = feature_array[k].to(self.dev, non_blocking=True)
                else:
                    tensor_feature = self.feat2tensor(feature_array, k)

                if feat_type.endswith("sparse"):
                    emb_k = self.sparse_emb[k](tensor_feature)
                    feat_list.append(emb_k)

                elif feat_type.endswith("array"):
                    tf = tensor_feature  # [B, L, A], int64, 0 = PAD
                    if self.training and self.array_token_dropout > 0:
                        keep = torch.rand_like(tf.float()) > self.array_token_dropout
                        tf = tf * keep.long()
                    B, L, A = tf.shape
                    tf_flat = tf.reshape(-1, A)
                    maskA = (tf_flat != 0)
                    lengths = maskA.sum(dim=1)
                    idx_flat = tf_flat[maskA]
                    offsets = torch.zeros(B * L + 1, device=self.dev, dtype=torch.long)
                    if lengths.numel() > 0:
                        offsets[1:] = torch.cumsum(lengths, dim=0)
                    bag_out = self.sparse_emb[k](idx_flat, offsets)  # [sum(len), H] -> pooled [B*L, H]
                    pooled = bag_out.view(B, L, -1)
                    feat_list.append(pooled)

                elif feat_type.endswith("continual"):
                    feat_list.append(tensor_feature.unsqueeze(2).to(self.dev, non_blocking=True))

        # multimodal item embeddings
        for k in self.ITEM_EMB_FEAT:
            if feature_is_packed:
                tensor_feature = feature_array[k].to(self.dev, non_blocking=True).float()
            else:
                batch_size = len(feature_array)
                emb_dim = self.ITEM_EMB_FEAT[k]
                seq_len = len(feature_array[0]) if batch_size > 0 else 1
                batch_emb_data = np.zeros((batch_size, seq_len, emb_dim), dtype=np.float32)
                for i, seq_feats in enumerate(feature_array):
                    for j, item in enumerate(seq_feats):
                        if k in item:
                            batch_emb_data[i, j] = np.asarray(item[k], dtype=np.float32)
                tensor_feature = torch.from_numpy(batch_emb_data).to(self.dev)
            mod = self.emb_norm[k](self.emb_transform[k](tensor_feature))  # [B, L, 128]
            item_feat_list.append(mod)

        all_item_emb = torch.cat(item_feat_list, dim=2) if len(item_feat_list) > 1 else item_feat_list[0]
        all_item_emb = self.emb_dropout(all_item_emb)
        all_item_emb = self.itemdnn(all_item_emb)

        if include_user:
            all_user_emb = torch.cat(user_feat_list, dim=2) if len(user_feat_list) > 1 else user_feat_list[0]
            all_user_emb = self.emb_dropout_u(all_user_emb)
            all_user_emb = F.relu(self.userdnn(all_user_emb))
            all_user_emb = self.user_post_dropout(all_user_emb)

            user_mask = (mask == 2).to(self.dev, non_blocking=True)
            denom = user_mask.sum(dim=1, keepdim=True).clamp_min(1)
            u = (all_user_emb * user_mask.unsqueeze(-1)).sum(dim=1) / denom  # [B, D]

            gamma, beta = self.cond(u).chunk(2, dim=-1)
            gamma = torch.tanh(gamma)
            seqs_emb = all_item_emb * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        else:
            seqs_emb = all_item_emb
        return seqs_emb

    # ---- Build sequence & run HSTU ----
    def log2feats(self, log_seqs, mask, next_action_type, seq_feature, seq_time):
        seq_time = seq_time.to(self.dev, non_blocking=True)
        batch_size, maxlen = log_seqs.shape
        seqs = self.feat2emb(log_seqs, seq_feature, mask=mask, include_user=True)

        item_pos_mask = (mask != 0).unsqueeze(-1).to(self.dev, non_blocking=True)
        time_feats = self._make_time_feats(seq_time)
        seqs = seqs + self.time_feat_proj(time_feats) * item_pos_mask

        act_hist = next_action_type.to(self.dev)
        act_vec = self.action_emb(act_hist)
        seqs = seqs + act_vec * item_pos_mask

        poss = torch.arange(1, maxlen + 1, device=self.dev).unsqueeze(0).expand(batch_size, -1).clone()
        nonpad = (log_seqs != 0)
        poss = poss * nonpad
        user_col = (mask == 2).to(self.dev, non_blocking=True)
        poss = poss.masked_fill(user_col, 0)
        seqs = seqs + self.pos_emb(poss)

        L = seqs.shape[1]
        ones_matrix = torch.ones((L, L), dtype=torch.bool, device=self.dev)
        attention_mask_tril = torch.tril(ones_matrix)                 # causal
        attention_mask_pad = (mask != 0).to(self.dev)                 # visible columns
        attention_mask = attention_mask_tril.unsqueeze(0) & attention_mask_pad.unsqueeze(1)

        ts_pair = (seq_time, seq_time)
        for layer in self.attention_layers:
            mha_outputs = layer(seqs, attn_mask=attention_mask, seq_time=ts_pair)
            seqs = seqs + mha_outputs
        return seqs

    # ---- Training forward ----
    def forward(
        self,
        user_item,
        pos_seqs,
        neg_seqs,
        mask,
        next_mask,
        next_action_type,
        seq_time,
        seq_feature,
        pos_feature,
        neg_feature,
        hard_topk: int = 0,
        glb_neg_ratio: float = 1.0,
        use_inbatch_neg: bool = True,
    ):
        log_feats = self.log2feats(user_item, mask, next_action_type, seq_feature, seq_time)

        loss_mask = (next_mask == 1).to(self.dev)

        pos_embs = self.feat2emb(pos_seqs, pos_feature, include_user=False)
        neg_embs = self.feat2emb(neg_seqs, neg_feature, include_user=False)

        pos_logits, neg_list_main = CDN_Softmax(
            log_feats,
            pos_embs,
            pos_seqs,
            neg_global=neg_embs,
            neg_global_item_ids=neg_seqs,
            log_q_table=self.log_q_table,
            use_inbatch_neg=use_inbatch_neg,
            temperature=0.01,
        )

        neg_ibn = None
        neg_global_list: list[torch.Tensor] = []
        if use_inbatch_neg:
            neg_ibn = neg_list_main[-1]
            neg_global_list.extend(neg_list_main[:-1])
        else:
            neg_global_list.extend(neg_list_main)

        neg_global_cat = torch.cat(neg_global_list, dim=1) if len(neg_global_list) > 0 else None

        def _select_topk(mat: Optional[torch.Tensor], k: int) -> Optional[torch.Tensor]:
            if mat is None or mat.numel() == 0 or k <= 0:
                return None
            k = min(k, mat.size(1))
            return torch.topk(mat, k=k, dim=1).values

        if hard_topk and hard_topk > 0:
            gk = int(hard_topk * glb_neg_ratio)
            ik = hard_topk - gk
            gcap = neg_global_cat.size(1) if neg_global_cat is not None else 0
            icap = neg_ibn.size(1) if neg_ibn is not None else 0
            if gcap < gk:
                ik += (gk - gcap)
                gk = gcap
            if icap < ik:
                gk += (ik - icap)
                ik = icap
            top_g = _select_topk(neg_global_cat, gk)
            top_i = _select_topk(neg_ibn, ik)
            parts = [p for p in (top_g, top_i) if p is not None]
            topk_vals = torch.cat(parts, dim=1) if len(parts) > 0 else torch.empty(pos_logits.size(0), 0, device=pos_logits.device, dtype=pos_logits.dtype)
            denom = torch.logsumexp(torch.cat([pos_logits, topk_vals], dim=1), dim=1)
        else:
            parts = []
            if neg_global_cat is not None:
                parts.append(neg_global_cat)
            if neg_ibn is not None:
                parts.append(neg_ibn)
            neg_cat = torch.cat(parts, dim=1) if len(parts) > 0 else torch.empty(pos_logits.size(0), 0, device=pos_logits.device, dtype=pos_logits.dtype)
            denom = torch.logsumexp(torch.cat([pos_logits, neg_cat], dim=1), dim=1)

        num = torch.logsumexp(pos_logits, dim=1)
        loss_flat = denom - num

        B, L, _ = log_feats.shape
        BN = B * L
        m_flat = loss_mask.reshape(BN)
        a_flat = (next_action_type != 0).to(self.dev).reshape(BN).float()

        w_click = 1.0
        w_expo = 1.0
        delta = w_click - w_expo
        weights_full = w_expo + a_flat * delta

        idx = m_flat
        num = (loss_flat[idx] * weights_full[idx]).sum()
        den = weights_full[idx].sum().clamp_min(1e-8)
        loss = num / den
        return loss

    # ---- Inference ----
    def predict(self, log_seqs, seq_feature, mask, seq_time, next_action_type):
        log_feats = self.log2feats(log_seqs, mask, next_action_type, seq_feature, seq_time)
        final_feat = log_feats[:, -1, :]
        q = F.normalize(final_feat, dim=-1, eps=1e-8)
        return q
