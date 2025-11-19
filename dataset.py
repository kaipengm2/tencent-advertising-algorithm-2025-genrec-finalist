"""
PyTorch datasets for sequential recommendation with dense multimodal embeddings.

Train/Valid — MyDataset
- Reads: seq.jsonl via seq_offsets.pkl; indexer.pkl; item_feat_dict.json
- Negatives: optional $USER_CACHE_PATH/item_last_ts.json (cutoff 2025-05-29)
- Embeddings (dense): $USER_CACHE_PATH/emb_table_{fid}_{D}.mmap (built from data_dir/creative_emb if missing)
- Collate: packs sparse/array/continuous features and gathers dense memmap embeddings

Test — MyTestDataset
- Reads: predict_seq.jsonl via predict_seq_offsets.pkl; user_action_type.json to set each user’s next-action condition
- Cold-start: unseen strings → 0; embeddings default to zeros (no memmap gather)

Utils
- build_mm_memmaps/open_mm_memmaps: manage dense float32 memmaps (itemnum+1, D), row 0 = zeros

Env: USER_CACHE_PATH required
Deps: numpy, torch, tqdm
"""


from __future__ import annotations

import io
import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
import warnings


# =========================
# Multimodal feature dimensions
# =========================
SHAPE_DICT: Dict[str, int] = {
    "81": 32,
    "82": 1024,
    "83": 3584,
    "84": 4096,
    "85": 3584,
    "86": 3584,
}


# ===============================================
# Datasets (train/valid)
# ===============================================

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, args, latest: int = -1, memmap_dirname: str = "mm_memmap") -> None:
        """
        Args (in `args`):
          - maxlen: int
          - mm_emb_id: List[str] (e.g., ['81', '82', ...])
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self._load_data_and_offsets()  # offsets only

        self.maxlen = int(args.maxlen)
        self.mm_emb_ids: List[str] = list(args.mm_emb_id)
        self.latest = int(latest)

        # Lazy-open data file (per worker process)
        self._data_file: Optional[io.BufferedReader] = None
        self._data_file_pid: Optional[int] = None

        # Indexer (need itemnum/usernum & cid->iid)
        with open(self.data_dir / "indexer.pkl", "rb") as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer["i"])
            self.usernum = len(indexer["u"])
        self.indexer: Dict[str, Any] = indexer
        self.indexer_i_rev = {v: k for k, v in indexer["i"].items()}
        self.indexer_u_rev = {v: k for k, v in indexer["u"].items()}

        # Item feature dict (non-embedding)
        with open(self.data_dir / "item_feat_dict.json", "r", encoding="utf-8") as f:
            self.item_feat_dict: Dict[str, Dict[str, Any]] = json.load(f)

        # Open/build memmap tables
        cache_dir = os.environ.get("USER_CACHE_PATH")
        if not cache_dir:
            raise EnvironmentError("USER_CACHE_PATH is not set")
        self.mm_memmap_dir = Path(cache_dir)
        missing = [
            fid
            for fid in self.mm_emb_ids
            if not (self.mm_memmap_dir / f"emb_table_{fid}_{SHAPE_DICT[fid]}.mmap").exists()
        ]
        if missing:
            warnings.warn(
                f"memmap files missing for {missing}; building from creative_emb",
                RuntimeWarning,
            )
            build_mm_memmaps(
                mm_path=self.data_dir / "creative_emb",
                feat_ids=self.mm_emb_ids,
                indexer=self.indexer,
                itemnum=self.itemnum,
                out_dir=self.mm_memmap_dir,
            )
        self.mm_tables: Dict[str, np.memmap] = open_mm_memmaps(
            out_dir=self.mm_memmap_dir, feat_ids=self.mm_emb_ids, itemnum=self.itemnum
        )

        # Feature schemas & defaults
        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()

        # Global negative sampling pool
        self.valid_items = self._build_valid_items()

    # ---------- file/process handles ----------
    def _load_data_and_offsets(self) -> None:
        with open(self.data_dir / "seq_offsets.pkl", "rb") as f:
            self.seq_offsets: List[int] = pickle.load(f)

    def _ensure_data_file(self) -> None:
        pid = os.getpid()
        if (self._data_file is None) or (self._data_file_pid != pid):
            if self._data_file is not None:
                try:
                    self._data_file.close()
                except Exception:
                    pass
            self._data_file = open(self.data_dir / "seq.jsonl", "rb", buffering=1024 * 1024)
            self._data_file_pid = pid

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_data_file"] = None
        state["_data_file_pid"] = None
        return state

    def __del__(self) -> None:
        try:
            if self._data_file is not None:
                self._data_file.close()
        except Exception:
            pass

    # ---------- data loading / sampling ----------
    def _load_user_data(self, uid: int):
        """Load a single user's sequence from file."""
        self._ensure_data_file()
        self._data_file.seek(self.seq_offsets[uid], os.SEEK_SET)
        line = self._data_file.readline()
        data = json.loads(line.decode("utf-8"))
        return data

    def _random_neq(self, seen_set: set) -> int:
        """Sample a negative from valid_items that's not in seen_set."""
        while True:
            cand = int(self.valid_items[np.random.randint(0, len(self.valid_items))])
            if cand not in seen_set:
                return cand

    def _build_valid_items(self) -> np.ndarray:
        """Build a fresh global negative pool based on recent activity cutoff."""
        valid: List[int] = []
        cutoff_dt = datetime(2025, 5, 29, 0, 0, 0)
        cutoff_ts = int(cutoff_dt.timestamp())

        cache_dir = os.environ.get("USER_CACHE_PATH")
        ts_path = Path(cache_dir) / "item_last_ts.json" if cache_dir else None
        last_ts = None
        if ts_path and ts_path.exists():
            try:
                with open(ts_path, "r", encoding="utf-8") as f:
                    last_ts = json.load(f)
            except Exception as e:
                warnings.warn(f"loading {ts_path} failed: {e}")

        if last_ts:
            for iid in range(1, self.itemnum + 1):
                sid = str(iid)
                if sid in self.item_feat_dict and sid in last_ts and int(last_ts[sid]) > cutoff_ts:
                    valid.append(iid)
        else:
            # Fallback: items present in item_feat_dict
            valid = [iid for iid in range(1, self.itemnum + 1) if str(iid) in self.item_feat_dict]

        return np.asarray(valid, dtype=np.int32)

    # ---------- feature schema / defaults ----------
    def _init_feat_info(self) -> Tuple[Dict[str, Any], Dict[str, List[str]], Dict[str, int]]:
        feat_default_value: Dict[str, Any] = {}
        feat_statistics: Dict[str, int] = {}
        feat_types: Dict[str, List[str]] = {}

        feat_types["user_sparse"] = ["103", "104", "105", "109"]
        feat_types["item_sparse"] = [
            "100",
            "117",
            "118",
            "101",
            "102",
            "119",
            "120",
            "114",
            "112",
            "121",
            "115",
            "122",
            "116",
        ]
        feat_types["item_array"] = []
        feat_types["user_array"] = ["106", "107", "108", "110"]
        feat_types["item_emb"] = self.mm_emb_ids
        feat_types["user_continual"] = []
        feat_types["item_continual"] = []

        for fid in feat_types["user_sparse"]:
            feat_default_value[fid] = 0
            feat_statistics[fid] = len(self.indexer["f"][fid])
        for fid in feat_types["item_sparse"]:
            feat_default_value[fid] = 0
            feat_statistics[fid] = len(self.indexer["f"][fid])
        for fid in feat_types["item_array"]:
            feat_default_value[fid] = [0]
            feat_statistics[fid] = len(self.indexer["f"][fid])
        for fid in feat_types["user_array"]:
            feat_default_value[fid] = [0]
            feat_statistics[fid] = len(self.indexer["f"][fid])
        for fid in feat_types["user_continual"]:
            feat_default_value[fid] = 0
        for fid in feat_types["item_continual"]:
            feat_default_value[fid] = 0

        # item_emb defaults: zeros of appropriate dimensionality (rows fetched from memmap later)
        for fid in feat_types["item_emb"]:
            D = self.mm_tables[fid].shape[1]
            feat_default_value[fid] = np.zeros(D, dtype=np.float32)

        return feat_default_value, feat_types, feat_statistics

    def fill_missing_feat(self, feat: Optional[Dict[str, Any]], item_id: int) -> Dict[str, Any]:
        """
        Fill missing non-embedding keys with defaults.
        Note: item_emb is *not* injected here; embeddings are gathered from memmaps in collate.
        """
        feat = {} if feat is None else feat
        filled = {k: v for k, v in feat.items()}

        non_emb_types = {
            "user_sparse",
            "item_sparse",
            "item_array",
            "user_array",
            "user_continual",
            "item_continual",
        }
        all_non_emb_ids: List[str] = []
        for tname, ids in self.feature_types.items():
            if tname in non_emb_types:
                all_non_emb_ids.extend(ids)

        missing = set(all_non_emb_ids) - set(filled.keys())
        for fid in missing:
            filled[fid] = self.feature_default_value[fid]
        return filled

    # ---------- pack: list-of-dict -> tensors; embeddings gathered via memmap ----------
    def _pack_features_batch(
        self,
        feats_list: List[np.ndarray],
        item_ids: Optional[np.ndarray] = None,  # (B, L), non-item positions set to 0
    ) -> Dict[str, torch.Tensor]:
        B = len(feats_list)
        L = len(feats_list[0])
        flat = [d for seq in feats_list for d in seq]  # B*L
        BL = B * L

        out: Dict[str, torch.Tensor] = {}

        # 1) Sparse ids (int64, B×L)
        sparse_ids = self.feature_types["item_sparse"] + self.feature_types["user_sparse"]
        for k in sparse_ids:
            dv = self.feature_default_value[k]
            arr = np.fromiter((d.get(k, dv) for d in flat), dtype=np.int64, count=BL)
            out[k] = torch.from_numpy(arr.reshape(B, L))

        # 2) Continuous (float32, B×L)
        cont_ids = self.feature_types["item_continual"] + self.feature_types["user_continual"]
        for k in cont_ids:
            dv = float(self.feature_default_value[k])
            arr = np.fromiter((d.get(k, dv) for d in flat), dtype=np.float32, count=BL)
            out[k] = torch.from_numpy(arr.reshape(B, L))

        # 3) Variable-length arrays (int64, B×L×Amax)
        array_ids = self.feature_types["item_array"] + self.feature_types["user_array"]
        for k in array_ids:
            dv = self.feature_default_value[k]  # e.g. [0]
            vals_list = [np.asarray(d.get(k, dv), dtype=np.int64) for d in flat]
            lens = np.fromiter((v.shape[0] for v in vals_list), dtype=np.int64, count=BL)
            Amax = max(int(lens.max()) if lens.size else 1, 1)

            arr = np.zeros((BL, Amax), dtype=np.int64)
            tot = int(lens.sum())
            if tot > 0:
                mask = np.arange(Amax)[None, :] < lens[:, None]
                flat_vals = np.concatenate([v for v in vals_list if v.size > 0], axis=0)
                arr[mask] = flat_vals
            out[k] = torch.from_numpy(arr.reshape(B, L, Amax))

        # 4) Multimodal embeddings (float32, B×L×D) — gather from memmaps
        emb_ids = self.feature_types["item_emb"]
        if item_ids is None:
            for k in emb_ids:
                dv = self.feature_default_value[k]
                D = int(np.asarray(dv).shape[0])
                out[k] = torch.zeros((B, L, D), dtype=torch.float32)
        else:
            ids = np.asarray(item_ids, dtype=np.int64)  # (B, L)
            ids_flat = ids.reshape(-1)  # BL
            for k in emb_ids:
                table = self.mm_tables[k]  # (itemnum+1, D)
                arr = table[ids_flat].reshape(B, L, table.shape[1])
                out[k] = torch.from_numpy(np.array(arr, copy=False))
        return out

    # ---------- __getitem__ (train/valid) ----------
    def __getitem__(self, uid: int):
        """
        Returns:
          seq, pos, neg, token_type, next_token_type, next_action_type,
          seq_time, next_time, seq_feat, pos_feat, neg_feat
        """
        user_sequence = self._load_user_data(uid)

        # Elements: (token_id, feat_dict, token_type, action_type, ts)
        ext_user_sequence: List[Tuple[int, Dict[str, Any], int, Optional[int], int]] = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, ts = record_tuple
            if u and user_feat:
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type, 0))
            if u and (not i) and (not user_feat) and (not item_feat):
                ext_user_sequence.insert(0, (u, {}, 2, action_type, 0))
            if i and item_feat:
                ext_user_sequence.append((i, item_feat, 1, action_type, ts))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        pos = np.zeros([self.maxlen + 1], dtype=np.int32)
        neg = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        seq_time = np.zeros([self.maxlen + 1], dtype=np.int64)
        next_time = np.zeros([self.maxlen + 1], dtype=np.int64)

        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        pos_feat = np.empty([self.maxlen + 1], dtype=object)
        neg_feat = np.empty([self.maxlen + 1], dtype=object)

        nxt = ext_user_sequence[self.latest] if len(ext_user_sequence) > 0 else (0, {}, 1, None, 0)
        idx = self.maxlen

        # Items seen by the user (for negative sampling)
        seen_item_ids: set[int] = set()
        for i0, _, t0, _, _ in ext_user_sequence:
            if t0 == 1 and i0:
                seen_item_ids.add(i0)

        # Left-padding: fill from the back
        for record_tuple in reversed(ext_user_sequence[: self.latest]):
            i, feat, type_, act_type, ts = record_tuple
            next_i, next_feat, next_type, next_act_type, next_ts = nxt

            feat = self.fill_missing_feat(feat, i)
            next_feat = self.fill_missing_feat(next_feat, next_i)

            seq[idx] = i
            token_type[idx] = type_
            next_token_type[idx] = next_type
            if next_act_type is not None:
                next_action_type[idx] = next_act_type
            seq_feat[idx] = feat
            seq_time[idx] = ts

            # Positive/negative (only when next is item)
            if next_type == 1 and next_i != 0:
                pos[idx] = next_i
                pos_feat[idx] = next_feat
                next_time[idx] = next_ts

                neg_id = self._random_neq(seen_item_ids)
                neg[idx] = neg_id
                neg_feat[idx] = self.fill_missing_feat(
                    self.item_feat_dict.get(str(neg_id), {}), neg_id
                )

            nxt = record_tuple
            idx -= 1
            if idx == -1:
                break

        # Fill empty object slots with defaults
        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)
        pos_feat = np.where(pos_feat == None, self.feature_default_value, pos_feat)
        neg_feat = np.where(neg_feat == None, self.feature_default_value, neg_feat)

        return (
            seq,
            pos,
            neg,
            token_type,
            next_token_type,
            next_action_type,
            seq_time,
            next_time,
            seq_feat,
            pos_feat,
            neg_feat,
        )

    def __len__(self) -> int:
        return len(self.seq_offsets)

    # ---------- collate (embeddings gathered via memmaps) ----------
    def collate_fn(self, batch):
        (
            seq,
            pos,
            neg,
            token_type,
            next_token_type,
            next_action_type,
            seq_time,
            next_time,
            seq_feat,
            pos_feat,
            neg_feat,
        ) = zip(*batch)

        seq_np = np.array(seq)
        pos_np = np.array(pos)
        neg_np = np.array(neg)
        ttype_np = np.array(token_type)

        # Only positions with token_type==1 are items; others set to 0 (memmap row 0 is zeros)
        seq_item_ids = np.where(ttype_np == 1, seq_np, 0)

        # Basic tensors
        seq = torch.from_numpy(seq_np)
        pos = torch.from_numpy(pos_np)
        neg = torch.from_numpy(neg_np)
        token_type = torch.from_numpy(ttype_np)
        next_token_type = torch.from_numpy(np.array(next_token_type))
        next_action_type = torch.from_numpy(np.array(next_action_type))
        seq_time = torch.from_numpy(np.array(seq_time))
        next_time = torch.from_numpy(np.array(next_time))

        # Feature packing (embeddings from memmaps)
        seq_feat = self._pack_features_batch(list(seq_feat), item_ids=seq_item_ids)
        pos_feat = self._pack_features_batch(list(pos_feat), item_ids=pos_np)
        neg_feat = self._pack_features_batch(list(neg_feat), item_ids=neg_np)

        return (
            seq,
            pos,
            neg,
            token_type,
            next_token_type,
            next_action_type,
            seq_time,
            next_time,
            seq_feat,
            pos_feat,
            neg_feat,
        )


# ===============================================
# Dataset (test)
# ===============================================

class MyTestDataset(MyDataset):
    """Test dataset (predict)."""

    def __init__(self, data_dir: str, args) -> None:
        super().__init__(data_dir, args)  # parent leaves _data_file/_data_file_pid as None
        self._load_user_act_map()  # map: user_id(str) -> act_type(int)

    def _load_user_act_map(self) -> None:
        path = self.data_dir / "user_action_type.json"
        with open(path, "r", encoding="utf-8") as f:
            self.user_act_map = json.load(f)

    def _load_data_and_offsets(self) -> None:
        """Only load offsets; avoid opening the file here (workers must not share an fd)."""
        with open(Path(self.data_dir, "predict_seq_offsets.pkl"), "rb") as f:
            self.seq_offsets = pickle.load(f)

    def _ensure_data_file(self) -> None:
        """Lazily open predict_seq.jsonl for the current process."""
        pid = os.getpid()
        if (self._data_file is None) or (self._data_file_pid != pid):
            if self._data_file is not None:
                try:
                    self._data_file.close()
                except Exception:
                    pass
            self._data_file = open(
                self.data_dir / "predict_seq.jsonl",
                "rb",
                buffering=io.DEFAULT_BUFFER_SIZE,
            )
            self._data_file_pid = pid

    @staticmethod
    def _process_cold_start_feat(feat: Dict[str, Any]) -> Dict[str, Any]:
        """Cold-start handling: set unseen string values to 0 (for both scalar and list types)."""
        processed_feat: Dict[str, Any] = {}
        for feat_id, feat_value in feat.items():
            if isinstance(feat_value, list):
                processed_feat[feat_id] = [0 if isinstance(v, str) else v for v in feat_value]
            elif isinstance(feat_value, str):
                processed_feat[feat_id] = 0
            else:
                processed_feat[feat_id] = feat_value
        return processed_feat

    def __getitem__(self, uid: int):
        """
        uid: line index in predict_seq.jsonl (reid)
        Returns:
          seq, token_type, next_action_type, seq_time, next_time, seq_feat, user_id
        """
        user_sequence = self._load_user_data(uid)

        ext_user_sequence: List[Tuple[int, Dict[str, Any], int, Optional[int], int]] = []
        user_id: Optional[str] = None
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, ts = record_tuple
            if u:
                if isinstance(u, str):  # real user_id
                    user_id = u
                else:                   # int(re_id) -> reverse map to user_id
                    user_id = self.indexer_u_rev[u]
            if u and user_feat:
                if isinstance(u, str):
                    u = 0
                user_feat = self._process_cold_start_feat(user_feat) if user_feat else user_feat
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type, ts))
            if u and not i and not user_feat and not item_feat:
                ext_user_sequence.insert(0, (u, {}, 2, action_type, ts))
            if i and item_feat:
                # Unseen item ids in predict set (huge creative_id) -> 0
                if i > self.itemnum:
                    i = 0
                item_feat = self._process_cold_start_feat(item_feat) if item_feat else item_feat
                ext_user_sequence.append((i, item_feat, 1, action_type, ts))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        seq_time = np.zeros([self.maxlen + 1], dtype=np.int64)
        next_time = np.zeros([self.maxlen + 1], dtype=np.int64)
        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)

        idx = self.maxlen
        ua = self.user_act_map.get(str(user_id))
        next_action_type[idx] = np.int32(ua)

        # Normalize the first timestamp to 0
        i0, feat0, type0, act0, _ = ext_user_sequence[0]
        ext_user_sequence[0] = (i0, feat0, type0, act0, 0)

        for record_tuple in reversed(ext_user_sequence):
            i, feat, type_, act_type, ts = record_tuple
            feat = self.fill_missing_feat(feat, i)
            seq[idx] = i
            token_type[idx] = type_
            seq_feat[idx] = feat
            seq_time[idx] = ts
            if type_ == 1:
                if act_type is not None:
                    next_action_type[idx - 1] = act_type
            idx -= 1
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)

        return seq, token_type, next_action_type, seq_time, next_time, seq_feat, user_id

    def __len__(self) -> int:
        return len(self.seq_offsets)

    def collate_fn(self, batch):
        """
        Returns:
          seq: (B, L+1) LongTensor
          token_type: (B, L+1) LongTensor
          seq_feat: Dict[str, Tensor] (CPU)
          user_id: List[str]
        """
        seq, token_type, next_action_type, seq_time, next_time, seq_feat, user_id = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        token_type = torch.from_numpy(np.array(token_type))
        next_action_type = torch.from_numpy(np.array(next_action_type))
        seq_time = torch.from_numpy(np.array(seq_time))
        next_time = torch.from_numpy(np.array(next_time))
        # Test set: embeddings default to zeros (no memmap gather)
        seq_feat = self._pack_features_batch(list(seq_feat))
        return seq, token_type, next_action_type, seq_time, next_time, seq_feat, list(user_id)

# ===============================================
# Utilities: build/open memmapped dense multimodal embedding
# ===============================================

def build_mm_memmaps(
    mm_path: Path,
    feat_ids: List[str],
    indexer: Dict[str, Any],
    itemnum: int,
    out_dir: Path,
) -> None:
    """
    Materialize creative_emb embeddings (JSON lines or PKL) into memmapped dense tables.
    For each feature id (fid), generate a float32 memmap of shape (itemnum+1, D),
    where row 0 is reserved as all zeros.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for fid in tqdm(feat_ids, desc="Build memmap tables"):
        if fid not in SHAPE_DICT:
            raise KeyError(f"Unknown fid {fid} in SHAPE_DICT")
        D = SHAPE_DICT[fid]

        mmap_path = out_dir / f"emb_table_{fid}_{D}.mmap"
        table = np.memmap(mmap_path, dtype=np.float32, mode="w+", shape=(itemnum + 1, D))
        table[:] = 0.0  # row 0 zeroed

        if fid != "81":
            base = mm_path / f"emb_{fid}_{D}"
            if not base.exists():
                warnings.warn(f"memmap build skipped for fid={fid}: missing directory {base}")
            else:
                for jf in base.glob("*.json"):
                    with open(jf, "r", encoding="utf-8") as f:
                        for line in f:
                            o = json.loads(line)
                            cid = o["anonymous_cid"]
                            vec = np.asarray(o["emb"], dtype=np.float32)
                            iid = indexer["i"].get(cid)
                            if iid is not None and 0 < iid <= itemnum:
                                table[iid] = vec
        else:
            # fid == "81": PKL layout
            pkl_path = mm_path / f"emb_{fid}_{D}.pkl"
            if not pkl_path.exists():
                warnings.warn(f"memmap build skipped for fid={fid}: missing file {pkl_path}")
            else:
                with open(pkl_path, "rb") as f:
                    emb_dict = pickle.load(f)  # {cid: np.ndarray | list}
                for cid, vec in emb_dict.items():
                    iid = indexer["i"].get(cid)
                    if iid is not None and 0 < iid <= itemnum:
                        table[iid] = np.asarray(vec, dtype=np.float32)

        table.flush()
        del table  # ensure file is closed/unmapped


def open_mm_memmaps(
    out_dir: Path,
    feat_ids: List[str],
    itemnum: int,
) -> Dict[str, np.memmap]:
    """Open memmapped dense tables in read-only mode. Returns {fid: memmap((itemnum+1, D), float32)}."""
    tables: Dict[str, np.memmap] = {}
    for fid in feat_ids:
        if fid not in SHAPE_DICT:
            raise KeyError(f"Unknown fid {fid} in SHAPE_DICT")
        D = SHAPE_DICT[fid]
        p = out_dir / f"emb_table_{fid}_{D}.mmap"
        if not p.exists():
            raise FileNotFoundError(f"Missing memmap file for fid={fid}: {p}")
        tables[fid] = np.memmap(p, dtype=np.float32, mode="r", shape=(itemnum + 1, D))
    return tables