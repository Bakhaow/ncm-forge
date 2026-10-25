"""Memory-mapped FIM shards.

`build_fim_dataset` keeps every tokenised prefix/suffix in Python lists, which
costs ~900 ints per window: 3.6 M windows need ~90 GB and cannot fit in the
7 GB of this box. Shards store the same tensors as int32 blobs and memory-map
them, so host RAM stays flat and the corpus size is bounded by disk instead.

Layout (cond.bin, tgt.bin, meta.json):

  cond (N, prefix_len + suffix_len) int32, [suffix ; prefix] by default
  tgt  (N, target_len) int32, the middle
  the attention mask is (cond == pad_id), computed on the fly
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ShardDataset(Dataset):
    def __init__(self, path):
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        self.n = int(meta["n"])
        self.width = int(meta["prefix_len"])
        self.target_len = int(meta["target_len"])
        self.pad_id = int(meta["pad_id"])
        self.cond = np.memmap(path / "cond.bin", dtype=np.int32, mode="r",
                              shape=(self.n, self.width))
        self.tgt = np.memmap(path / "tgt.bin", dtype=np.int32, mode="r",
                             shape=(self.n, self.target_len))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        cond = torch.from_numpy(np.array(self.cond[i], dtype=np.int64))
        tgt = torch.from_numpy(np.array(self.tgt[i], dtype=np.int64))
        return tgt, cond, cond == self.pad_id


def build_shards(records, pre_tokenizer, out_dir, prefix_len, suffix_len,
                 target_len, pad_id, cond_order="suffix_first", batch=512,
                 max_records=0, log_every=200000):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    width = int(prefix_len) + int(suffix_len)
    n = 0
    # 64 tokens never need more than ~700 chars of context; trimming before the
    # tokenizer is what makes 3.6 M windows cheap to encode
    keep_chars = 1400

    with (out / "cond.bin").open("wb") as fc, (out / "tgt.bin").open("wb") as ft:
        pres, sufs, mids = [], [], []

        def flush():
            nonlocal n
            if not pres:
                return
            enc_p = pre_tokenizer(pres, add_special_tokens=False)["input_ids"]
            enc_s = pre_tokenizer(sufs, add_special_tokens=False)["input_ids"]
            enc_m = pre_tokenizer(mids, add_special_tokens=False)["input_ids"]
            rows_c = np.full((len(pres), width), pad_id, dtype=np.int32)
            rows_t = np.full((len(pres), target_len), pad_id, dtype=np.int32)
            for j, (p, s, m) in enumerate(zip(enc_p, enc_s, enc_m)):
                p = list(p[-prefix_len:])
                s = list(s[:suffix_len])
                m = list(m[:target_len])
                if not m:
                    continue
                block = (s if cond_order == "suffix_first" else p) + \
                        (p if cond_order == "suffix_first" else s)
                rows_c[j, :len(block)] = block
                rows_t[j, :len(m)] = m
            rows_c.tofile(fc)
            rows_t.tofile(ft)
            n += len(pres)
            pres.clear(); sufs.clear(); mids.clear()

        for rec in records:
            if max_records and n + len(pres) >= max_records:
                break
            mid = rec.get("middle") or rec.get("target") or ""
            if not mid.strip():
                continue
            pres.append(rec.get("prefix", "")[-keep_chars:])
            sufs.append(rec.get("suffix", "")[:keep_chars])
            mids.append(mid[:keep_chars])
            if len(pres) >= batch:
                flush()
                if log_every and n % log_every < batch:
                    print(f"[shards] {n} windows", flush=True)
        flush()

    (out / "meta.json").write_text(json.dumps({
        "n": n, "prefix_len": width, "target_len": target_len,
        "pad_id": pad_id, "cond_order": cond_order,
        "prefix_half": prefix_len, "suffix_half": suffix_len,
    }, indent=2))
    return n
