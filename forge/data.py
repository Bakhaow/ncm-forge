"""Corpus -> token windows.

Three tasks:

  unconditional  a flat token stream chunked into fixed windows (LM1B / TinyStories style)
  conditional    (prefix -> target) pairs, prefix is the conditioning block
  fim            (prefix, suffix -> middle) pairs, the conditioning block is
                 [suffix ; prefix] so that the level-0 head reads the code that
                 sits immediately before the cursor (see forge/fim.py)

Every item is returned as (target_ids, cond_ids, cond_mask):

  target_ids  (T,)   tokens encoded by the VQ-VAE tokenizer
  cond_ids    (P,)   conditioning token ids
  cond_mask   (P,)   True where cond_ids is padding (masked out of attention)
"""

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


def read_documents(path: str | Path, fmt: str = "auto") -> list[str]:
    """Read a corpus file into a list of documents.

    jsonl : one JSON object per line; the `text` / `content` / `prefix` field is
            used when present, otherwise the raw line
    text  : one document per non-empty line
    """
    path = Path(path)
    if fmt == "auto":
        fmt = "jsonl" if path.suffix in (".jsonl", ".ndjson") else "text"
    docs: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        if fmt == "jsonl":
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    docs.append(line)
                    continue
                if isinstance(rec, dict):
                    payload = rec.get("text") or rec.get("content") or rec.get("prefix")
                    docs.append(payload if isinstance(payload, str) else line)
                else:
                    docs.append(line)
        else:
            for line in fh:
                line = line.rstrip("\n")
                if line.strip():
                    docs.append(line)
    return docs


def encode(tokenizer, text: str) -> list[int]:
    """Tokenize with either a raw HF tokenizer or a vendor tokenizer."""
    if not text:
        return []
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))



def jsonl_records(path: str | Path) -> list[dict]:
    """Read a .jsonl file into a list of dicts."""
    records = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class UnconditionalWindows(Dataset):
    """Fixed-size windows cut from a flat token stream."""

    def __init__(self, ids: torch.Tensor, length: int, prefix_len: int, prefix_id: int):
        n = (ids.numel() // length) * length
        self.data = ids[:n].view(-1, length)
        self.length = length
        self.prefix_len = prefix_len
        self.prefix_id = prefix_id

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, i: int):
        cond = torch.full((self.prefix_len,), self.prefix_id, dtype=torch.long)
        # The unconditional prefix is a real token (EOS), not padding: it must
        # stay visible, otherwise the level-0 head has nothing to read.
        mask = torch.zeros(self.prefix_len, dtype=torch.bool)
        return self.data[i], cond, mask


class ConditionalPairs(Dataset):
    """(prefix, suffix) -> middle pairs.

    `cond_order` controls how the conditioning block is laid out:
      suffix_first : [suffix ; prefix] (default) - the level-0 head then reads the
                     end of the prefix, i.e. the code right before the cursor
      prefix_first : [prefix ; suffix]
    """

    def __init__(
        self,
        prefixes: list[list[int]],
        suffixes: list[list[int]],
        targets: list[list[int]],
        prefix_len: int,
        suffix_len: int,
        target_len: int,
        pad_id: int,
        cond_order: str = "suffix_first",
    ):
        self.prefixes = prefixes
        self.suffixes = suffixes
        self.targets = targets
        self.prefix_len = prefix_len
        self.suffix_len = suffix_len
        self.target_len = target_len
        self.pad_id = pad_id
        self.cond_order = cond_order

    def __len__(self) -> int:
        return len(self.targets)

    def _fit(self, ids: list[int], width: int, keep: str = "head") -> tuple[torch.Tensor, torch.Tensor]:
        """Truncate to `width` tokens and right-pad.

        The prefix keeps its TAIL (the tokens adjacent to the cursor are the
        ones the level-0 head reads), the suffix keeps its HEAD.
        """
        ids = list(ids[-width:] if keep == "tail" else ids[:width])
        pad = width - len(ids)
        mask = torch.zeros(width, dtype=torch.bool)
        if pad > 0:
            mask[-pad:] = True
            ids = ids + [self.pad_id] * pad
        return torch.tensor(ids, dtype=torch.long), mask

    def __getitem__(self, i: int):
        pre, pre_mask = self._fit(self.prefixes[i], self.prefix_len, keep="tail")
        suf, suf_mask = self._fit(self.suffixes[i], self.suffix_len, keep="head")
        tgt, _ = self._fit(self.targets[i], self.target_len)
        if self.cond_order == "suffix_first":
            cond = torch.cat([suf, pre])
            mask = torch.cat([suf_mask, pre_mask])
        else:
            cond = torch.cat([pre, suf])
            mask = torch.cat([pre_mask, suf_mask])
        return tgt, cond, mask



def build_stream_dataset(
    ids: torch.Tensor,
    target_len: int,
    prefix_len: int,
    prefix_id: int,
) -> UnconditionalWindows:
    return UnconditionalWindows(ids, target_len, prefix_len, prefix_id)


def build_fim_dataset(
    records: list[dict],
    tokenizer,
    prefix_len: int,
    suffix_len: int,
    target_len: int,
    cond_order: str = "suffix_first",
    seed: int = 0,
) -> ConditionalPairs:
    """Turn FIM records into a dataset.

    Each record must carry `prefix`, `suffix` and `middle` (or `target`).
    A record may instead carry `text` plus `cursor` and optionally `mid_len`:
    the middle is then the `mid_len` tokens that follow the cursor.
    """
    rng = random.Random(seed)
    prefixes, suffixes, targets = [], [], []
    for rec in records:
        if "prefix" in rec and "suffix" in rec and ("middle" in rec or "target" in rec):
            pre = encode(tokenizer, rec["prefix"])
            suf = encode(tokenizer, rec["suffix"])
            mid = encode(tokenizer, rec.get("middle", rec.get("target", "")))
        elif "text" in rec and "cursor" in rec:
            text = rec["text"]
            cut = int(rec["cursor"])
            mid_len = int(rec.get("mid_len", target_len))
            pre = encode(tokenizer, text[:cut])
            suf = encode(tokenizer, text[cut + mid_len :])
            mid = encode(tokenizer, text[cut : cut + mid_len])
        else:
            continue
        if len(mid) == 0:
            continue
        prefixes.append(pre)
        suffixes.append(suf)
        targets.append(mid)
    order = list(range(len(targets)))
    rng.shuffle(order)
    prefixes = [prefixes[i] for i in order]
    suffixes = [suffixes[i] for i in order]
    targets = [targets[i] for i in order]
    return ConditionalPairs(
        prefixes, suffixes, targets, prefix_len, suffix_len, target_len,
        tokenizer.pad_token_id, cond_order,
    )


def build_conditional_dataset(
    records: list[dict],
    tokenizer,
    prefix_len: int,
    target_len: int,
    source_key: str = "source",
    target_key: str = "target",
) -> ConditionalPairs:
    """Turn (source, target) records into a dataset."""
    prefixes, suffixes, targets = [], [], []
    for rec in records:
        pre = encode(tokenizer, rec.get(source_key, ""))
        tgt = encode(tokenizer, rec.get(target_key, ""))
        if not pre or not tgt:
            continue
        prefixes.append(pre)
        suffixes.append([])
        targets.append(tgt)
    return ConditionalPairs(
        prefixes, suffixes, targets, prefix_len, 0, target_len,
        tokenizer.pad_token_id, "prefix_first",
    )
