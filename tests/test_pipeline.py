"""Non-regression tests for the parts that break silently.

Run with: .venv/bin/python -m pytest tests -q
"""

import sys
from pathlib import Path

import torch

try:
    import pytest
except ImportError:  # pytest is optional: the file also runs standalone
    pytest = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forge import config as cfgmod
from forge import data as datamod
from forge.train_generator import level_ce, level_weights
from forge.vendor import require


def tiny_cfg():
    cfg = cfgmod.load("configs/smoke.yaml")
    return cfg


def test_config_levels():
    cfg = tiny_cfg()
    assert cfgmod.levels_of(cfg) == [1, 4, 8, 16]
    assert cfgmod.model_levels(cfg) == [4, 8, 16]
    assert cfgmod.model_vocab_sizes(cfg) == [512, 512, 512]
    assert cfgmod.model_head_levels(cfg) == [1, 4, 8, 16]
    # the generator processes the conditioning block plus one position per code
    assert cfgmod.generator_sequence_length(cfg) == 8 + 4 + 8 + 16


def test_level_weights_sum_to_one():
    w = level_weights([1, 4, 9, 16], 0.0)
    assert abs(float(w.sum()) - 1.0) < 1e-6
    assert torch.allclose(w, torch.full((4,), 0.25))


def test_level_ce_alignment():
    """Every level block must be matched with its own codes, in order."""
    levels = [1, 4, 8, 16]
    vocab = [512, 512, 512, 512]
    lvl_first = 1
    predict_level0 = True

    # one-hot logits: level k predicts the code value it is given
    idx = torch.zeros(2, sum(levels), dtype=torch.long)
    off = 0
    for k, level in enumerate(levels):
        idx[:, off : off + level] = k + 1
        off += level

    total_positions = levels[0] + sum(levels[lvl_first:])
    logits = torch.full((2, total_positions, 512), -1e4)
    pos = 0
    logits[:, pos : pos + levels[0], 1] = 1e4  # level-0 head
    pos += levels[0]
    off_idx = sum(levels[:lvl_first])
    for j, level in enumerate(levels[lvl_first:]):
        code = lvl_first + j + 1
        for i in range(level):
            logits[:, pos + i, code] = 1e4
        pos += level
        off_idx += level
    assert pos == total_positions

    loss, stats = level_ce(logits, idx, levels, vocab, lvl_first, predict_level0, 0.0)
    assert loss < 1e-3
    assert [name for name, _ in stats] == ["level0", "level1", "level2", "level3"]


def test_fim_dataset_keeps_triples_aligned():
    """prefix / suffix / middle must never be shuffled apart."""
    tok = FakeTokenizer()
    records = [
        {"prefix": f"pre{i}", "middle": f"mid{i}", "suffix": f"suf{i}"} for i in range(20)
    ]
    ds = datamod.build_fim_dataset(records, tok, prefix_len=4, suffix_len=4, target_len=4)
    seen = set()
    for i in range(len(ds)):
        tgt, cond, mask = ds[i]
        seen.add(
            (
                tuple(cond[4:].tolist()),   # prefix (tail)
                tuple(cond[:4].tolist()),   # suffix (head)
                tuple(tgt.tolist()),       # middle
            )
        )
    expected = {
        (
            tuple(tok.encode(f"pre{i}")[-4:]),  # the prefix keeps its tail
            tuple(tok.encode(f"suf{i}")[:4]),   # the suffix keeps its head
            tuple(tok.encode(f"mid{i}")[:4]),   # the middle is truncated to the budget
        )
        for i in range(20)
    }
    assert seen == expected


def test_prefix_keeps_its_tail():
    """The tokens next to the cursor must survive the truncation."""
    tok = FakeTokenizer()
    records = [{"prefix": "abcdefgh", "middle": "M", "suffix": "z"}]
    ds = datamod.build_fim_dataset(records, tok, prefix_len=4, suffix_len=4, target_len=4)
    _, cond, _ = ds[0]
    assert cond[4:].tolist() == tok.encode("efgh")  # tail of the prefix


def test_cursor_branch_middle_is_not_the_suffix():
    tok = FakeTokenizer()
    records = [{"text": "abcdefghij", "cursor": 3, "mid_len": 4}]
    ds = datamod.build_fim_dataset(records, tok, prefix_len=4, suffix_len=4, target_len=4)
    tgt, _, _ = ds[0]
    assert tgt.tolist() == tok.encode("defg")


class FakeTokenizer:
    """Minimal stand-in: one token per character."""

    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 97 + 2 for c in text]


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception as exc:
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    raise SystemExit(1 if failures else 0)
