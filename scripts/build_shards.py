"""Encode FIM jsonl records into memory-mapped shards.

    python scripts/build_shards.py --corpus data/js_ab/fim.jsonl \
        --out data/js_shards --prefix-len 64 --suffix-len 64 --target-len 64
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forge.shards import build_shards  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--prefix-len", type=int, default=64)
    ap.add_argument("--suffix-len", type=int, default=64)
    ap.add_argument("--target-len", type=int, default=64)
    ap.add_argument("--cond-order", default="suffix_first")
    ap.add_argument("--max-records", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def records():
        with open(args.corpus, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)

    n = build_shards(records(), tok, args.out, args.prefix_len, args.suffix_len,
                     args.target_len, pad, args.cond_order,
                     max_records=args.max_records)
    print(f"{n} windows -> {args.out}")


if __name__ == "__main__":
    main()
