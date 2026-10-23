"""Build a JavaScript corpus straight from bigcode/starcoderdata parquet shards.

Two outputs:

  raw.jsonl  {"text": <file>}             -> stage 1 (tokenizer, unconditional)
  fim.jsonl  {"prefix","middle","suffix"} -> stage 2 (generator, FIM)

FIM windows are *disjoint*: the file is cut into consecutive middles of a few
lines, a window's prefix/suffix is the neighbouring text. The overlapping-line
sampler in build_fim_dataset.py multiplies the same tokens ~16x, which inflates
the token count and lets the model memorise long spans.

    python scripts/build_js_corpus.py --parquet data/raw/.../*.parquet \
        --out-dir data/js --max-tokens 250000000
"""

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

HEADER = re.compile(r"^<reponame>.*?<filename>[^\n]*\n", re.S)


def clean(content: str) -> str | None:
    """Strip the starcoderdata header, reject generated or minified files."""
    content = HEADER.sub("", content, count=1)
    if len(content) < 1500:
        return None
    if "sourceMappingURL=" in content or "/*! For license information" in content:
        return None
    lines = content.splitlines()
    if len(lines) < 25:
        return None
    if sum(len(line) for line in lines) / len(lines) > 160:
        return None
    return content


def windows(content: str, rng: random.Random, middle_lines: int, context_lines: int):
    """Disjoint middles: stride equals the middle length, no shared target tokens."""
    lines = content.splitlines()
    for start in range(0, len(lines) - middle_lines, middle_lines):
        size = rng.randint(max(2, middle_lines - 2), middle_lines + 2)
        end = min(start + size, len(lines))
        if end - start < 2:
            continue
        yield {
            "prefix": "\n".join(lines[max(0, start - context_lines):start]) + "\n",
            "middle": "\n".join(lines[start:end]),
            "suffix": "\n".join(lines[end:end + context_lines]),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", nargs="+", required=True)
    ap.add_argument("--out-dir", default="data/js")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B",
                    help="pre-tokenizer used only to count and cap tokens")
    ap.add_argument("--skip-count", action="store_true",
                    help="skip the token count (it costs ~4 min on 800 MB)")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="stop once the raw corpus reaches this many tokens")
    ap.add_argument("--context-lines", type=int, default=14)
    ap.add_argument("--middle-lines", type=int, default=6)
    ap.add_argument("--max-per-repo", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tok = None if args.skip_count else AutoTokenizer.from_pretrained(args.tokenizer)
    rng = random.Random(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    per_repo: Counter = Counter()
    stats = Counter()
    tokens = 0
    middle_lens: list[int] = []

    with (out / "raw.jsonl").open("w", encoding="utf-8") as raw, \
         (out / "fim.jsonl").open("w", encoding="utf-8") as fim:
        for path in args.parquet:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=4000, columns=[
                    "content", "max_stars_repo_name", "max_stars_repo_path"]):
                contents = batch.column("content").to_pylist()
                repos = batch.column("max_stars_repo_name").to_pylist()
                paths = batch.column("max_stars_repo_path").to_pylist()
                for content, repo, fpath in zip(contents, repos, paths):
                    stats["rows"] += 1
                    if args.max_tokens and tokens >= args.max_tokens:
                        break
                    if not fpath.endswith((".js", ".jsm", ".es6")):
                        stats["not_js"] += 1
                        continue
                    doc = clean(content)
                    if doc is None:
                        stats["filtered"] += 1
                        continue
                    digest = hashlib.blake2b(doc.encode("utf-8"), digest_size=16).hexdigest()
                    if digest in seen:
                        stats["dupes"] += 1
                        continue
                    wins = list(windows(doc, rng, args.middle_lines, args.context_lines))
                    if len(wins) < 3:
                        stats["too_short"] += 1
                        continue
                    if per_repo[repo] + len(wins) > args.max_per_repo:
                        stats["repo_capped"] += 1
                        continue
                    n_tok = 0 if args.skip_count else len(tok(doc, add_special_tokens=False)["input_ids"])
                    seen.add(digest)
                    per_repo[repo] += len(wins)
                    tokens += n_tok
                    stats["files"] += 1
                    stats["windows"] += len(wins)
                    raw.write(json.dumps({"text": doc}) + "\n")
                    for w in wins:
                        w["file"] = fpath
                        w["repo"] = repo
                        fim.write(json.dumps(w) + "\n")
                        middle_lens.append(len(w["middle"].split()))
                if args.max_tokens and tokens >= args.max_tokens:
                    break
            if args.max_tokens and tokens >= args.max_tokens:
                break

    middle_lens.sort()
    report = {
        **stats,
        "tokens": tokens,
        "median_middle_words": middle_lens[len(middle_lens) // 2] if middle_lens else 0,
        "blank_middles": 0,
        "top_repos": per_repo.most_common(10),
    }
    (out / "stats.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "top_repos"}, indent=2))
    print(f"top repos: {report['top_repos']}")
    print(f"-> {out/'raw.jsonl'} ({tokens/1e6:.1f} M tokens)")
    print(f"-> {out/'fim.jsonl'} ({stats['windows']} windows, "
          f"~{stats['windows'] * 64 / 1e6:.1f} M target tokens)")


if __name__ == "__main__":
    sys.exit(main())
