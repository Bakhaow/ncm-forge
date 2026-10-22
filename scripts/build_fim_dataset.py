"""Turn a source tree into FIM training records.

For every file, each line becomes a candidate middle. The prefix is the
preceding lines, the suffix the following ones, both truncated to the model's
conditioning budget.

    python scripts/build_fim_dataset.py --src ~/projects --out data/code.jsonl \
        --prefix-lines 8 --suffix-lines 8 --max-middle-lines 4
"""

import argparse
import json
import random
from pathlib import Path

# Only real source code. JSON/HTML/Markdown were 48% of the first corpus and
# taught the tokenizer to reconstruct openapi dumps, not code.
CODE_SUFFIXES = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".kt", ".c",
    ".h", ".cpp", ".hpp", ".cs", ".rb", ".php", ".swift", ".scala", ".ex",
    ".exs", ".m", ".mm", ".lua", ".zig", ".sh", ".sql",
}


def iter_files(root: Path, suffixes: set[str], exclude: list[str]):
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        rel = str(path.relative_to(root))
        if any(ex in rel.split("/") or ex in rel for ex in exclude):
            continue
        yield path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix-lines", type=int, default=8)
    ap.add_argument("--suffix-lines", type=int, default=8)
    ap.add_argument("--max-middle-lines", type=int, default=4)
    ap.add_argument("--min-middle-lines", type=int, default=1)
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--max-records", type=int, default=0,
                    help="stop after this many records (keeps the corpus in RAM budget)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", nargs="*", default=[
        "node_modules", ".git", ".venv", "vendor", "dist", "build", "__pycache__",
        # generated / vendored / irrelevant trees
        "generated", ".worktrees", ".next", ".nuxt", "target", "coverage",
        "migrations", ".cache", "out", "tmp", "assets", "public", "static",
        "third_party", "vendor",
    ])
    ap.add_argument("--max-per-file", type=int, default=200,
                    help="cap records per file so one 15k-line file cannot dominate")
    ap.add_argument("--max-per-repo", type=int, default=40000,
                    help="cap records per top-level directory")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.src)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    per_repo: dict[str, int] = {}
    files = list(iter_files(root, CODE_SUFFIXES, args.exclude))
    rng.shuffle(files)
    if args.max_files:
        files = files[: args.max_files]

    with out.open("w", encoding="utf-8") as fh:
        for path in files:
            if args.max_records and written >= args.max_records:
                break
            repo = str(path.relative_to(root)).split("/")[0]
            if per_repo.get(repo, 0) >= args.max_per_repo:
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            if len(lines) < args.prefix_lines + args.suffix_lines + args.max_middle_lines:
                continue
            file_written = 0
            for i in range(len(lines)):
                if file_written >= args.max_per_file:
                    break
                size = rng.randint(args.min_middle_lines, args.max_middle_lines)
                start = i
                end = min(i + size, len(lines))
                if end - start < args.min_middle_lines:
                    continue
                prefix_start = max(0, start - args.prefix_lines)
                suffix_end = min(len(lines), end + args.suffix_lines)
                record = {
                    "file": str(path.relative_to(root)),
                    "prefix": "\n".join(lines[prefix_start:start]) + "\n",
                    "middle": "\n".join(lines[start:end]),
                    "suffix": "\n".join(lines[end:suffix_end]),
                }
                fh.write(json.dumps(record) + "\n")
                written += 1
                file_written += 1
                per_repo[repo] = per_repo.get(repo, 0) + 1
            if args.max_records and written >= args.max_records:
                break

    print(f"{written} records from {len(files)} files -> {out}")


if __name__ == "__main__":
    main()
