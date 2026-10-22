"""Expose the vendored HazyResearch/ncm modules (models, tokenization, utils).

The official repo is MIT licensed and is vendored as-is under vendor/ncm so that
ncm-forge trains checkpoints that load with the upstream `evaluate.py` untouched.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor" / "ncm"


def install() -> Path:
    """Put vendor/ncm on sys.path (idempotent)."""
    path = str(VENDOR)
    if path not in sys.path:
        sys.path.insert(0, path)
    return VENDOR


def require() -> Path:
    """Ensure the vendored repo is present."""
    install()
    if not (VENDOR / "models" / "next_concept.py").exists():
        raise RuntimeError(
            f"vendored NCM repo missing at {VENDOR}. "
            "Run: git clone --depth 1 https://github.com/HazyResearch/ncm.git vendor/ncm"
        )
    return VENDOR
