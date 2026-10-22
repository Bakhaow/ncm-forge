"""ncm-forge command line.

    python -m forge.cli estimate        --config configs/code_171m.yaml --gpu rtx4070
    python -m forge.cli train-tokenizer --config configs/code_code.yaml
    python -m forge.cli train-generator --config configs/code_code.yaml
    python -m forge.cli make-fim        --config configs/code_code.yaml --prefix-len 64 --suffix-len 64
    python -m forge.cli smoke           --config configs/smoke.yaml
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from forge import config as cfgmod
from forge.estimate import estimate, format_estimate


def _load(args):
    return cfgmod.load(args.config, args.overrides)


def cmd_estimate(args):
    from forge.estimate import GPU_PRESETS

    if args.gpu not in GPU_PRESETS:
        raise SystemExit(
            f"unknown gpu {args.gpu!r}; known: {', '.join(sorted(GPU_PRESETS))}"
        )
    cfg = _load(args)
    est = estimate(cfg, gpu=args.gpu, mfu=args.mfu)
    if args.json:
        print(json.dumps(est, indent=2))
    else:
        print(format_estimate(est))
    return 0


def cmd_train_tokenizer(args):
    from forge.train_tokenizer import train

    cfg = _load(args)
    train(cfg)
    return 0


def cmd_train_generator(args):
    from forge.train_generator import train

    cfg = _load(args)
    train(cfg)
    return 0


def cmd_make_fim(args):
    from omegaconf import OmegaConf

    from forge.fim import make_fim_config

    cfg = _load(args)
    out = make_fim_config(cfg, args.prefix_len, args.suffix_len)
    dest = Path(args.out or "configs/fim.yaml")
    cfgmod.save(out, dest)
    print(f"wrote {dest}")
    return 0


def cmd_smoke(args):
    """End-to-end sanity run: tiny corpus -> tokenizer -> generator -> generation."""
    root = Path(__file__).resolve().parent.parent
    script = root / "scripts" / "smoke.sh"
    cmd = ["bash", str(script)]
    if args.config:
        cmd += ["--config", args.config]
    return subprocess.call(cmd, cwd=root)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="forge")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("estimate", help="estimate GPU time and cost")
    p.add_argument("--config", required=True)
    p.add_argument("--gpu", default="rtx4070")
    p.add_argument("--mfu", type=float, default=0.35)
    p.add_argument("--json", action="store_true")
    p.add_argument("overrides", nargs="*")
    p.set_defaults(func=cmd_estimate)

    p = sub.add_parser("train-tokenizer", help="stage 1: train the VQ-VAE tokenizer")
    p.add_argument("--config", required=True)
    p.add_argument("overrides", nargs="*")
    p.set_defaults(func=cmd_train_tokenizer)

    p = sub.add_parser("train-generator", help="stage 2: train the NCM generator")
    p.add_argument("--config", required=True)
    p.add_argument("overrides", nargs="*")
    p.set_defaults(func=cmd_train_generator)

    p = sub.add_parser("make-fim", help="derive a FIM config from a base config")
    p.add_argument("--config", required=True)
    p.add_argument("--prefix-len", type=int, default=64)
    p.add_argument("--suffix-len", type=int, default=64)
    p.add_argument("--out", default="")
    p.add_argument("overrides", nargs="*")
    p.set_defaults(func=cmd_make_fim)

    p = sub.add_parser("smoke", help="tiny end-to-end run")
    p.add_argument("--config", default="")
    p.set_defaults(func=cmd_smoke)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
