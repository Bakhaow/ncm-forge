# ncm-forge

Training code for **Next-Concept Modeling** (NCM) — the coarse-to-fine parallel
text generation framework from Stanford's Hazy Research lab
([arXiv:2610.14162](https://arxiv.org/abs/2610.14162), MIT).

Upstream ships the model code and the checkpoints but **no training code**
(issues #2 and #10 of the official repo). This project fills that gap: it reuses
the upstream modules verbatim and adds the two training loops, a dataset
pipeline, a FIM (fill-in-the-middle) variant and a cost estimator.

**Status: no trained model is shipped here.** The repo is the training code
plus measured numbers. The latency figures below come from random weights (same
compute, same memory); the coarse-to-fine probe runs on the official Sudoku
checkpoint. Any claim about model quality would need a real run.

```
ncm-forge/
  forge/
    data.py             corpus -> token windows (unconditional / conditional / FIM)
    train_tokenizer.py  stage 1: multiscale residual VQ-VAE
    train_generator.py  stage 2: NCM transformer on frozen codes
    fim.py              fill-in-the-middle variant + generation helper
    estimate.py         GPU time / money estimator
    cli.py              command line
  configs/
    code_171m.yaml      the published recipe, retargeted at code
    code_pilot.yaml     the same, sized for a single day on one consumer GPU
    smoke.yaml          tiny end-to-end configuration
  tests/
    test_pipeline.py     non-regression tests (level offsets, FIM alignment)
  scripts/
    bench_generate.py     latency/throughput of a config (random weights)
    build_fim_dataset.py  source tree -> FIM records
    generate.py           generation with a trained checkpoint
    probe_levels.py       coarse-to-fine probe on any text + latency
    smoke.sh              end-to-end sanity run
  vendor/ncm            upstream repo (MIT), unmodified
  docs/BENCH.md         measured latency, throughput and memory limits (RTX 4070)
  docs/RESEARCH_NOTES.md  state of the art, gaps, experiment plan
```

## Measured on a consumer GPU (RTX 4070, 12 GB)

Inference, `code_pilot` shape (136 M generator + 156 M tokenizer, 204 codes):

| batch | best-of-N | per request | codes/s |
|---|---|---|---|
| 1 | 1 | 66.9 ms | 3 048 |
| 8 | 1 | **21.6 ms** | 9 459 |
| 1 | 8 | 560.7 ms | 364 |

A single request is launch-bound (31 ms of CUDA for 67 ms of wall clock, ~3 800
kernels); batching amortises it 3x. One 4070 therefore serves ~46 requests/s,
i.e. **$1.80 per million requests** at $0.30/h of rental.

Training, same card (tokenizer at effective batch 256): **1.6 it/s tokenizer,
0.34 it/s generator**, 8.8 GiB VRAM, 10.5 GB of GPU memory in use. The full
published recipe (100k x 512) would take ~180 h here; `code_pilot.yaml` is
sized for ~10 h. Limits, OOM thresholds and the 4x speedup from keeping the
tokenizer logits in bf16 are in `docs/BENCH.md`.

## The pilot in one command

```bash
.venv/bin/python scripts/build_fim_dataset.py --src ~/projects \
    --out data/corpus.jsonl --max-records 150000     # code-only, ~20M tokens
bash scripts/run_pilot.sh --force                    # tokenizer then generator
tail -f logs/pilot.log
```

## Why reuse upstream instead of forking it

The checkpoints produced here are **drop-in for the upstream `evaluate.py`**:
the config schema is copied verbatim, the tokenizer checkpoint is saved as
`{"model": state_dict}` and the generator checkpoint as `{"model": state_dict,
"config": ...}`, exactly what `scripts/eval_shard.py` expects.

## Install

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch transformers omegaconf hydra-core \
    rouge_score sacrebleu regex numpy
git clone --depth 1 https://github.com/HazyResearch/ncm.git vendor/ncm
```

## Quick start

```bash
# 0. non-regression tests (no pytest needed, runs standalone)
.venv/bin/python tests/test_pipeline.py

# 1. end-to-end sanity check (synthetic corpus, a few steps, prints the levels)
bash scripts/smoke.sh

# 2. latency on this GPU (no training needed)
.venv/bin/python scripts/bench_generate.py --config configs/code_pilot.yaml

# 3. what will this cost me?
python -m forge.cli estimate --config configs/code_pilot.yaml --gpu rtx4070

# 4. build a FIM dataset from a source tree
.venv/bin/python scripts/build_fim_dataset.py --src ~/projects --out data/code.jsonl

# 5. stage 1 - the multiscale tokenizer
.venv/bin/python -m forge.cli train-tokenizer --config configs/code_pilot.yaml

# 6. stage 2 - the generator
.venv/bin/python -m forge.cli train-generator --config configs/code_pilot.yaml
```

Any config key can be overridden on the command line (`key=value`).

## The recipe (from the upstream replication `iskhare/ncm-owt-8sq-ctx64-16k`)

| Stage | Setting |
|---|---|
| Tokenizer | multiscale residual VQ-VAE, 64-token windows, levels `(1,4,9,16,25,36,49,64)` = 204 latent positions, 16384 codes per level, 100k steps on 9.04B tokens |
| Generator | 16 layers, width 1024, trained on the frozen codec's codes, block-diagonal attention, conditioning corruption |

### What it actually costs (measured with `forge.cli estimate`)

| Config | Generator | Wall clock on one RTX 4070 | GPU budget |
|---|---|---|---|
| `code_pilot.yaml` (12 layers, 768 wide, 8192 codes, 10k steps) | 136 M params | **~8 h** | ~2 USD |
| `code_171m.yaml` (16 layers, 1024 wide, 16384 codes, 100k steps) | 336 M params | **~180 h** | ~36 USD |

The generator processes `prefix_len + sum(levels[1:])` = 331 positions per
sample, not 64: the hierarchy has one position per code, and that is what makes
the training bill 3x higher than a naive "tokens in the window" count suggests.

The published replication reports a 171M generator. This repo's config uses
untied trainable heads and predicts level 0 (needed for FIM), which brings it to
336M. Tie the heads to the codebooks (`tie_heads_to_codebook: true`) and set
`predict_level0: false` to get back to that figure, at the cost of the
level-0 head.

## Fill in the middle (FIM)

The upstream generator is a prefix-LM: the conditioning block precedes the
target and the target attends to it. FIM needs no mask surgery — the
conditioning block simply becomes `[suffix ; prefix]` and
`model.prefix_len = prefix_len + suffix_len`:

```bash
python -m forge.cli make-fim --config configs/code_171m.yaml \
    --prefix-len 64 --suffix-len 64 --out configs/fim.yaml
```

Putting the suffix first means the level-0 head (the coarsest level, one code
per window) reads the code sitting right before the cursor: the global shape
of the completion is decided from the local context. The prefix keeps its
**tail** when it is longer than the budget — the tokens adjacent to the cursor
are the ones that matter.

Inference helper:

```python
from forge.fim import generate_fim
levels, final = generate_fim(net, tk, prefix_ids, suffix_ids, 64, 64)
```

`levels` is the list of progressively refined decodings, one per hierarchy
level — the coarse-to-fine view that the paper renders as a GIF.

## Deviations from upstream, and why

* **The EMA accumulator is zeroed** when the tokenizer is built. Upstream seeds
  `ema_w` with the random init (`vendor/ncm/models/quant.py:303`), so the first
  EMA step divides that init by `(1-decay)*count`: the code norms blow up to
  ~1e6, `vq_loss` starts around 1e3 and the quantiser picks codes by norm
  instead of direction for the first ~100 steps.
* **`query_level0: true` is rejected** with an explicit error: `idx_to_ncm_input`
  only produces `levels[1:]` while the model would expect a level-0 chunk.
* The prefix construction mirrors upstream's `generate()` including the
  `contextual_prefix` branch, and `raw_token_prefix` defaults to `False` as
  upstream does.

## What upstream still does not give you

* `generate()` keeps its tensors alive between calls (peak memory ~2.3x higher
  than needed) — issue #4.
* the `use_kv_cache=True` path produces different outputs than the eager path
  and refuses the `query_level0` models — issue #5. Once fixed it is
  2.1-2.4x (WMT14) and 4.5-5.1x (XSum) faster, and it only applies to models
  with `query_level0: false`.
* no CPU/MPS path in the released `evaluate.py` — issue #6.

Those three are the cheapest possible contributions to upstream and they
unblock every self-hosted deployment.
