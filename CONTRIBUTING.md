# Contributing

## Ground rules

- **Never edit `vendor/ncm`.** It is the upstream MIT repo, cloned by
  `scripts/setup.sh` and gitignored on purpose. Keep the two comparable: every
  change lives in `forge/` or `scripts/`.
- **Keep checkpoints loadable by upstream.** `tokenizer.pt` and `generator.pt`
  keep the upstream schema (`{"model": state_dict, ...}`) and the config keys, so
  `HazyResearch/ncm`'s `evaluate.py` can open them.
- **One logical change per pull request**, with the command you ran and the
  numbers you measured (loss, it/s, VRAM) in the description.

## Setup

```bash
bash scripts/setup.sh          # venv, deps, vendored upstream
.venv/bin/python -m pytest tests -q
```

The tests run on CPU and in CI (`.github/workflows/ci.yml`).

## Adding a domain

1. `scripts/build_fim_dataset.py` for a source tree, or
   `scripts/build_js_corpus.py` for a Hugging Face parquet shard.
2. Copy `configs/code_js.yaml`, point `data.corpus` (or `data.shards`) at the new
   corpus, set `train.steps` to one epoch.
3. Train the tokenizer, then the generator (`scripts/run_pilot.sh` does both).
4. Record the measured throughput in `docs/BENCH.md`; the estimator reads
   `measured_tok_it_s` / `measured_gen_it_s` from the config.

## Style

No comments unless they explain a non-obvious *why*. 4-space Python, 2-space
YAML. Keep `scripts/` runnable standalone (they are not imported by the tests).
