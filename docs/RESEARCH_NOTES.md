# Research notes — NCM, state of the art and where the gaps are

Last update: 2026-10-22. Sources: arXiv 2610.14162, the HazyResearch/ncm repo
(issues and PRs), the HF checkpoints, and the neighbouring literature.

## 1. What the paper actually claims (and what it does not)

| Claim | Number | Status |
|---|---|---|
| Faster than strong diffusion LMs | 10-31x wall clock on one H100 | author's numbers, not independently replicated |
| Fewer training tokens | 4-5x fewer than DUO/MDLM | author's numbers |
| Better on conditional generation | XSum ROUGE-L 28.0 (NFE 8) vs 26.6 (NFE 128) | author's numbers |
| Better on constraint reasoning | Sudoku hard 86.1 % (NFE 12) vs 81.3 % (NFE 16) | author's numbers |
| Beats frontier LLMs | — | **not claimed, not tested** |

Model scale: ~100M parameters (25M for Sudoku, 88M XSum, 120M WMT14, up to
725M on LM1B). Context: the target window is 64 tokens (XSum, WMT14) or 128
(LM1B, TinyStories). The generator sees `prefix_len` conditioning tokens
(96 for the conditional tasks, 1 for the unconditional ones).

**Not to be conflated with**: Meta's Large Concept Models (Dec 2024, sentence
level, SONAR space) and "NCM = Neuromorphic Cognitive Memory" (a Czech startup's
product, Aug 2026, a browser extension for persistent memory).

## 2. What is in the box (as of 2026-10-22)

* `HazyResearch/ncm`: MIT, model + evaluation code only. 675 stars, 9 open
  issues. Checkpoints for XSum, WMT14 De-En, Sudoku (easy/medium/hard), LM1B
  (97M/317M/725M), TinyStories. OpenWebText is still a `TODO.md`.
* Training code: **absent**. The author answered "Yes" on issue #10, so it is
  coming.
* Extra artifacts published by the authors: `ncm-vqvae-checkpoints` (the whole
  tokenizer ablation grid: graduated vs per-level schedules, codebook 4k-32k,
  shared vs per-level codebooks) and `ncm-synthetics-checkpoints` (their NTP
  baselines). Both are Apache-2.0, 0 likes, no README.
* `iskhare/ncm-owt-8sq-ctx64-16k` (co-author): a full replication with the
  recipe written down — multiscale residual VQ-VAE over 64-token windows,
  levels (1,4,9,16,25,36,49,64) = 204 latent positions, 16384 codes per level,
  100k steps on 9.04B tokens; generator 16 layers / 1024 wide / 171M params,
  100k steps on the frozen codes; codec reconstructs 99.4 % of content words.
  It also contains a scale-semantics probe (what each level recovers).
* Third parties: two hobbyist repos reusing the name on top of a
  "DeepSeek + MLA + GRPO" architecture. Nobody has trained NCM on code.

## 3. The gaps, ranked by how hard they are to close

| Gap | Who can close it | Why it matters |
|---|---|---|
| No training code | anyone who reimplements it (2-3 days with the published recipe) | blocks every fine-tune |
| No inference story (no serving, no CPU, no quant, memory leak, broken KV cache) | 1-2 days of work | blocks every self-hosted deployment |
| Tokenizer is task-specific and frozen | requires the training code | one tokenizer per domain |
| No hosted API | Together AI (co-author Simran Arora is their SVP Research) will serve it | generic hosting is a dead end |
| No benchmark beyond the paper's 5 tasks | anyone with a corpus | needed for a credible product |

## 4. The research questions nobody has answered

1. **Does the hierarchy encode structure or topic?** The paper probes it on
   prose (Appendix H). Nobody has probed it on code, on structured data, or on
   any domain where "structure" has a ground truth (AST, schema, table).
   Tool: `scripts/probe_levels.py`.
2. **Does coarse-to-fine help global constraints on real tasks?** Sudoku is a
   toy. The same property should show up on: JSON schema generation, table
   filling, instruction following, planning.
3. **Is NCM-FIM better than prefix-LM for code?** Upstream is prefix-LM only.
   FIM needs no mask surgery (the conditioning block becomes `[suffix ; prefix]`),
   but nobody has trained or evaluated it.
4. **What is the real cost per candidate?** The paper counts function
   evaluations, not wall clock on consumer hardware. The latency of one
   candidate on an RTX 4070 is the number that decides whether a local
   autocomplete is viable at all.

## 5. The wedge that is actually open

The frontier of "fast generation" is being industrialised elsewhere: Google
DiffusionGemma (26B A4B, native in vLLM, NVIDIA NIM container, ~1500 tok/s on
H100), Mercury (Inception, proprietary API), LLaDA-8B, Dream-7B, Seed
Diffusion, Gemini Diffusion. Those are bigger and already served.

So NCM is not a "faster model" story. The open wedges are:

* **domain models nobody else has** (a fine-tune on a proprietary corpus is not
  reproducible by a competitor);
* **on-premise / sovereign deployment** (a 100M model runs on a workstation or
  even a CPU; the data never leaves);
* **deterministic structured outputs** (short, schema-bound outputs are exactly
  what a 100M model can do);
* **the tooling itself**: the training code, the serving path, the benchmarks.
  That is what this repository is.

## 6. Experiment plan (in order)

1. `scripts/probe_levels.py` on the upstream Sudoku / XSum checkpoints: does
   level 1-3 look like a structure or like a topic? Measure per-candidate
   latency on the RTX 4070.
2. `scripts/smoke.sh`: prove the pipeline (done, works).
3. `configs/code_pilot.yaml`: 14 h / ~3 USD on one 4070 — a real code model,
   end to end, with the published recipe scaled down.
4. Benchmark against `llama.cpp` + Qwen2.5-Coder-1.5B on the same machine:
   quality, latency, energy. NCM is expected to lose on quality; the pitch is
   latency, privacy and cost, and the benchmark must say so honestly.
5. If the latency and the quality curve hold, the product is a local, private,
   first-draft-later-refine completion server; if not, the fallback is the
   tooling (the CLI, the checkpoints, the benchmark) sold as a service.
