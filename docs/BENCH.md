# Measured numbers

Machine: RTX 4070 (12 GB VRAM, 46 SMs, sm_89), driver 610.88, 7 GB host RAM,
torch 2.11.0+cu128, bf16 autocast, fp32 weights. Everything below is measured,
not estimated; reproduce with the commands at the bottom.

## Inference (random weights, same compute as a trained model)

`code_pilot` = 135.7 M generator + 155.9 M tokenizer, 204 codes per request,
128-token prefix, 8 levels.

| model | batch | candidates | p50 | per request | codes/s |
|---|---|---|---|---|---|
| code_pilot | 1 | 1 | 66.9 ms | 66.9 ms | 3 048 |
| code_pilot | 1 | 4 | 298.5 ms | 298.5 ms | 683 |
| code_pilot | 1 | 8 | 560.7 ms | 560.7 ms | 364 |
| code_pilot | 8 | 1 | 172.5 ms | **21.6 ms** | 9 459 |
| code_171m (517 M) | 1 | 1 | 98.6 ms | 98.6 ms | 2 068 |
| sudoku 25 M (upstream ckpt) | 1 | 1 | 8.9 ms | 8.9 ms | — |
| sudoku 25 M (upstream ckpt) | 1 | 8 | 71.5 ms | 8.9 ms | — |

Peak VRAM: 1.8 GB (pilot, batch 8), 2.4 GB (171M).

Reading: a single request is **launch-bound**, not compute-bound (profile: 31 ms
of CUDA, 67 ms wall; ~3 800 tiny elementwise kernels per request). Batching
amortises it 3x. One 4070 therefore serves ~46 requests/s at batch 8, i.e.
~4 M requests/day, at $1.80 per million requests (GPU rental $0.30/h).
Best-of-8 is the honest comparison with diffusion LMs (which need 64+ NFE per
sample, ours needs 7 level passes).

## Training throughput

| stage | physical batch | accumulation | effective | it/s | tokens/s (positions/s) |
|---|---|---|---|---|---|
| tokenizer | 64 | 4 | 256 | 1.06 | 17 k |
| tokenizer | 64 | 16 | 1024 | ~0.26 | 17 k |
| generator | 32 | 8 | 256 | 0.34 | 29 k positions |
| generator | 32 | 1 | 32 | ~2.7 | 29 k positions |

Generator MFU: 23 TFLOP/s of 57.7 peak (~40%), so the small-batch regime is
already near what this card can do. The pilot budget (10k x 256 x 332
positions, 135 M params) is simply 8 h of it.

## Where the time goes

- **Tokenizer**: the output head is 151 680-way (Qwen3 vocab). At batch 64 x 64
  tokens the logits alone are 2.5 GB in fp32 — `logits.float()` OOMed at batch
  256. Keeping the logits in the autocast dtype is a 4x speedup (0.25 -> 1.06
  it/s) and makes batch 64 fit.
- **Generator**: one forward per level over prefix + that level's codes; the
  prefix is recomputed at every level (upstream KV-cache path is broken, issue
  #5). That is 7 x 128 redundant positions per request.

## Memory limits (12 GB card, 7 GB RAM)

| setting | result |
|---|---|
| tokenizer batch 256, no accumulation | CUDA OOM (needs 16 GB+) |
| tokenizer batch 128 | 12.4 GB, 8x slower (thrashing) |
| tokenizer batch 64 | 7.1 GB, fine |
| generator batch 32 | 8.8 GB, fine |
| generator corpus | 151 k FIM records (~30 M tokens) fits in 1.5 GB RAM |

## Real corpus (122 k FIM records, 1 233 files, ~19.6 M tokens)

- First corpus attempt was junk: 341 files, 30 % `.json` (an 11 k-record
  openapi dump, a 15 k-record game animation), 14 % HTML, plus a client repo
  that must stay out. Rebuilt with code-only suffixes, per-file (200) and per-repo
  (50 k) caps and generated-dir excludes: 1 233 files, 44 % Java, 26 % JS,
  20 % TS.
- Tokenizer reached **98.97 % exact reconstruction at step 2 000 of 10 000**
  (25 min) — but on the junk corpus; retrain it on the new one (~20 min) with every codebook level at 100 % utilization: the stage is
  converged long before its budget. The pilot now cuts it there.
- Generator at step 150/10 000: loss 14.4 -> 6.9 (uniform CE over an 8192-code
  level is ln 8192 = 9.01, so it is learning). 0.34 it/s, 8.8 GiB VRAM,
  10.5 GB used on the GPU, ~2 GB of free host RAM: the dataset holds 149 229
  encoded windows and that is the RAM ceiling on this box.
- Reading the loss: < 6.5 beats uniform, < 4 is decent, < 3 is good
  (the 150-step overfit run on 4 k records reached 2.6).

## Reproduce

```bash
.venv/bin/python scripts/bench_generate.py --config configs/code_pilot.yaml \
    --batch 1,8 --candidates 1,8
.venv/bin/python -m forge.cli train-tokenizer --config configs/code_pilot.yaml \
    train.steps=40 train.log_every=20
.venv/bin/python -m forge.cli train-generator --config configs/code_pilot.yaml \
    data.max_docs=4000 train.steps=10 train.log_every=5
.venv/bin/python -m forge.cli estimate --config configs/code_pilot.yaml
```

## Optimisation (RTX 4070, batch 32, 332 positions)

Décomposition d'un micro-pas (moyenne sur 15 pas, `torch.cuda.Event` par phase,
avec synchronisation — le total réel est ~15 % plus bas) :

| Phase | ms | % |
|---|---|---|
| forward | 237 | 46 % |
| backward | 171 | 33 % |
| encode + prefix latents | 65 | 13 % |
| optimiseur | 35 | 7 % |
| loss (CE par niveau) | 7 | 1 % |
| DataLoader | ~1 | 0 % |

Le forward dure 1,4× le backward : le coût est dominé par les 12 blocs du
transformer (seq 332), pas par les têtes (7 × 8 192). Mesures comparatives :

| Réglage | ms/micro-pas | it/s | VRAM |
|---|---|---|---|
| fp32, sans TF32 | OOM à batch 32 | — | >12 G |
| fp32 + TF32, batch 16 | 243 | 4,1 | 7,0 G |
| bf16 autocast, batch 32 | 310 | 3,2 | 8,5 G |
| **bf16 + `torch.compile`** | **197** | **5,1** | **6,7 G** |
| bf16 + compile, batch 64 | 1 761 | 0,6 | 10,7 G |

- `torch.compile(dynamic=False)` : **1,6× sur le modèle**, et 1,8 Go de VRAM en
  moins (graphes fusionnés, moins d'activations conservées). Équivalence
  numérique vérifiée : |Δ logits| max 2,1e-6, argmax identique à 100 %.
  Premier appel : 33 s de compilation (à amortir sur ≥ quelques centaines de pas).
- `AdamW(fused=True)` : 7,2 ms contre 15,0 ms pour 200 tenseurs ; gain modeste
  (~3 % du pas) mais gratuit.
- Batch 64 : thrashing mémoire, 9× plus lent qu'avec deux pas de 32.
- Prendre les codes + latents de préfixe en amont économiserait 13 % du pas,
  mais coûte ~12 Go de cache pour 122 k fenêtres (128 × 384 floats chacune) :
  non retenu.
- `torch.compile` sur le tokenizer : 92 ms/pas contre ~78 ms en eager → aucun
  gain, non activé (l'EMA et les mises à jour de codebook ajoutent du contrôle
  de flux dynamique).

Bilan bout en bout : **2,8 → 3,17 it/s** (+13 %), soit ~8,6 h pour 200 M tokens.
