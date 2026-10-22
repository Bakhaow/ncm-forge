"""GPU cost estimation.

Training FLOPs follow the standard 6*N*D rule (fwd + bwd) with N = non
embedding parameters, which is what actually drives the compute: the embedding
tables and the output head are memory-bound lookups, not matmuls.

Parameter counts are exact when torch is importable (the models are
instantiated and counted); otherwise an analytic fallback is used.

Numbers are estimates: they ignore dataloader stalls, evaluation passes, and
the fact that a rented GPU is shared.
"""

from forge.config import generator_sequence_length

GPU_PRESETS = {
    "rtx4070": {"tflops_bf16": 165.0, "usd_per_hour": 0.20, "vram_gib": 12},
    "rtx4090": {"tflops_bf16": 330.0, "usd_per_hour": 0.40, "vram_gib": 24},
    "a100-80": {"tflops_bf16": 312.0, "usd_per_hour": 1.30, "vram_gib": 80},
    "h100": {"tflops_bf16": 989.0, "usd_per_hour": 2.80, "vram_gib": 80},
    "l40s": {"tflops_bf16": 362.0, "usd_per_hour": 0.90, "vram_gib": 48},
}

# The output head is a real matmul (embed_dim x vocab) so it stays in the
# compute; the codebooks and the input embedding are lookups.
EXCLUDE_FROM_FLOPS = ("wte", "wle", "wpe", "embeddings")


def _count(module):
    total = sum(p.numel() for p in module.parameters())
    non_emb = sum(
        p.numel()
        for name, p in module.named_parameters()
        if not any(name.endswith(s) or f".{s}" in name for s in EXCLUDE_FROM_FLOPS)
    )
    return total, non_emb


def exact_counts(cfg):
    """Instantiate both stages and count their parameters."""
    from forge.vendor import require

    require()
    from models.next_concept import NextConceptTransformer
    from models.vqvae import VQVAE

    vq_cfg = dict(cfg.tokenizer.vqvae)
    vq_cfg.pop("checkpoint_path", None)
    vq = VQVAE(**vq_cfg)
    tok_total, tok_non_emb = _count(vq)

    net = NextConceptTransformer(vqvae_local=vq, config=cfg.model)
    gen_total, gen_non_emb = _count(net)
    return {
        "estimated": False,
        "tokenizer_total": tok_total,
        "tokenizer_non_emb": tok_non_emb,
        "generator_total": gen_total,
        "generator_non_emb": gen_non_emb,
    }


def analytic_counts(cfg) -> dict:
    """Fallback when torch is unavailable."""
    n_layer = int(cfg.model.n_layer)
    n_embd = int(cfg.model.n_embd)
    vq_dim = int(cfg.tokenizer.vqvae.embed_dim)
    in_vocab = int(cfg.tokenizer.vqvae.in_vocab_size)
    vocab_sizes = list(cfg.model.vocab_size)
    codebooks = list(cfg.tokenizer.vqvae.vqvae_vocab_size)

    gen_total = n_layer * 12 * n_embd * n_embd + sum(n_embd * v for v in vocab_sizes)
    gen_non_emb = gen_total - n_embd * vq_dim
    tok_total = (
        6 * 12 * vq_dim * vq_dim
        + sum(vq_dim * v for v in codebooks)
        + in_vocab * vq_dim          # input embedding (lookup)
        + vq_dim * in_vocab          # output head (matmul)
    )
    return {
        "estimated": True,
        "tokenizer_total": tok_total,
        "tokenizer_non_emb": tok_total - in_vocab * vq_dim - sum(vq_dim * v for v in codebooks),
        "generator_total": gen_total,
        "generator_non_emb": gen_non_emb,
    }


def counts(cfg) -> dict:
    try:
        return exact_counts(cfg)
    except Exception as exc:  # torch missing or config incomplete
        print(f"[estimate] exact count unavailable ({type(exc).__name__}), using the analytic fallback")
        return analytic_counts(cfg)


def estimate(
    cfg,
    gpu: str = "rtx4070",
    mfu: float = 0.35,
    train_tokens: int | None = None,
    tokenizer_tokens: int | None = None,
) -> dict:
    preset = GPU_PRESETS[gpu]
    eff_tflops = preset["tflops_bf16"] * mfu
    c = counts(cfg)

    target_len = int(cfg.data.get("target_len", cfg.tokenizer.vqvae.context_length))
    # The generator processes prefix_len + sum(levels[1:]) positions per sample,
    # not target_len: the hierarchy has one position per code.
    seq_len = generator_sequence_length(cfg)
    gen_tokens = train_tokens
    if gen_tokens is None:
        eff_batch = int(cfg.train.batch_size) * int(cfg.train.get("grad_accum", 1))
        gen_tokens = int(cfg.train.steps) * eff_batch * seq_len
    tok_tokens = tokenizer_tokens
    if tok_tokens is None:
        # the tokenizer stage reads the same train.steps / batch_size / grad_accum
        tok_tokens = (
            int(cfg.train.get("tokenizer_steps", cfg.train.steps))
            * int(cfg.train.get("tokenizer_batch_size", cfg.train.batch_size))
            * int(cfg.train.get("tokenizer_grad_accum", cfg.train.get("grad_accum", 1)))
            * target_len
        )

    gen_flops = 6 * c["generator_non_emb"] * gen_tokens
    tok_flops = 6 * c["tokenizer_non_emb"] * tok_tokens
    gen_hours = gen_flops / (eff_tflops * 1e12) / 3600
    tok_hours = tok_flops / (eff_tflops * 1e12) / 3600

    # Measured throughput beats any FLOP model: small batches run far below
    # peak, and the 151680-way tokenizer head is not in the non-embedding count.
    measured = False
    if cfg.train.get("measured_gen_it_s"):
        gen_hours = int(cfg.train.steps) / float(cfg.train.measured_gen_it_s) / 3600
        measured = True
    if cfg.train.get("measured_tok_it_s"):
        tok_hours = (
            int(cfg.train.get("tokenizer_steps", cfg.train.steps))
            / float(cfg.train.measured_tok_it_s)
            / 3600
        )
        measured = True

    return {
        "gpu": gpu,
        "estimated_params": bool(c.get("estimated", False)),
        "effective_tflops": round(eff_tflops, 1),
        "generator_params_M": round(c["generator_total"] / 1e6, 1),
        "generator_compute_M": round(c["generator_non_emb"] / 1e6, 1),
        "tokenizer_params_M": round(c["tokenizer_total"] / 1e6, 1),
        "tokenizer_compute_M": round(c["tokenizer_non_emb"] / 1e6, 1),
        "generator_train_tokens_B": round(gen_tokens / 1e9, 2),
        "tokenizer_train_tokens_B": round(tok_tokens / 1e9, 2),
        "generator_hours": round(gen_hours, 1),
        "tokenizer_hours": round(tok_hours, 1),
        "total_hours": round(gen_hours + tok_hours, 1),
        "total_usd": round((gen_hours + tok_hours) * preset["usd_per_hour"], 2),
        "vram_gib": preset["vram_gib"],
        "measured_throughput": measured,
    }


def format_estimate(est: dict) -> str:
    return "\n".join(
        [
            f"GPU                 {est['gpu']}  ({est['effective_tflops']} TFLOPS effective, {est['vram_gib']} GiB)"
            + ("  [param counts estimated]" if est.get("estimated_params") else "")
            + ("  [throughput measured]" if est.get("measured_throughput") else ""),
            f"generator           {est['generator_params_M']} M params "
            f"({est['generator_compute_M']} M counted for FLOPs), {est['generator_train_tokens_B']} B tokens",
            f"tokenizer           {est['tokenizer_params_M']} M params "
            f"({est['tokenizer_compute_M']} M counted for FLOPs), {est['tokenizer_train_tokens_B']} B tokens",
            f"wall clock          {est['generator_hours']} h generator + {est['tokenizer_hours']} h tokenizer "
            f"= {est['total_hours']} h",
            f"budget              ~{est['total_usd']} USD at {GPU_PRESETS[est['gpu']]['usd_per_hour']} $/h",
        ]
    )
