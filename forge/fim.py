"""Fill-in-the-middle (FIM) support.

The upstream generator is a prefix-LM: the conditioning block sits before the
target and the target attends to it. FIM (prefix, suffix -> middle) therefore
needs no mask surgery: the conditioning block simply becomes

    [suffix ; prefix]

with `model.prefix_len = suffix_len + prefix_len`. The suffix positions cannot
see the target (top-right block of the upstream mask stays -inf) and the target
sees the whole conditioning block, which is exactly what FIM requires.

The ordering matters for one thing: the level-0 head reads the *last* positions
of the conditioning block, so putting the suffix first makes the coarsest
level (the global shape of the completion) depend on the code that sits right
before the cursor.
"""

import torch
from omegaconf import OmegaConf


def make_fim_config(cfg, prefix_len: int, suffix_len: int | None = None):
    """Return a copy of `cfg` configured for FIM training."""
    out = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    suffix_len = suffix_len if suffix_len is not None else prefix_len
    out.data.task = "fim"
    out.data.prefix_len = prefix_len
    out.data.suffix_len = suffix_len
    out.data.cond_order = "suffix_first"
    out.model.prefix_len = prefix_len + suffix_len
    return out


@torch.no_grad()
def generate_fim(
    net,
    tk,
    prefix_ids: list[int],
    suffix_ids: list[int],
    prefix_len: int,
    suffix_len: int,
    temperature: float = 1.0,
    top_k: int = 1,
    top_p: float = 0.0,
    cfg_scale: float = 0.0,
):
    """Generate a completion for a cursor position.

    Returns (levels, final_text) where `levels` is the list of progressively
    refined decodings, one per hierarchy level.
    """
    device = next(net.parameters()).device
    pad_id = tk.pre_tokenizer.pad_token_id
    eos_id = tk.pre_tokenizer.eos_token_id
    cfg_pad = getattr(net.config, "prefix_pad_token_id", None)
    if cfg_pad is not None and int(cfg_pad) != int(pad_id):
        raise ValueError(
            f"model.prefix_pad_token_id ({cfg_pad}) must equal the tokenizer pad id "
            f"({pad_id}): upstream masks the conditioning block by comparing token ids"
        )

    def fit(ids, width, keep="head"):
        ids = list(ids[-width:] if keep == "tail" else ids[:width])
        pad = width - len(ids)
        if pad > 0:
            ids = ids + [pad_id] * pad
        return ids

    cond = fit(suffix_ids, suffix_len) + fit(prefix_ids, prefix_len, keep="tail")
    x = torch.tensor([cond], dtype=torch.long, device=device)

    l0 = None
    if not bool(getattr(net.config, "predict_level0", False)):
        vq = tk.vqvae.vq
        l0 = torch.randint(0, vq.vqvae_vocab_sizes[0], (1, vq.levels[0]), device=device)

    gen = net.generate(
        l0, x, temperature=temperature, top_k=top_k, top_p=top_p,
        cfg_scale=cfg_scale, cfg_null_eos=eos_id,
    )
    levels = tk.decode_multiscales(gen)[0]
    return levels, levels[-1]
