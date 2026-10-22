"""Config loading for ncm-forge.

A config is an OmegaConf node mirroring the upstream layout:

    tokenizer:      same block as upstream config/eval/*.yaml (tokenizer section)
    model:          same block as upstream config/eval/*.yaml (model section)
    data:           corpus description (forge-specific)
    train:          optimization settings (forge-specific)

Upstream blocks are copied verbatim so that a checkpoint trained here is a
drop-in for the upstream evaluate.py.
"""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"


def load(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Load a YAML config, optionally applying `key=value` overrides."""
    path = Path(path)
    if not path.exists():
        candidate = CONFIGS / path
        if candidate.exists():
            path = candidate
        else:
            raise FileNotFoundError(f"config not found: {path}")
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def save(cfg: DictConfig, path: str | Path) -> None:
    """Write a config back to disk (used to freeze the resolved run config)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(OmegaConf.to_yaml(cfg))


def levels_of(cfg: DictConfig) -> list[int]:
    """Number of codes per level, read from the tokenizer's quantizer config."""
    return list(cfg.tokenizer.vqvae.vector_quantizer_config.levels.manual_levels)


def vocab_sizes_of(cfg: DictConfig) -> list[int]:
    """Codebook size per level."""
    return list(cfg.tokenizer.vqvae.vqvae_vocab_size)


def model_levels(cfg: DictConfig) -> list[int]:
    """Levels the generator actually predicts (level 0 excluded when queried)."""
    levels = levels_of(cfg)
    if bool(cfg.model.get("query_level0", False)):
        return levels
    return levels[1:]


def model_vocab_sizes(cfg: DictConfig) -> list[int]:
    """Per-level head sizes, aligned with `model_levels`."""
    sizes = vocab_sizes_of(cfg)
    if bool(cfg.model.get("query_level0", False)):
        return sizes
    return sizes[1:]


def model_head_levels(cfg: DictConfig) -> list[int]:
    """The blocks `_run_stack` actually emits, in order.

    With predict_level0 the level-0 head output comes first, then one block per
    level in levels[lvl_first:].
    """
    heads = list(model_levels(cfg))
    if bool(cfg.model.get("predict_level0", False)):
        heads = [levels_of(cfg)[0]] + heads
    return heads


def generator_sequence_length(cfg: DictConfig) -> int:
    """Positions the generator processes per sample.

    The conditioning block (prefix_len) plus the NCM input, which carries one
    position per code of levels[1:].
    """
    levels = levels_of(cfg)
    return int(cfg.model.get("prefix_len", 1)) + sum(levels[1:])
