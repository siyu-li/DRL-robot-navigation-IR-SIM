"""
Switcher Scalar Configuration Loader.

Loads ``switcher_config.yaml`` and exposes a ``SwitcherScalarConfig`` dataclass
with computed ``group_scalar_dim`` and ``state_scalar_dim`` properties.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

_VALID_AGGS = {"mean", "min", "max", "sum", "first"}

# Default config path (next to this file)
_DEFAULT_CONFIG = Path(__file__).resolve().parent / "switcher_config.yaml"


@dataclass
class SwitcherScalarConfig:
    """Unified scalar configuration for supervised and RL feature builders.

    Attributes:
        coupling_mode: ``"min"`` or ``"mean"`` for group velocity coupling.
        base_scalars: Whether to include the 4 base scalars
            (size_feat, A_in, A_out, A_obs).
        extra_group: Per-group extra features as ``[(key, agg), ...]``.
        extra_global: Global broadcast scalars as ``[key, ...]``.
        state_scalars: State-level scalars for RL critic as
            ``[(key, agg), ...]``.
    """

    coupling_mode: str = "min"
    pooling: str = "mean"  # "mean", "max", or "attention"
    # Attention pooling hyper-parameters (used when pooling == "attention")
    attn_n_heads: int = 2
    attn_score_hidden: List[int] = field(default_factory=lambda: [128, 64])
    attn_dropout: float = 0.0
    base_scalars: bool = True
    extra_group: List[Tuple[str, str]] = field(default_factory=list)
    extra_global: List[str] = field(default_factory=list)
    state_scalars: List[Tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.pooling not in ("mean", "max", "attention"):
            raise ValueError(
                f"pooling must be 'mean', 'max', or 'attention', got '{self.pooling}'"
            )
        if self.coupling_mode not in ("min", "mean"):
            raise ValueError(
                f"coupling_mode must be 'min' or 'mean', got '{self.coupling_mode}'"
            )
        for key, agg in self.extra_group:
            if agg not in _VALID_AGGS:
                raise ValueError(
                    f"Invalid aggregation '{agg}' for extra_group key '{key}'. "
                    f"Must be one of {_VALID_AGGS}."
                )
        for key, agg in self.state_scalars:
            if agg not in _VALID_AGGS:
                raise ValueError(
                    f"Invalid aggregation '{agg}' for state_scalars key '{key}'. "
                    f"Must be one of {_VALID_AGGS}."
                )

    # ── Computed dimensions ────────────────────────────────────────────

    @property
    def base_scalar_dim(self) -> int:
        """Number of base scalars (4 if enabled, 0 otherwise)."""
        return 4 if self.base_scalars else 0

    @property
    def group_scalar_dim(self) -> int:
        """Total group scalar dimension: base + extra_group + extra_global."""
        return self.base_scalar_dim + len(self.extra_group) + len(self.extra_global)

    @property
    def state_scalar_dim(self) -> int:
        """Number of state-level scalars for the RL critic."""
        return len(self.state_scalars)

    # ── Serialisation helpers ──────────────────────────────────────────

    def to_dict(self) -> Dict:
        """Return a plain dict suitable for YAML / checkpoint serialisation."""
        return {
            "coupling_mode": self.coupling_mode,
            "pooling": self.pooling,
            "attn_n_heads": self.attn_n_heads,
            "attn_score_hidden": list(self.attn_score_hidden),
            "attn_dropout": self.attn_dropout,
            "base_scalars": self.base_scalars,
            "extra_group": [list(pair) for pair in self.extra_group],
            "extra_global": list(self.extra_global),
            "state_scalars": [list(pair) for pair in self.state_scalars],
        }


def load_switcher_config(path: str | Path | None = None) -> SwitcherScalarConfig:
    """Load a ``SwitcherScalarConfig`` from a YAML file.

    Args:
        path: Path to YAML config.  ``None`` → default ``switcher_config.yaml``
            next to this module.

    Returns:
        Parsed ``SwitcherScalarConfig``.
    """
    if path is None:
        path = _DEFAULT_CONFIG
    path = Path(path)

    with open(path) as f:
        raw = yaml.safe_load(f)

    extra_group = [tuple(pair) for pair in (raw.get("extra_group") or [])]
    extra_global = list(raw.get("extra_global") or [])
    state_scalars = [tuple(pair) for pair in (raw.get("state_scalars") or [])]

    attn_cfg = raw.get("attention_pooling") or {}
    return SwitcherScalarConfig(
        coupling_mode=raw.get("coupling_mode", "min"),
        pooling=raw.get("pooling", "mean"),
        attn_n_heads=int(attn_cfg.get("n_heads", 2)),
        attn_score_hidden=list(attn_cfg.get("score_hidden", [128, 64])),
        attn_dropout=float(attn_cfg.get("dropout", 0.0)),
        base_scalars=raw.get("base_scalars", True),
        extra_group=extra_group,
        extra_global=extra_global,
        state_scalars=state_scalars,
    )


def config_from_dict(d: Dict) -> SwitcherScalarConfig:
    """Reconstruct a ``SwitcherScalarConfig`` from a plain dict (e.g. checkpoint)."""
    extra_group = [tuple(pair) for pair in (d.get("extra_group") or [])]
    extra_global = list(d.get("extra_global") or [])
    state_scalars = [tuple(pair) for pair in (d.get("state_scalars") or [])]

    return SwitcherScalarConfig(
        coupling_mode=d.get("coupling_mode", "min"),
        pooling=d.get("pooling", "mean"),
        attn_n_heads=int(d.get("attn_n_heads", 2)),
        attn_score_hidden=list(d.get("attn_score_hidden", [128, 64])),
        attn_dropout=float(d.get("attn_dropout", 0.0)),
        base_scalars=d.get("base_scalars", True),
        extra_group=extra_group,
        extra_global=extra_global,
        state_scalars=state_scalars,
    )


def build_attn_pool(cfg: SwitcherScalarConfig, embed_dim: int = 512):
    """Create an ``AttentionGroupPooling`` from a ``SwitcherScalarConfig``.

    Returns *None* when ``cfg.pooling != 'attention'``.

    Args:
        cfg: Loaded switcher config.
        embed_dim: Robot embedding dimension.

    Returns:
        ``AttentionGroupPooling`` instance, or ``None``.
    """
    if cfg.pooling != "attention":
        return None
    from robot_nav.models.MARL.switcher.attention_pooling import AttentionGroupPooling
    return AttentionGroupPooling(
        embed_dim=embed_dim,
        n_heads=cfg.attn_n_heads,
        score_hidden=tuple(cfg.attn_score_hidden),
        dropout=cfg.attn_dropout,
    )
