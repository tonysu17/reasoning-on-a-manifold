"""Single source of truth for experiment configuration.

Loads ``configs/config.yaml`` and exposes the parameters that were previously
hardcoded — and had drifted out of sync — across the phase runners: the model
registry, the per-model steering layer, the global seed, and the target
behaviours.

Background: historically *nothing* loaded ``config.yaml``; every runner
re-declared its own ``MODELS`` / ``STEERING_LAYERS`` dicts. Those copies
diverged (e.g. ``05b_geometric_diagnostics.py`` omitted the baseline model, so
it silently fell back to layer 27 via ``.get(..., 27)`` — correct only by
coincidence). Import from here instead of re-declaring, so editing the YAML
actually changes behaviour and the registry can never disagree with itself.

    from src.config import STEERING_LAYERS, MODELS, SEED, TARGET_BEHAVIOURS
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"


@lru_cache(maxsize=8)
def load_config(path: "str | Path | None" = None) -> dict:
    """Parse and cache configs/config.yaml (or an explicit path)."""
    p = Path(path) if path is not None else CONFIG_PATH
    with open(p) as f:
        return yaml.safe_load(f)


def _model_registry(cfg: dict) -> dict:
    """short_name -> full model spec (with its config role attached)."""
    reg: dict[str, dict] = {}
    for role, spec in (cfg.get("models") or {}).items():
        if not isinstance(spec, dict) or "short_name" not in spec:
            continue
        reg[spec["short_name"]] = {"role": role, **spec}
    return reg


def get_steering_layers(cfg: "dict | None" = None) -> dict:
    cfg = cfg or load_config()
    return {sn: int(spec.get("steering_layer", 27))
            for sn, spec in _model_registry(cfg).items()}


def get_seed(cfg: "dict | None" = None) -> int:
    cfg = cfg or load_config()
    return int((cfg.get("project") or {}).get("seed", 42))


def get_target_behaviours(cfg: "dict | None" = None) -> list:
    cfg = cfg or load_config()
    return list((cfg.get("annotation") or {}).get(
        "target_behaviours",
        ["backtracking", "uncertainty-estimation", "example-testing", "adding-knowledge"],
    ))


# Module-level conveniences (computed once from the YAML).
_CFG = load_config()
MODELS = _model_registry(_CFG)
STEERING_LAYERS = get_steering_layers(_CFG)
SEED = get_seed(_CFG)
TARGET_BEHAVIOURS = get_target_behaviours(_CFG)

__all__ = [
    "load_config", "get_steering_layers", "get_seed", "get_target_behaviours",
    "CONFIG_PATH", "MODELS", "STEERING_LAYERS", "SEED", "TARGET_BEHAVIOURS",
]
