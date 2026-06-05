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


def get_peak_layers(cfg: "dict | None" = None) -> dict:
    """Per-behaviour peak (PR-trough) layers from layer triangulation. Previously
    hardcoded + duplicated in build_phase6.py and robustness_geometry.py."""
    cfg = cfg or load_config()
    return dict((cfg.get("analysis") or {}).get("peak_layers", {}))


def provenance(args=None, inputs=None) -> dict:
    """Provenance stamp to embed in result files so any figure/JSON can be traced
    to the exact code + config + inputs that produced it (AUDIT.md §5).

    Captures the git commit (+ dirty flag), the config seed, the runner args,
    and optional input-file SHA-256s. Best-effort: never raises. Deliberately
    omits a timestamp so the stamp is reproducible (stamp the time at the call
    site if needed)."""
    import hashlib
    import subprocess

    def _git(*a):
        try:
            return subprocess.check_output(
                ["git", *a], cwd=str(CONFIG_PATH.parent.parent),
                stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None

    stamp: dict = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "seed": SEED,
    }
    if args is not None:
        raw = vars(args) if hasattr(args, "__dict__") else dict(args)
        stamp["args"] = {
            k: (v if isinstance(v, (int, float, bool, str, type(None))) else str(v))
            for k, v in raw.items()
        }
    if inputs:
        digests = {}
        for p in inputs:
            try:
                with open(p, "rb") as f:
                    digests[str(p)] = hashlib.sha256(f.read()).hexdigest()[:16]
            except Exception:
                digests[str(p)] = None
        stamp["input_sha256"] = digests
    return stamp


# Module-level conveniences (computed once from the YAML).
_CFG = load_config()
MODELS = _model_registry(_CFG)
STEERING_LAYERS = get_steering_layers(_CFG)
SEED = get_seed(_CFG)
TARGET_BEHAVIOURS = get_target_behaviours(_CFG)
PEAK_LAYERS = get_peak_layers(_CFG)

__all__ = [
    "load_config", "get_steering_layers", "get_seed", "get_target_behaviours",
    "get_peak_layers", "provenance",
    "CONFIG_PATH", "MODELS", "STEERING_LAYERS", "SEED", "TARGET_BEHAVIOURS",
    "PEAK_LAYERS",
]
