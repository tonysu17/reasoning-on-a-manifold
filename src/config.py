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


def _models_by_cli(cfg: dict) -> dict:
    """cli_alias (the --model value) -> full model spec. Single source for the
    per-runner MODELS dicts in 02/04/07, which previously hand-maintained
    divergent copies keyed by short codes like '1.5b'."""
    out: dict[str, dict] = {}
    for spec in _model_registry(cfg).values():
        alias = spec.get("cli_alias")
        if alias:
            out[alias] = spec
    return out


def model_tuple(alias: str, cfg: "dict | None" = None) -> tuple:
    """(id, short_name, dtype) for a runner --model alias (the 04/07 shape)."""
    cfg = cfg or load_config()
    m = _models_by_cli(cfg)[alias]
    return (m["id"], m["short_name"], m["dtype"])


def model_dict(alias: str, cfg: "dict | None" = None) -> dict:
    """{id, short, dtype} for a runner --model alias (the 02 shape)."""
    m_id, short, dtype = model_tuple(alias, cfg)
    return {"id": m_id, "short": short, "dtype": dtype}


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


def require_file(path, hint: str = ""):
    """Exit with a clear one-line message instead of a raw traceback when an
    expected input (file or directory) is missing — for the ad-hoc analysis
    scripts that hardcode their inputs (AUDIT.md §5). Returns the Path."""
    from pathlib import Path as _P
    import sys as _sys
    p = _P(path)
    if not p.exists():
        msg = f"ERROR: required input not found: {p}"
        if hint:
            msg += f"\n  hint: {hint}"
        _sys.exit(msg)
    return p


def backup_existing(path) -> "Path | None":
    """Copy an existing file to ``<name>.bak`` before a run overwrites it, so a
    shorter or failed re-run can't silently destroy a good prior artifact
    (AUDIT.md §5). Copy (not move) so resume logic can still read the original.
    Best-effort; returns the .bak path or None."""
    from pathlib import Path as _P
    import shutil
    p = _P(path)
    if not p.exists():
        return None
    bak = p.with_suffix(p.suffix + ".bak")
    try:
        shutil.copy2(p, bak)
        return bak
    except OSError:
        return None


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
MODELS_BY_CLI = _models_by_cli(_CFG)

__all__ = [
    "load_config", "get_steering_layers", "get_seed", "get_target_behaviours",
    "get_peak_layers", "provenance", "backup_existing", "require_file",
    "model_tuple", "model_dict",
    "CONFIG_PATH", "MODELS", "STEERING_LAYERS", "SEED", "TARGET_BEHAVIOURS",
    "PEAK_LAYERS", "MODELS_BY_CLI",
]
