"""Recipe-analysis core for S4 — pure, file-free, testable.

``analyze_recipes`` takes in-memory per-model activation matrices and produces the
post-training fingerprint comparison: per-recipe separation sharpness (H2),
same-ambient refusal-direction cosine (H4 output-convergence), and paired linear
CKA across architectures of different width (H4, gpt-oss-2880 vs R1-1536). The
``14_safety_geometry.py`` runner does the file IO and calls this.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from src.safety.refusal_direction import (
    refusal_direction, separation, recipe_direction_cosine, linear_cka,
)


def analyze_recipes(per_model: dict, *, effort: Optional[dict] = None) -> dict:
    """Compare the safety geometry across post-training recipes.

    Parameters
    ----------
    per_model : {model_name: {"hidden_dim": int,
                              "layers": {layer:int -> {"harmful": (N,d), "harmless": (M,d)}}}}
        The harmful/harmless rows MUST be in the same stimulus order across models
        for CKA to be meaningful (same prompts, per-model activations).
    effort : optional {model_name: {effort_level: (N,d)}}
        gpt-oss reasoning-effort activations at that model's best layer, projected
        onto its own refusal direction (the within-model H2 knob).

    Returns
    -------
    dict with: fingerprint, best_layer, cosine, cka, effort_engagement.
    """
    fingerprint: dict = {}
    directions: dict = {}
    for name, m in per_model.items():
        fingerprint[name] = {}
        directions[name] = {}
        for layer, acts in m["layers"].items():
            fingerprint[name][layer] = separation(acts["harmful"], acts["harmless"])
            directions[name][layer] = refusal_direction(acts["harmful"], acts["harmless"])

    best_layer = {
        name: (max(f, key=lambda L: f[L]["cohens_d"]) if f else None)
        for name, f in fingerprint.items()
    }

    names = list(per_model)
    cosine: dict = {}
    cka: dict = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            da = per_model[a]["hidden_dim"]
            db = per_model[b]["hidden_dim"]
            common = sorted(set(directions[a]) & set(directions[b]))
            if da == db:
                cosine[f"{a}|{b}"] = {
                    L: recipe_direction_cosine(directions[a][L], directions[b][L])
                    for L in common
                }
            for L in common:
                Ha = per_model[a]["layers"][L]["harmful"]
                Hb = per_model[b]["layers"][L]["harmful"]
                if Ha.shape[0] == Hb.shape[0]:
                    cka.setdefault(f"{a}|{b}", {})[L] = linear_cka(Ha, Hb)

    engagement: dict = {}
    if effort:
        for name, by_eff in effort.items():
            L = best_layer.get(name)
            if L is None or name not in directions:
                continue
            r = directions[name][L]
            engagement[name] = {
                eff: float(np.asarray(acts, dtype=float) @ (r / (np.linalg.norm(r) + 1e-12))).mean()
                for eff, acts in by_eff.items()
            }

    return {
        "fingerprint": fingerprint,
        "best_layer": best_layer,
        "cosine": cosine,
        "cka": cka,
        "effort_engagement": engagement,
    }


def summarise(result: dict) -> list[dict]:
    """Flatten ``analyze_recipes`` output to a per-model headline row (best layer)."""
    rows = []
    for name, layer in result["best_layer"].items():
        if layer is None:
            continue
        sep = result["fingerprint"][name][layer]
        rows.append({
            "model": name,
            "best_layer": layer,
            "cohens_d": round(sep["cohens_d"], 4),
            "auroc": round(sep["auroc"], 4),
            "dom_norm": round(sep["dom_norm"], 4),
            "n_harmful": sep["n_harmful"],
            "n_harmless": sep["n_harmless"],
        })
    rows.sort(key=lambda r: r["cohens_d"], reverse=True)
    return rows


# ── File IO (run-time; consumes a safety extraction pass) ─────────────────────

def load_safety_activations(model_dir, layers: Optional[list] = None) -> dict:
    """Load ``harmful_layer{L}.npy`` / ``harmless_layer{L}.npy`` from *model_dir*
    → {L: {"harmful": (N,d), "harmless": (M,d)}}. If *layers* is None, infer the
    available layers from the harmful files. Skips layers missing either side."""
    model_dir = Path(model_dir)
    if layers is None:
        layers = sorted(
            int(p.stem.split("layer")[-1])
            for p in model_dir.glob("harmful_layer*.npy")
            if p.stem.split("layer")[-1].isdigit()
        )
    out: dict = {}
    for L in layers:
        hp = model_dir / f"harmful_layer{L}.npy"
        lp = model_dir / f"harmless_layer{L}.npy"
        if hp.exists() and lp.exists():
            out[int(L)] = {"harmful": np.load(hp), "harmless": np.load(lp)}
    return out


def build_per_model(model_specs: dict, activations_root,
                   layers: Optional[list] = None) -> dict:
    """Assemble the ``analyze_recipes`` input from on-disk activations.

    model_specs: {name: {"short_name": str, "hidden_dim": int}}. Models with no
    safety activations on disk are skipped (build-now-run-later)."""
    root = Path(activations_root)
    per_model: dict = {}
    for name, spec in model_specs.items():
        layer_acts = load_safety_activations(root / spec["short_name"], layers)
        if not layer_acts:
            continue
        d = int(next(iter(layer_acts.values()))["harmful"].shape[1])
        per_model[name] = {"hidden_dim": d, "layers": layer_acts}
    return per_model


__all__ = ["analyze_recipes", "summarise", "load_safety_activations", "build_per_model"]
