#!/usr/bin/env python3
"""Phase-7 Δ_floor headline analysis runner (E8 analysis step).

Loads the records ``07_evaluate_steering.py`` writes and computes the
de-confounded suppression headline (``src.delta_floor``): per (behaviour, arm),
Δ_floor = suppression_arm − matched-floor suppression at the SEALED per-behaviour
α*, paired-BCa over tasks, Holm-corrected across the (behaviour×arm) family, with
McNemar-exact + sign-test corroborators and the fraction-RMS acceptance band.

This runner does NO generation/annotation/judging — pure arithmetic over the
saved JSON. The pass/fail gate is only meaningful with a non-builder annotator
(``--band-annotated``); without it every cell is "preliminary, band-ungated".

Usage:
    python 08_steering_analysis.py --eval-dir results/eval/R1-1.5B__L27
    python 08_steering_analysis.py --eval-dir results/eval/R1-1.5B__venhoff \
        --alpha-star-json results/saturation_predictions/R1-1.5B/predictions_layer18.json \
        --arms single_direction manifold_k3 manifold_k5 \
        --band-annotated results/eval/R1-1.5B__L27/annotated_steered_qwen3.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, ".")
from src.delta_floor import delta_floor_headline, floor_for_arm
from src.evaluation import behaviour_fraction

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("08_steering_analysis")

DEFAULT_ARMS = ["single_direction", "manifold_k3", "manifold_k5"]


def _load_json(path: Path):
    if not path.exists():
        raise SystemExit(f"missing required file: {path}")
    with open(path) as f:
        return json.load(f)


def load_alpha_star(path: Path) -> dict:
    """``{behaviour -> alpha_star_pred}`` from a saturation predictions JSON."""
    d = _load_json(path)
    preds = d.get("predictions", {})
    out = {b: float(v["alpha_star_pred"]) for b, v in preds.items()
           if isinstance(v, dict) and "alpha_star_pred" in v}
    if not out:
        raise SystemExit(f"no alpha_star_pred values in {path}")
    log.info(f"sealed alpha* (layer {d.get('layer')}): "
             + ", ".join(f"{b}={a:.3f}" for b, a in out.items()))
    return out


def parse_alpha_star_arg(spec: str) -> dict:
    """Parse ``beh=val,beh=val`` into ``{behaviour -> alpha}``."""
    out = {}
    for part in spec.split(","):
        k, _, v = part.partition("=")
        out[k.strip()] = float(v)
    return out


def cross_annotator_band(annotated_a: list, annotated_b: list,
                         behaviours: list) -> dict:
    """``band_b`` per behaviour = fraction-RMS of |frac_a − frac_b| over the SAME
    re-annotated chains scored by two annotators (the acceptance band). Skips
    chains either annotator left missing/empty."""
    def index(recs):
        return {(r["task_id"], r["behaviour"], r["method"], r["alpha"]):
                r.get("annotations", []) for r in recs}
    ia, ib = index(annotated_a), index(annotated_b)
    shared = set(ia) & set(ib)
    bands = {}
    for beh in behaviours:
        diffs = []
        for key in shared:
            aa, bb = ia[key], ib[key]
            if not aa or not bb:
                continue
            diffs.append(abs(behaviour_fraction(aa, beh) - behaviour_fraction(bb, beh)))
        if diffs:
            bands[beh] = float(np.sqrt(np.mean(np.square(diffs))))
            log.info(f"band_b[{beh}] = {bands[beh]:.4f} (RMS over {len(diffs)} paired chains)")
    return bands


def print_table(out: dict) -> None:
    print("\n" + "=" * 118)
    print("Δ_floor HEADLINE  (suppression_arm − matched-floor suppression at sealed α*; "
          "paired-BCa, Holm across the family)")
    print("=" * 118)
    hdr = (f"  {'behaviour':22s} {'arm':14s} {'floor':22s} {'α*':>5s} {'N':>3s} "
           f"{'Δ_floor':>8s} {'95% CI':>17s} {'holm_p':>7s} {'band':>6s} "
           f"{'McN_p':>6s} {'sign_p':>6s} {'verdict':>11s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for key, c in out["cells"].items():
        b = c.get("bootstrap")
        ci = f"[{b['ci_low']:+.3f},{b['ci_high']:+.3f}]" if b else "—"
        df = f"{c['delta_floor']:+.3f}" if c["delta_floor"] is not None else "—"
        holm = f"{c['holm_p']:.3f}" if c["holm_p"] is not None else "—"
        band = f"{c['band_b']:.3f}" if c["band_b"] is not None else "—"
        mcn = f"{c['mcnemar']['p_value']:.3f}" if c.get("mcnemar") else "—"
        sgn = f"{c['sign']['p_value']:.3f}" if c.get("sign") else "—"
        verdict = ("PASS" if c["passes"] is True else
                   "fail" if c["passes"] is False else
                   ("prelim" if b else c["status"]))
        print(f"  {c['behaviour']:22s} {c['arm']:14s} {c['floor']:22s} "
              f"{c['alpha']:>5.2f} {c['n_tasks']:>3d} {df:>8s} {ci:>17s} "
              f"{holm:>7s} {band:>6s} {mcn:>6s} {sgn:>6s} {verdict:>11s}")
    print(f"\n  PASS={out['n_pass']}  preliminary(band-ungated)={out['n_preliminary']}")
    if out["n_pass"] == 0 and out["n_preliminary"] > 0:
        print("  NOTE: band-ungated — supply --band-annotated (non-builder annotator) "
              "for a pass-grade verdict.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True,
                    help="dir with steering_results.json + annotated_steered.json")
    ap.add_argument("--alpha-star-json",
                    default="results/saturation_predictions/R1-1.5B/predictions_layer27.json")
    ap.add_argument("--alpha-star", default=None,
                    help="explicit override: 'backtracking=0.99,uncertainty-estimation=0.97,...'")
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    ap.add_argument("--behaviours", nargs="+", default=None)
    ap.add_argument("--band-json", default=None, help="JSON {behaviour: band_b}")
    ap.add_argument("--band-annotated", default=None,
                    help="a second annotated_steered.json (non-builder annotator) → "
                         "compute the cross-annotator fraction-RMS band")
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--n-resamples", type=int, default=10000,
                    help="bootstrap resamples (analysis is model-free/cheap; high B "
                         "keeps Holm thresholds out of the 1/B resolution floor)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="report path (default <eval-dir>/delta_floor_report.json)")
    a = ap.parse_args()

    eval_dir = Path(a.eval_dir)
    steered = _load_json(eval_dir / "steering_results.json")
    annotated = _load_json(eval_dir / "annotated_steered.json")
    log.info(f"loaded {len(steered)} generation records, {len(annotated)} annotated")

    for arm in a.arms:
        if floor_for_arm(arm) is None:
            raise SystemExit(f"--arms: {arm!r} has no matched floor (not a headline arm)")

    alpha_star = (parse_alpha_star_arg(a.alpha_star) if a.alpha_star
                  else load_alpha_star(Path(a.alpha_star_json)))
    behaviours = a.behaviours or list(alpha_star.keys())

    band_b = None
    if a.band_annotated:
        band_b = cross_annotator_band(annotated, _load_json(Path(a.band_annotated)), behaviours)
    elif a.band_json:
        band_b = {k: float(v) for k, v in _load_json(Path(a.band_json)).items()}
    else:
        log.warning("no band supplied (--band-annotated / --band-json) → "
                    "verdicts are PRELIMINARY (band-ungated), never PASS.")

    out = delta_floor_headline(
        steered, annotated, arms=a.arms, alpha_star=alpha_star,
        behaviours=behaviours, band_b=band_b, margin=a.margin,
        n_resamples=a.n_resamples, seed=a.seed)

    print_table(out)
    out_path = Path(a.out) if a.out else eval_dir / "delta_floor_report.json"
    out_path.write_text(json.dumps(out, indent=2))
    log.info(f"report → {out_path}")


if __name__ == "__main__":
    main()
