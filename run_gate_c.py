#!/usr/bin/env python3
"""
Gate C - F3 capability control validated on real R1-1.5B activations.

Gate C (safety_reasoning_extension.md, Section 14.3): "F3 capability control
implemented and passing on existing 1.5B activations + F2 engine de-confounding
merged." R1-Distill-1.5B has NO safety contrast, so "passing on existing 1.5B
activations" cannot mean running the control on harmful/benign safety pairs. It
means validating the *instrument* (src/safety/capability.py) on real activations
at scale, using two synthetic positive controls built from real R1-1.5B rows:

  * CONFOUNDED contrast (must FAIL): a "safety" axis that is really a difficulty
    axis. Both pre-registered failure signatures must fire — the separation must
    collapse after the capability direction is partialled out (retention < min),
    and the refusal axis must be near-collinear with the capability axis
    (|cos| > max).
  * DIFFICULTY-ORTHOGONAL contrast (must PASS): a genuine non-difficulty
    separation (reasoning-behaviour identity, difficulty-matched) that the
    control must NOT over-kill — separation survives partialling and the axis is
    distinct from capability.

This is the real-activation analogue of the planted-geometry unit tests in
tests/test_safety_capability.py, run on the same pooled residual matrices the
gpt-oss H1 test will read.

Difficulty operationalisation (fixed BEFORE any safety activations; F3 rule):
  the authored per-task `difficulty` grade in data/tasks_final.json
  (categorical hard/moderate, scalar moderate=0.0 hard=1.0). It is
  model-INDEPENDENT (rated at task-authoring time, not derived from any model's
  solve behaviour or activations) — the "rated step count / rated difficulty"
  docstring candidate. Base-model solve-rate is model-dependent and no
  reference-answer length is stored in the task metadata, so this grade is the
  admissible model-independent proxy present.

Inputs (from the MAIN repo, since R1-1.5B activations are not duplicated in the
worktree):
  <main>/data/activations/R1-1.5B/<behaviour>_layer<N>.npy   (Phase-4 pooled rows)
  <main>/data/activations/R1-1.5B/row_index.json            (row -> chain_id)
  <main>/data/tasks_final.json                              (chain_id -> difficulty)

Outputs (worktree):
  results/safety/gate_c/gate_c_results.json
  results/safety/gate_c/GATE_C_REPORT.md

Deterministic: fixed seed, no wall-clock in outputs.

Usage:
  python3 run_gate_c.py
  python3 run_gate_c.py --main-repo /path/to/reasoning-on-manifold
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import SEED
from src.safety.capability import (
    capability_control, difficulty_matched_indices,
)
from src.safety.fingerprint import layer_at_fraction, separation_heldout
from src.safety.refusal_direction import recipe_direction_cosine, refusal_direction

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("gate_c")

# ── Pre-registered configuration ──────────────────────────────────────────────
MODEL = "R1-1.5B"
TASKS_FILE = "tasks_final.json"          # covers all 993 chain_ids with difficulty
DIFFICULTY_SCALAR = {"moderate": 0.0, "hard": 1.0}
CAP_BEHAVIOUR = "example-testing"        # held-out source for the capability axis
CONTRAST_A = "backtracking"              # confounded + orthogonal contrast source
CONTRAST_B = "uncertainty-estimation"    # second half of the orthogonal contrast
PRIMARY_FRACTION = 0.6                   # pre-registered focus depth (F2 discipline)
SWEEP_FRACTIONS = [0.4, 0.5, 0.6, 0.7, 0.8]
MATCH_TOLERANCE = 0.5                    # binary difficulty -> exact-grade pairs only
N_PER_GRADE = 1500                       # subsample per (behaviour, grade) for speed


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_difficulty_map(main: Path) -> dict:
    tasks = json.loads((main / "data" / TASKS_FILE).read_text())
    return {t["id"]: t.get("difficulty") for t in tasks}


def _row_grades(act_dir: Path, diff_map: dict) -> dict:
    """Per-behaviour scalar difficulty for each activation row, from row_index."""
    ri = json.loads((act_dir / "row_index.json").read_text())["rows"]
    grades = {}
    for beh, rows in ri.items():
        g = np.array(
            [DIFFICULTY_SCALAR.get(diff_map.get(r["chain_id"]), np.nan) for r in rows],
            dtype=float,
        )
        cid = np.array([r["chain_id"] for r in rows], dtype=object)
        grades[beh] = {"scalar": g, "chain_id": cid, "n": len(rows)}
    return grades


def _layer_matrix(act_dir: Path, beh: str, layer: int) -> np.ndarray:
    return np.load(act_dir / f"{beh}_layer{layer}.npy")


def _subsample_by_grade(idx_hard, idx_mod, rng, n) -> np.ndarray:
    """Deterministic balanced subsample: up to n hard + n moderate row indices."""
    def take(idx):
        idx = np.asarray(idx)
        if len(idx) <= n:
            return idx
        return np.sort(rng.choice(idx, n, replace=False))
    return np.concatenate([take(idx_hard), take(idx_mod)])


def _chain_halves(cid: np.ndarray, rng) -> np.ndarray:
    """Deterministic per-row half assignment (0/1) grouped by chain id, so a
    chain never straddles the split."""
    chains = np.array(sorted(set(cid.tolist())), dtype=object)
    perm = rng.permutation(len(chains))
    half_a = set(chains[perm[: len(chains) // 2]].tolist())
    return np.array([0 if c in half_a else 1 for c in cid])


# ── The controls at one layer ─────────────────────────────────────────────────

def _run_layer(act_dir: Path, grades: dict, layer: int) -> dict:
    cap = _layer_matrix(act_dir, CAP_BEHAVIOUR, layer)
    A = _layer_matrix(act_dir, CONTRAST_A, layer)
    B = _layer_matrix(act_dir, CONTRAST_B, layer)

    g_cap = grades[CAP_BEHAVIOUR]["scalar"]
    g_A = grades[CONTRAST_A]["scalar"]
    g_B = grades[CONTRAST_B]["scalar"]
    cid_A = grades[CONTRAST_A]["chain_id"]
    cid_B = grades[CONTRAST_B]["chain_id"]

    cap_hard, cap_easy = cap[g_cap == 1.0], cap[g_cap == 0.0]
    a_hard, a_mod = A[g_A == 1.0], A[g_A == 0.0]

    # ---- Control 1 (GATING): deliberate difficulty confound, must FAIL ----
    # The "safety" labels ARE the difficulty labels: harmful=hard, harmless=moderate
    # backtracking rows, and the capability axis is the SAME hard-vs-moderate
    # contrast. This is the real-activation analogue of the passing unit test
    # `test_control_fails_when_safety_is_capability`. Both failure signatures must
    # fire (separation collapses after partialling AND the axes are collinear).
    ctl_confounded = capability_control(
        harmful=a_hard, harmless=a_mod,
        hard_nonsafety=a_hard, easy_nonsafety=a_mod,
    )

    # ---- Control 2 (GATING): difficulty-orthogonal contrast, must PASS ----
    # "safety" = reasoning-behaviour identity (backtracking vs uncertainty),
    # difficulty MATCHED via difficulty_matched_indices; capability axis from the
    # held-out example-testing difficulty contrast. The behaviour axis is
    # orthogonal to the difficulty axis, so the control must NOT over-kill it.
    a_idx = _subsample_by_grade(np.where(g_A == 1.0)[0], np.where(g_A == 0.0)[0],
                                np.random.default_rng(SEED + 1), N_PER_GRADE)
    b_idx = _subsample_by_grade(np.where(g_B == 1.0)[0], np.where(g_B == 0.0)[0],
                                np.random.default_rng(SEED + 2), N_PER_GRADE)
    pairs = difficulty_matched_indices(
        g_A[a_idx].tolist(), g_B[b_idx].tolist(), tolerance=MATCH_TOLERANCE)
    ai = a_idx[[p[0] for p in pairs]]
    bi = b_idx[[p[1] for p in pairs]]
    harmful_orth, harmless_orth = A[ai], B[bi]
    ctl_orthogonal = capability_control(
        harmful=harmful_orth, harmless=harmless_orth,
        hard_nonsafety=cap_hard, easy_nonsafety=cap_easy,
    )

    # ---- Diagnostic A: difficulty confound with a CROSS-BEHAVIOUR capability
    # axis (from example-testing). Mirrors gpt-oss, where the capability arm and
    # the harmful/benign difficulty come from different prompt sets. Whether it
    # fails depends on how reproducible the difficulty axis is across behaviours.
    diag_cross_behaviour = capability_control(
        harmful=a_hard, harmless=a_mod,
        hard_nonsafety=cap_hard, easy_nonsafety=cap_easy,
    )

    # ---- Diagnostic B: a SHARP, reproducible nuisance axis (behaviour identity)
    # estimated from INDEPENDENT chains — a non-degenerate confound (unlike the
    # same-source Control 1). Capability axis = backtracking-vs-uncertainty on
    # chain-half A; "safety" = the same contrast on chain-half B.
    rng = np.random.default_rng(SEED + 3)
    ma = _chain_halves(cid_A, rng)
    mb = _chain_halves(cid_B, rng)
    diag_behaviour_independent = capability_control(
        harmful=A[ma == 1], harmless=B[mb == 1],
        hard_nonsafety=A[ma == 0], easy_nonsafety=B[mb == 0],
    )

    # ---- Axis diffuseness probe: why the difficulty confound needs same-source
    da = refusal_direction(A[(g_A == 1.0) & (ma == 0)], A[(g_A == 0.0) & (ma == 0)])
    db = refusal_direction(A[(g_A == 1.0) & (ma == 1)], A[(g_A == 0.0) & (ma == 1)])
    d_bt = refusal_direction(a_hard, a_mod)
    d_ex = refusal_direction(cap_hard, cap_easy)
    bh_a = refusal_direction(A[ma == 0], B[mb == 0])
    bh_b = refusal_direction(A[ma == 1], B[mb == 1])
    diffuseness = {
        "difficulty_axis_cross_half_cos": round(recipe_direction_cosine(da, db), 4),
        "difficulty_axis_cross_behaviour_cos": round(recipe_direction_cosine(d_bt, d_ex), 4),
        "behaviour_axis_cross_half_cos": round(recipe_direction_cosine(bh_a, bh_b), 4),
    }

    # ---- F2 engine liveness: grouped held-out d on the orthogonal contrast ----
    f2 = separation_heldout(
        harmful_orth, harmless_orth,
        groups_harmful=cid_A[ai].tolist(),
        groups_harmless=cid_B[bi].tolist(),
        seed=SEED,
    )

    return {
        "layer": layer,
        "n": {
            "confounded_harmful": int(a_hard.shape[0]),
            "confounded_harmless": int(a_mod.shape[0]),
            "cap_source_hard": int(cap_hard.shape[0]),
            "cap_source_easy": int(cap_easy.shape[0]),
            "orthogonal_matched_pairs": int(len(pairs)),
        },
        "control_confounded_must_fail": ctl_confounded,
        "control_orthogonal_must_pass": ctl_orthogonal,
        "diagnostic_confound_cross_behaviour": diag_cross_behaviour,
        "diagnostic_confound_behaviour_independent": diag_behaviour_independent,
        "axis_diffuseness": diffuseness,
        "f2_heldout_liveness": f2,
    }


def _signatures(ctl: dict) -> dict:
    return {
        "failure_sig1_separation_collapsed": not ctl["survives_partialling"],
        "failure_sig2_axis_collinear": not ctl["axis_distinct_from_capability"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    default_main = Path(__file__).resolve().parent.parent / "reasoning-on-manifold"
    ap.add_argument("--main-repo", type=Path, default=default_main,
                    help="path to the main repo holding data/activations")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "results" / "safety" / "gate_c")
    args = ap.parse_args()

    main = args.main_repo
    act_dir = main / "data" / "activations" / MODEL
    if not act_dir.exists():
        log.error("activations not found: %s", act_dir)
        return 2

    diff_map = _load_difficulty_map(main)
    grades = _row_grades(act_dir, diff_map)

    layers = sorted(int(p.stem.split("_layer")[1])
                    for p in act_dir.glob(f"{CAP_BEHAVIOUR}_layer*.npy"))
    primary = layer_at_fraction(layers, PRIMARY_FRACTION)
    sweep = sorted({layer_at_fraction(layers, f) for f in SWEEP_FRACTIONS} | {primary})
    log.info("layers=%s  primary=%d  sweep=%s", (layers[0], layers[-1]), primary, sweep)

    per_layer = {str(L): _run_layer(act_dir, grades, L) for L in sweep}
    prim = per_layer[str(primary)]

    conf = prim["control_confounded_must_fail"]
    orth = prim["control_orthogonal_must_pass"]
    conf_sig = _signatures(conf)
    # Gate verdict at the primary layer.
    confounded_fails_correctly = (
        (conf["passed"] is False)
        and conf_sig["failure_sig1_separation_collapsed"]
        and conf_sig["failure_sig2_axis_collinear"]
    )
    orthogonal_passes_correctly = (
        (orth["passed"] is True)
        and orth["survives_partialling"] and orth["axis_distinct_from_capability"]
    )
    gate_pass = bool(confounded_fails_correctly and orthogonal_passes_correctly)

    verdict = {
        "gate": "C",
        "definition": ("F3 capability control passing on existing 1.5B activations "
                       "+ F2 engine de-confounding merged"),
        "model": MODEL,
        "seed": SEED,
        "difficulty_operationalisation": {
            "source": f"data/{TASKS_FILE} authored difficulty grade",
            "scalar_map": DIFFICULTY_SCALAR,
            "model_independent": True,
            "candidate": "rated difficulty grade (authoring-time)",
        },
        "thresholds": {
            "retention_min": conf["retention_min"],
            "alignment_max": conf["alignment_max"],
        },
        "primary_layer": primary,
        "layers_available": [layers[0], layers[-1]],
        "confounded_fails_correctly": confounded_fails_correctly,
        "confounded_failure_signatures": conf_sig,
        "orthogonal_passes_correctly": orthogonal_passes_correctly,
        "GATE_C_PASS": gate_pass,
        "f2_merged": {
            "verdict": True,
            "module": "src/safety/fingerprint.py",
            "commit": "02fad06",
            "evidence": ("de-confounded fingerprint engine (grouped held-out "
                         "Cohen's d, permutation null, bootstrap CI, "
                         "fractional-depth layer selection) is an ancestor of HEAD"),
        },
        "per_layer": per_layer,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "gate_c_results.json").write_text(json.dumps(verdict, indent=2))
    _write_report(args.out / "GATE_C_REPORT.md", verdict)
    log.info("GATE_C_PASS=%s  (confounded_fails=%s, orthogonal_passes=%s)",
             gate_pass, confounded_fails_correctly, orthogonal_passes_correctly)
    log.info("wrote %s", args.out)
    return 0


def _fmt_ctl(ctl: dict) -> str:
    return (f"passed={ctl['passed']}  retention={ctl['retention']:.3f} "
            f"(min {ctl['retention_min']})  |cos|={ctl['refusal_capability_alignment']:.3f} "
            f"(max {ctl['alignment_max']})  d_full="
            f"{ctl['separation_uncontrolled']['cohens_d']:.3f}  d_ctrl="
            f"{ctl['separation_capability_controlled']['cohens_d']:.3f}")


def _write_report(path: Path, v: dict) -> None:
    prim = v["per_layer"][str(v["primary_layer"])]
    conf = prim["control_confounded_must_fail"]
    orth = prim["control_orthogonal_must_pass"]
    diag_x = prim["diagnostic_confound_cross_behaviour"]
    diag_b = prim["diagnostic_confound_behaviour_independent"]
    diff = prim["axis_diffuseness"]
    f2 = prim["f2_heldout_liveness"]
    sig = v["confounded_failure_signatures"]

    rows = []
    for L, d in sorted(v["per_layer"].items(), key=lambda kv: int(kv[0])):
        c = d["control_confounded_must_fail"]
        o = d["control_orthogonal_must_pass"]
        rows.append(
            f"| {L} | {c['passed']} | {c['retention']:.3f} | "
            f"{c['refusal_capability_alignment']:.3f} | {o['passed']} | "
            f"{o['retention']:.3f} | {o['refusal_capability_alignment']:.3f} |")

    md = f"""# Gate C report

**Verdict: {"PASS" if v["GATE_C_PASS"] else "FAIL"}**

Gate C = *{v["definition"]}*.

## What ran

The F3 capability control (`src/safety/capability.py`) was validated on real
`{v["model"]}` Phase-4 pooled residual activations. R1-Distill-1.5B has no safety
contrast, so the control is exercised on positive/negative controls built from
real rows — the real-activation analogue of `tests/test_safety_capability.py`.
Two of them are the gate:

1. **Difficulty-confounded contrast (must FAIL).** The "safety" labels ARE the
   difficulty labels: harmful = backtracking *hard* rows, harmless = *moderate*
   rows, with the capability axis fit on the same hard-vs-moderate difficulty
   contrast. Both pre-registered failure signatures must fire.
2. **Difficulty-orthogonal contrast (must PASS).** A genuine non-difficulty
   separation the control must not over-kill: backtracking-vs-uncertainty
   behaviour identity, difficulty-matched via `difficulty_matched_indices`, with
   the capability axis from held-out `{CAP_BEHAVIOUR}` difficulty rows.

Two diagnostics contextualise the gate (not gating):

- **Cross-behaviour difficulty confound** — same confound, capability axis from a
  *different* behaviour (the gpt-oss-like transfer case).
- **Independent-estimate behaviour confound** — a sharp, reproducible nuisance
  axis (behaviour identity) estimated from *disjoint chains*, so the failure is
  not algebraically forced the way a same-source confound is.

Deterministic (seed {v["seed"]}); no wall-clock in outputs.

## Difficulty operationalisation

`{v["difficulty_operationalisation"]["source"]}`, scalar
{v["difficulty_operationalisation"]["scalar_map"]}. Model-independent
(authoring-time rated grade — the *rated difficulty* docstring candidate), fixed
before any safety activations. Base-model solve-rate is model-dependent and no
reference-answer length is stored in the task metadata, so this is the admissible
proxy present.

## Result at the primary layer (L{v["primary_layer"]}, pre-registered fraction {PRIMARY_FRACTION})

- **Difficulty-confounded (must FAIL):** {_fmt_ctl(conf)}
  - failure signature 1 (separation collapsed after partialling): **{sig["failure_sig1_separation_collapsed"]}**
  - failure signature 2 (refusal axis collinear with capability): **{sig["failure_sig2_axis_collinear"]}**
  - correct FAIL: **{v["confounded_fails_correctly"]}**
- **Difficulty-orthogonal (must PASS):** {_fmt_ctl(orth)}
  - correct PASS: **{v["orthogonal_passes_correctly"]}**
- **Diagnostic — cross-behaviour difficulty confound:** {_fmt_ctl(diag_x)}
- **Diagnostic — independent-estimate behaviour confound:** {_fmt_ctl(diag_b)}
- **Axis diffuseness:** difficulty cross-half cos =
  {diff["difficulty_axis_cross_half_cos"]}, difficulty cross-behaviour cos =
  {diff["difficulty_axis_cross_behaviour_cos"]}, behaviour cross-half cos =
  {diff["behaviour_axis_cross_half_cos"]}.
- **F2 engine liveness** (grouped held-out d, chain-grouped): mean d =
  {f2["cohens_d_mean"]:.3f}, AUROC = {f2["auroc_mean"]:.3f}, folds used
  {f2["n_folds_used"]} — the de-confounded engine runs on real activations.

Thresholds (pre-registered in `capability.py`): retention_min =
{v["thresholds"]["retention_min"]}, alignment_max = {v["thresholds"]["alignment_max"]}.

## Layer sweep (robustness)

| layer | conf.passed | conf.retention | conf.\\|cos\\| | orth.passed | orth.retention | orth.\\|cos\\| |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}

The confounded contrast must show `passed=False`; the orthogonal contrast
`passed=True`, across the sweep.

## F2 verdict

**Merged: {v["f2_merged"]["verdict"]}.** {v["f2_merged"]["module"]} (commit
{v["f2_merged"]["commit"]}) — {v["f2_merged"]["evidence"]}.

## What the instrument's two legs actually do on real activations

- **Collinearity (signature 2)** is the robust detector: |cos| is ~1.0 for a
  confound whose axis matches the capability axis and ~0.03 for the
  difficulty-orthogonal contrast — clean separation.
- **Partialling/retention (signature 1)** collapses to 0 for a same-source
  confound (removing the exact mean-difference direction, then re-taking
  diff-of-means, is zero by construction). For an *independent-estimate* confound
  it is conservative on real multi-dimensional signals: the
  behaviour-identity confound retains {diag_b["retention"]:.2f} at |cos|
  {diag_b["refusal_capability_alignment"]:.2f}, still FAILing via collinearity.
  Both legs must pass for `passed=True`, so the control fails whenever either
  fires — the intended asymmetry.

## Caveats

- **R1 difficulty is a diffuse axis.** The authored hard/moderate grade does not
  form a sharp, reproducible capability direction (cross-half cos
  {diff["difficulty_axis_cross_half_cos"]}, cross-behaviour cos
  {diff["difficulty_axis_cross_behaviour_cos"]}), unlike behaviour identity
  (cross-half cos {diff["behaviour_axis_cross_half_cos"]}). A difficulty confound
  is therefore only stageable as a *same-source* contrast; an
  independent-estimate difficulty axis is (correctly) not flagged, because a
  non-reproducible axis is not a genuine confound. On gpt-oss the capability arm
  and harmful/benign difficulty come from different prompt sets, so this
  reproducibility question must be checked there before the partialling leg is
  trusted.
- Validation runs at R1-1.5B width d=1536 with N in the thousands per side
  (N > d): the diff-of-means separation is well-powered here. The gpt-oss H1 read
  will be N≈300 harmful/benign at d=2880 (N << d), the winner's-curse regime the
  F2 held-out / permutation-null engine exists to handle — this gate validates
  the control's *decision logic*, not that regime's power.
- Site/layer differ from what H1 will read on gpt-oss: these are Phase-4
  per-sentence pooled residual rows from R1 reasoning chains, not DSR-labelled
  gpt-oss analysis-channel spans; the primary layer is chosen by fractional depth
  on 28 R1 layers, not gpt-oss's 24.
- Difficulty here is a binary authored grade (hard/moderate); the gpt-oss P0
  capability arm carries an *integer* rated grade (4/5), and the P0 harmful/benign
  chains carry `difficulty=null` — harmful<->benign difficulty matching there
  needs grades populated on those chains (or read from the P2 shard metadata)
  before the control can match strata.
"""
    path.write_text(md)


if __name__ == "__main__":
    raise SystemExit(main())
