# Project checkpoint — 28 May 2026, 10:30 BST

> ⚠️ **Predates the 2026-06-05 estimator fixes.** The "fresh, corrected re-run"
> below is the May-28 *cap* fix only. A later 2026-06-05 audit (`AUDIT.md` §2)
> found the intrinsic-dim and curvature estimators themselves biased; the
> geometry numbers in this checkpoint are **superseded, pending regeneration**
> (`results/_STALE_pre_fix_20260605/`, `INVENTORY.md`).

Snapshot after a **fresh, corrected re-run** of the geometry pipeline. The earlier (saturated-d_eff) run is archived under `results/_archive_run1_*`; nothing below is contaminated by it.

---

## 1. Top-line status

**The curved-manifold hypothesis is statistically supported at 1.5B, and now on a much stronger footing.** With the PCA cap bug fixed and the chain-stratified null run at **every one of the 28 layers**, the evidence is:

| Test | Result | Verdict |
|---|---|---|
| d_eff_70 ≫ 1 (Venhoff falsification) | **45–98** across (behaviour, layer) | ✅ decisively falsifies single-direction |
| Intrinsic dim ≪ PCA dim (curved manifold) | TwoNN 10–13 vs PCA 47–61 at L17 (~5×) | ✅ manifold signature |
| Curvature (geodesic / tangent / local-vs-global) | all three agree: strongly curved | ✅ curved manifold |
| Chain-stratified null rejects (real, not confound) | backtrack & uncertainty **28/28 layers**; example-test 19/28; add-knowledge L17–19 | ✅ real signal |

**Backtracking & uncertainty are rock-solid (significant at all 28 layers). Example-testing is strong (19/28). Adding-knowledge is real but localised to L17–19.**

---

## 2. What's complete

### Data
- 1,000 R1-Distill-Qwen-1.5B chains; 993/1000 annotated (77,183 spans); 7 partial chains excluded
- 1,000 baseline chains (Qwen-Math-1.5B) generated on cluster — **annotation pending API credits**
- Phase-4 activations: 37,851 vectors × 28 layers (6.5 GB)

### Pipeline (fresh local re-run, this morning)
- ✅ Phase 1–4 (tasks, chains, annotation, extraction)
- ✅ Phase 5 — PCA across all 28 layers (cap=100) **+ chain-stratified null at every layer**
- ✅ Phase 5c — cross-layer probing (28 layers)
- ✅ Triangulation — PR-argmin geometry + probe + (patching deferred)
- ✅ Phase 5d — sub-type clustering at PR-trough layers
- 🔄 Phase 5b — geometric deep-dive (B=300, layers 11/14/17/20/27) — finishing
- ⏳ Phase 6/7 — steering construction + evaluation (blocked on API credits; build at the PR-trough layers)

### Code changes this session (the master-pipeline fixes)
1. **PCA component cap 50 → 100** — d_eff_70 was pinned at exactly 50 for 3/4 behaviours; now reads true 45–98.
2. **Triangulation geometry signal → participation-ratio argmin** (was d_eff argmax, which was unreliable once d_eff saturated and forced the {L18,L27} fallback).
3. **5d focus layer → argmin(PR)** (was argmax(d_eff), which returned the layer-0 artifact).
4. **Fixed a missing `import numpy as np`** in `05_pca_analysis.py`'s `--with-nulls` path (crashed the first launch; caught + fixed).
5. **Triangulation robustness** — boundary-aware smoothing (`edge` not `reflect`, so example-testing's L27 trough survives) + honest "single-signal" agreement labels.
6. Old results archived; clean re-run.

---

## 3. Headline results — the fresh scorecard

![Scorecard](fig9_scorecard.png)

| Behaviour | Intrinsic dim (L17) | PCA d_eff_70 (L17) | Compression | Geodesic/Eucl | Tangent | PR trough | Null sig (p<.01) |
|---|---|---|---|---|---|---|---|
| backtracking | 10.4 | 47 | 4.5× | 1.83 | 58° | L14 | **28/28** |
| uncertainty-estimation | 11.1 | 57 | 5.1× | 1.82 | 62° | L14 | **28/28** |
| example-testing | 10.7 | 54 | 5.1× | 1.68 | 58° | L27 | 19/28 |
| adding-knowledge | 13.3 | 61 | 4.6× | 1.62 | 68° | L17 | 3/28 (L17–19) |

*Layer-matched view (each behaviour at its own PR-trough): compression rises to **6.6× / 8.6× / 7.0×** for backtracking / uncertainty / example-testing as intrinsic dim drops to 6–7. The table above uses L17 for continuity.*

### Two observations worth raising with the supervisor

1. **Geometry vs decodability come apart.** Every behaviour is linearly decodable 85–92% at *all* layers (probe is flat), yet the *geometric* concentration (PR) peaks at middle layers (L14–17). "Where it's decodable" ≠ "where it's geometrically structured." This is a clean methodological point.
2. **Curvature strengthens with depth.** d_eff_70 climbs to 98 by L26 while intrinsic dim stays ~10–13 — the curved-manifold gap widens toward late layers.

---

## 4. Comparison with prior work

| Study | Concept type | Intrinsic dim | Statistical control |
|---|---|---|---|
| Goodfire 2026 (manifold steering) | Cyclic concepts (days, months) | 1 (1-D circle) | None — visualisation only |
| Engels et al. 2024 | Generic features | 2–6 | None |
| Venhoff et al. 2025 | Reasoning behaviours | 1 (assumed) | Bootstrap CIs only |
| **Us 2026** | **Reasoning behaviours** | **10–13** | **Chain-stratified permutation null at every layer** |

We **extend** the literature: same manifold signature (intrinsic ≪ ambient + curvature) but at higher dimensionality in non-parametric reasoning behaviours, plus a methodological contribution (the per-layer chain-stratified null).

---

## 5. What's running

| Process | Started | Output |
|---|---|---|
| Phase 5b deep-dive (laptop) | 10:10 | `results/geometric/R1-1.5B/` |
| 7B chain gen (cluster, 500 balanced) | 09:31 | `data/chains_R1-7B.json` (5/500; slow, ~11+ days) |

---

## 6. Morning / next-session checklist

### 1. Confirm the local re-run finished cleanly
```bash
cd "/Users/tonysu/Documents/Reasoning on a Manifold/reasoning-on-manifold"
tail -n 30 logs/master_rerun.log        # should end with "=== FRESH RE-RUN COMPLETE ==="
ls results/geometric/R1-1.5B/           # diagnostics_layer{11,14,17,20,27}.json present?
```

### 2. Read the triangulation candidate layers
```bash
cat results/triangulation/R1-1.5B/summary.md
```
PR-driven candidates (probe flat, patching deferred): backtrack **L13**, uncertainty **L13**, example-test **L27**, add-knowledge **L18**.

### 3. Inspect sub-type clustering (note: weak)
```bash
for f in results/clustering/R1-1.5B/*/summary.json; do cat "$f"; echo; done
```
All k=2, silhouette 0.11–0.18 → continuous manifold, not discrete sub-types.

### 4. When API credits arrive — kick off baseline annotation (on cluster)
```bash
ssh spark-06aa "cd reasoning-on-manifold && nohup env CLAUDE_PROXY_URL=... CLAUDE_PROXY_KEY=... /home/tony/venv/bin/python3 03_annotate_chains.py --model-short QwenMath-1.5B > logs/baseline_annotate.log 2>&1 &"
```

### 5. Check on the 7B run
```bash
ssh spark-06aa "cd reasoning-on-manifold && python3 -c 'import json;print(len(json.load(open(\"data/chains_R1-7B.json\"))),\"/500\")'"
```

---

## 7. Decisions to weigh

| Decision | Tradeoff |
|---|---|
| Request API credits ($290) now | Bigger ask, unblocks the steering half of the paper |
| 7B scope: keep 500, or cut to ~200 / shorten max_new_tokens? | 500 ≈ 11+ days; 200 ≈ ~4–5 days and still a cross-scale replication |
| Run 14B too? | Probably no — diminishing returns vs cost |
| Reframe the "sub-type steering" study | 5d found no discrete sub-types → pitch as continuous-manifold steering instead |
| Venue | Workshop (faster) vs conference |

---

## 8. Supporting documents in `results/supervisor_meeting/`

- `SUMMARY.md` / `.html` — overall status (+ §11 "what changed since yesterday")
- `PHASE_5B_DEEP_DIVE.md` / `.html` — full methodology + Goodfire comparison
- **`CHECKPOINT.md` (this file)** — read first
- Figures: fig1 (layer sweep, fresh), fig8 (28-layer null, fresh), fig9 (scorecard, fresh); fig3/4 (corpus, unchanged); fig6/7 (5b intrinsic/curvature, stable)

---

## 9. One-paragraph status for the supervisor

> *"As of 28 May (corrected re-run), all four target behaviours in R1-Distill-Qwen-1.5B have PCA d_eff_70 of 45–98 — decisively falsifying the single-direction prediction — with intrinsic dim ≈ 10–13 (a ~5× compression, the curved-manifold signature) and three curvature diagnostics in agreement. The chain-stratified permutation null, now run at **every one of the 28 layers**, rejects the chain-confound at p<0.01 for backtracking and uncertainty at all 28 layers, example-testing at 19, and adding-knowledge at L17–19. Geometric concentration (participation ratio) peaks at middle layers L14–17 even though the behaviours are linearly decodable 85–92% everywhere — geometry and decodability come apart. Sub-type clustering finds no clean discrete sub-types (continuous manifolds). A 7B cross-scale replication is generating on the cluster (500 balanced chains). The only blocker for the steering half of the paper is ~$290 in Sonnet API credit for baseline + Phase-7 annotation."*
