# E10.1 — DAS-1D on backtracking: pre-registration

> Sealed before the pod run. Pairs with `METHODOLOGY.md` §8, `RESULTS_LEDGER.md` §B4,
> thesis `sec:steering-featurizer-programme`. Script: `20_das_backtracking.py`.
> Launch wrapper: `runpod_e10_das.sh`. **Last updated: 2026-07-06.**

## Question
Does a **causally selected** one-dimensional frame for backtracking steer at least as
well as the **correlationally built** difference-of-means direction (E1)? This is the
featurizer upgrade of Phase 7: the frame is chosen because swapping the coordinate it
designates *transfers the behaviour* (interchange-intervention / DAS criterion), not
because it correlates with annotations.

## AMENDMENT 1 (2026-07-06, BEFORE the pod run — objective upgraded)
Minival-1 (48 pairs, L17, MPS, single-token objective) had its **shuffled-pair illusion
control FIRE**: learned sym Δlogprob +6.56 but shuffled-pair +5.87 (random +0.004,
positional +0.33). Diagnosis: with a single onset-token counterfactual label, every
source target is a backtracking-onset token, so a **generic onset-token promoter**
("Wait-booster") satisfies the objective with no pair-specific alignment — exactly the
Makelov failure mode the control exists to catch. `results/das/R1-1.5B/minival/`.

**Fix (IIT-style):** the counterfactual label is now the donor's real **W=12-token
continuation**, teacher-forced (`--window`); the loss is symmetric
(induce: base ctx + source continuation; remove: source ctx + base continuation). A
generic promoter can raise the first token but not the donor's specific continuation, so
pair shuffling now breaks what matters. New endpoint: **sym Δlogprob** (mean per-token,
patched vs clean, averaged over induce+remove) + **pair_specificity_gap** =
learned − shuffled_pair. The single-token objective (W=1) is retained as an ablation
arm, not a headline. Predictions P1/P2 unchanged in substance and re-sealed on the new
endpoint BEFORE any pod run; minival-2 (32 pairs, windowed) is the local convergence
check, also pre-pod.

## AMENDMENT 2 (2026-07-06, BEFORE the pod run — the right null for a binary variable)
Minival-2 (32 pairs, W=12, L17): learned sym Δlogprob **+0.130** vs diff-of-means
**+0.043** (3×, at cos(learned, dm)=0.233 — the causal criterion finds a *different,
stronger* direction than the correlational recipe: P1 directionally supported at pilot
scale); random +0.001; positional +0.040. But shuffled-pair (+0.152) again matched the
learned direction. **Reinterpretation:** for a BINARY behaviour variable, within-class
donor shuffling is NOT an illusion null — every behaviour-positive donor is a valid
interchange, so shuffled ≈ learned is the expected signature of a *shared* behaviour
feature (the thesis's own low-dimensionality claim), not evidence of optimizer noise.
The Makelov-style illusion is instead tested by the new `--stage controls`
(`controls.json`):
- **base-donor null** — swap in a non-behaviour donor; a state-reading frame collapses,
  a constant-bias frame keeps firing. Minival-2 learned: real +0.107 vs null +0.052
  (state-dependence +0.055, ratio ≈ 2×); diff-of-means: +0.031 vs +0.007 (ratio ≈ 4.8×,
  but 3× weaker absolute).
- **coordinate separation** — AUC of h·d, source vs base prediction states. Minival-2:
  learned 0.771, diff-of-means 0.757.
Shuffled-pair is RETAINED but re-labelled *within-class transfer* (a positive
diagnostic); the illusion verdict rests on base-donor null + coordinate AUC + positional.
Pod run: all arms at n=400 pairs, 3 layers, with per-pair bootstrap CIs computed in
analysis (point estimates above are n=32, no CIs, directional only).

## ADJUDICATION (2026-07-06 — interchange stage RUN, n=400, RunPod 4090, pod killed post-pull)
Full numbers: `results/das/R1-1.5B/main/ANALYSIS_2026-07-06.md`.
- **P1 SUPPORTED** on the interchange endpoint: learned +0.649 vs diff-of-means +0.048
  sym Δlogprob at L17 (≈13×); random +0.0003; positional +0.029 (÷22). Margins three
  orders over the floor; per-pair CIs still owed (cannot plausibly overturn a 13× gap).
- **Unique, different object:** cold and warm starts converge to one axis
  (|cos| ≥ 0.999 at every layer), nearly orthogonal to diff-of-means (|cos| ≈ 0.1–0.2).
- **Binary-variable signature confirmed:** shuffled_pair ≈ learned at all layers.
- **Grounding controls split the layers:** L17 passes (coord-AUC 0.760,
  state-dependence 4.3×). L11 (AUC 0.357) and L27 (AUC 0.211, worst state-dependence,
  largest base-donor leak) are high-transfer but ungrounded, optimizer-carved frames —
  by raw transfer L27 would have ranked first; the coordinate-AUC control catches the
  read-out-proximity trap for the third time in this programme.
- **P2 remains sealed and unrun** (generation stage: swap vs projective ablation at
  matched on-target effect, collapse endpoints, E8 battery). E10.2 unrun.

## Method (fixed)
- **Model / corpus:** R1-1.5B, the 986 annotated chains (`data/annotated_R1-1.5B.json`),
  Sonnet-primary annotations. Eval is within-annotator (builder) — same caveat as E8;
  the non-builder band is a later, separate spend.
- **Counterfactual pairs (onset-anchored):**
  - *source* = the token position that predicts a **backtracking**-span onset (model about
    to emit "Wait"/"Actually"/…);
  - *base* = the position that predicts an ordinary forward-reasoning continuation
    (**deduction** or **initializing** span onset).
  - Contexts right-truncated to `ctx` tokens; onsets located with the same
    `src.text_offsets` machinery Phase 4 uses (occurrence-aware, cursor-advanced over ALL
    labels). Target = the real onset token that position predicts.
- **Interchange swap** at steering layer `L` (`resid_pre[L]` == `hidden_states[L]`), for a
  learnable unit direction `d`:
  - `h_b' = h_b + d dᵀ(h_s − h_b)` (induce backtracking in the base),
  - `h_s' = h_s + d dᵀ(h_b − h_s)` (remove it from the source).
  - `h_b, h_s` are frozen constants from clean forwards; **only `d` is learned**.
- **Objective:** symmetric teacher-forced cross-entropy of the patched run toward the
  *other* member's real onset token (the counterfactual label). Adam, lr 1e-2, 60 epochs.
- **Layers:** `17` (E1 attribution), `11` (de-confounded 07d mid-peak for backtracking),
  `27` (read-out-proximity control the sweep taught us to carry).
- **Trained directions per layer:** *cold* (random init — the honest "found from scratch"
  arm) and *warm* (init from the E1 diff-of-means direction — does the causal criterion
  move it further?).

## Endpoints
Primary interchange metric (annotation-free, computed on the pairs):
- **Δlogprob** = logP(source onset token | base patched) − logP(source onset token |
  base clean) — the induction of backtracking into a non-backtracking context. (Δlogprob,
  not argmax accuracy: a single 1-D swap rarely flips the greedy token, but shifts the
  target's probability measurably — confirmed in the local smoke, diff-of-means Δlogprob
  +0.41 vs random +0.004.)
- Secondary: induction accuracy (argmax == source token), base CE.

## Pre-registered predictions (SEALED)
- **P1 (headline):** `Δlogprob(learned) ≥ Δlogprob(diff_of_means)` at the E1 layer (17).
  The causal frame does at least as well as the correlational one; a strict win is the
  interesting outcome, parity is the null-but-consistent outcome.
- **Controls that MUST be near zero** (else the result is a Makelov illusion, not a causal
  frame):
  - `shuffled_pair` (train `d` on permuted base↔source correspondence) ≈ 0;
  - `random_rotation` (untrained `d`) ≈ 0;
  - `learned_positional` (apply the learned swap at a misaligned position) ≈ 0.
- **P2 (follow-on, generation stage, NOT in this run):** at matched on-target effect, the
  interchange *swap* collapses less than the projective *ablation* of `eq:steer-apply`
  (the collapse-account prediction; requires free generation + the E8/E9 collapse table).

## What this run does and does not settle
- Settles: whether a causally-optimised 1-D frame exists for backtracking and how it
  compares to diff-of-means on the interchange metric, with the illusion controls.
- Does **not** settle: free-generation steering effectiveness (P2), causal width `k*`
  (that is E10.2, boundless-DAS), or the non-builder annotator band.

## Logistics
- **Where:** one RunPod GPU (4090/A100). No API credits (the training signal is the
  model's own teacher-forced CE; the annotator is not called in this run).
- **Cost:** pairs stage is CPU (seconds). Train = 3 directions × 3 layers × 60 epochs;
  each epoch is `n_pairs/bs` batches × 2 short forwards. Eval = one pass × 6 directions ×
  3 layers. Estimate **~1–3 GPU-hours total** ⇒ **~$1–5** on a 4090, well under the E8/E9
  budget class.
- **De-risking done locally (MPS):** gradient-flow asserted (grad_norm 2.1e-2, finite,
  nonzero); eval metric shown sensitive (diff-of-means ≫ random); end-to-end smoke green.
  A 48-pair mini-validation confirms the learned direction converges before pod spend.

## Run recipe
```
# local: build pairs (CPU) and sanity-smoke
python 20_das_backtracking.py --stage pairs
python 20_das_backtracking.py --smoke

# pod (once a GPU host 'runpod' is in ~/.ssh/config):
POD=runpod ./runpod_e10_das.sh setup     # one-time deps on a fresh pod
POD=runpod ./runpod_e10_das.sh push      # sync code + annotated data + E1 vector
POD=runpod ./runpod_e10_das.sh launch    # DAS train+eval in tmux 'e10', all 3 layers
POD=runpod ./runpod_e10_das.sh status    # tail progress
POD=runpod ./runpod_e10_das.sh pull      # rsync results/das back
# then STOP THE POD.
```

## AMENDMENT 3 (2026-07-06, sealed BEFORE pod run 2 — CIs, site-check, P2 operationalization, E10.2 widths)

**Context.** The E10.1 interchange stage is executed (pod 1, n=400, `results/das/R1-1.5B/main/`):
P1 supported at L17 (learned +0.65 vs diff-means +0.05 sym Δlogprob; coord-AUC 0.76;
state-dependence 4.3×); L11/L27 transfer is UNGROUNDED (coord-AUC 0.36/0.21) = optimizer-carved,
the Makelov mode caught by the Amendment-2 controls. Pod run 2 closes three owed items; all
decision rules below are sealed before any of it executes.

**(a) Per-pair bootstrap CIs** (`20_das --stage cis`): percentile bootstrap (10k) over the 400
pairs for every arm's sym Δlogprob + the paired learned−diff-means difference. P1 is CONFIRMED
(upgraded from "supported") iff the paired-difference CI95 excludes 0.

**(b) Diff-of-means SITE-CHECK.** Discovered post-E10.1: the pipeline's "layer L" vectors are
built at block-L OUTPUT = hidden_states[L+1]; E1 "bt17" diff-means therefore lives at hs[18],
while E10.1 evaluated all arms at hs[17]. The cis stage re-evaluates dm at BOTH sites.
**Sealed decision rule:** if sym_dlp(dm@hs18) > 2× sym_dlp(dm@hs17), the E10.1 headline ratio
(13×) is revised to the hs18 comparison and both are reported; the L17 grounding results are
unaffected (they are statements about hs[17]).

**(c) P2 operationalization** (`22_swap_vs_ablation.py`, `results/eval/R1-1.5B__E10_P2/`):
behaviour = backtracking; directions = DAS-learned (site hs17 → SteeredModel layer 16) and
E1 diff-means (site hs18 → layer 17), each at its OWN site; arms per direction = add α∈{0.5,1.0},
clamp_on (coordinate clamped to class-mean source value c_on), sub α=1.0, clamp_off; + shared
vanilla through the identical generation path. Greedy, cap 8192, the 50 held-out eval tasks.
Endpoints annotation-free: collapse = 4-gram repetition > 0.8 (primary), boxed rate, tokens.
Matching variable = mean realized |Δ(rᵀh)| recorded by the hook. **Sealed P2 prediction:**
clamp_on collapse < the add-curve collapse interpolated to the clamp's realized displacement;
"supported" iff the paired-bootstrap CI95 of (observed − interpolated) excludes 0 from above.
Secondary: exact McNemar clamp_on vs add(1.0). E9.1b anchors expected direction (amplify raises
collapse 36%→58–64%).

**(d) E10.2 causal width** (`21_das_width.py`, widths k∈{1,2,4,8,16,32} at hs[17], same pairs,
cold+shuffled+random-frame per width, grounding = k-dim coord-AUC + base-donor state-dependence).
**Sealed readings:** saturation at k=1 → causal object is 1-D and the E8 manifold null is
explained; rise to k≈6–8 matching corr-dim → low-dimensionality certified causally; further
ungrounded rise (AUC/state-dep flat while transfer climbs) → optimizer carving, width capped at
the last GROUNDED k. Bonus: adding-knowledge 1-D REMOVAL probe (donors=ak onsets, same recipe)
— "causally removable" iff remove_dlp beats random with grounded coord-AUC > 0.65.

**Budget:** one 4090 pod, sequential (cis ~20 min → P2 ~4–5 h → width ~3 h) ≈ 8–9 h ≈ $3–6.
Kill chain: local watcher (pull → `runpodctl remove pod 1r80whceblndrr`) + on-pod selfstop
backstop (12 h hard cap, 30 min post-marker grace).

## ADJUDICATION 2 (2026-07-07 — P2 pulled; E10.2 pending re-pull)
Pod-2 chain completed cleanly (markers cis 22:09 / P2 01:14 / width 03:53; the pod self-killed
03:54:53, `pod removed` — the instant-kill worked). P2 pulled and adjudicated; width complete on
the volume but not yet pulled (rsync chain bug + premature pod kill — retrieval owed).

**P2 sealed prediction (clamp collapse < projective-ablation collapse at matched displacement):
NOT cleanly testable from this run — CONFOUNDED.** The constant-coordinate clamp delivered mean
displacement 20.6 (das) / 24.3 (dm), far beyond the strongest ablation arm's 7.3 / 16.4, so the
add-curve interpolation flat-extrapolates and "matched displacement" was never achieved. Verdict
recorded as inconclusive, not as support/refute.

**What the run DOES show (descriptive, clean):**
- Grounded causal (L17-DAS) clamp collapses **0.44** vs correlational diff-means clamp **0.86**
  (McNemar vs add(1.0): das p=0.33 n.s.; dm p=0.004 — dm clamp significantly WORSE). The causal
  direction is far better-behaved under the same intervention family.
- **Causal ablation** (das subtract α=1) → collapse **0.30 ≈ vanilla 0.32**, and the **highest
  boxed rate of any arm (0.20** vs vanilla 0.06) — ablating backtracking via the grounded causal
  direction reduces looping and preserves answer-emission, corroborating the E9.1b odd-parity
  finding (ablate backtracking ⇒ less collapse) through a different (causal-basis) direction.
- **Methodological lesson (sealed for the redesign):** a constant class-mean coordinate CLAMP at
  every position is NOT a faithful generation-time analogue of the bounded paired interchange
  swap — it is its own aggressive, off-distribution intervention (hence the huge displacement).
  A clean P2 must displacement-MATCH the clamp/swap to the ablation arms (scale the clamp, or
  sweep clamp targets, or apply a true paired-state swap at generation time). Re-registered as
  the P2 redesign; the current run stands as the descriptive + lesson result.

## ADJUDICATION 3 / P2 REDESIGN (2026-07-07, SEALED before the run — `23_p2_redesign.py`)
The original P2 confound (clamp displacement 20–24 >> projective 7–16 ⇒ flat extrapolation) is
fixed by dosing BOTH families so their realized-displacement ranges overlap, then comparing the
collapse-vs-displacement CURVES on common support:
- projective amplify h'=h+α(rᵀh)r, α∈{0.5,1,2,3};
- bounded partial clamp h'=h+β(c_on−rᵀh)r, β∈{0.5,1.0} (β<1 makes the clamp dose-able);
- both on BOTH directions (grounded causal L17-DAS = headline; correlational diff-means = secondary),
  each at its own site; one shared vanilla; greedy, cap 8192, 50 held-out tasks. 13 arms × 50 = 650.
Local MPS smoke confirms overlap achieved (das clamp 11.6/22.7 inside proj ≤22.7; dm clamp 17.4/34.8
inside proj ≤37.2) ⇒ interpolation now IN-RANGE.

**SEALED prediction (per direction, headline = das):** on overlapping displacement support, clamp
collapse − projective collapse (interpolated to the clamp's displacement) is
- **< 0 (CI95 upper < 0)** ⇒ bounded/on-distribution clamp is GENTLER (the collapse-account claim);
- **CI95 brackets 0** ⇒ intervention TYPE is irrelevant, collapse is a pure function of displacement
  (a clean negative — "how far you move the coordinate is all that matters");
- **> 0** ⇒ clamp is WORSE (hard targeting fights the dynamics).
Matching variable = mean realized |Δ(rᵀh)| (annotation-free). Secondary on-target axis = lexical
backtracking-cue rate /1k tokens (crude proxy, secondary only). Endpoints: collapse (4-gram rep
>0.8), boxed rate, tokens. Kill chain as before (watcher pull + on-pod selfstop, 60s post-marker).

## ADJUDICATION 4 / P2 REDESIGN RESULT (2026-07-07 — RUN, 650 chains, pod killed post-verified-pull)
Confound fixed: both intervention families dose-swept, displacement ranges overlap, interpolation
in-range. **The sealed prediction is SUPPORTED — and the support is DIRECTION-SPECIFIC, which is
the real finding.**

- **Grounded causal direction (L17-DAS):** at matched coordinate displacement 11.0, the bounded
  clamp collapses **0.54 vs the projective shift's 0.85 — diff −0.31, CI95 [−0.44, −0.19]**
  (in-range, significant). The b1.0 point agrees (−0.52) though its displacement 20.6 slightly
  exceeds the α-sweep max 16.5 (projective already saturated at 0.96 there, so the sign is safe).
  ⇒ on the grounded causal axis the bounded/on-distribution intervention is GENTLER, exactly the
  collapse-account prediction.
- **Correlational direction (diff-of-means):** clamp is EQUAL (b0.5 +0.02 [−0.13,+0.16]) or WORSE
  (b1.0 +0.18 [+0.07,+0.30]). No gentleness.
- **Interpretation:** "the swap is gentler because it targets a real on-distribution value" is true
  ONLY when the axis is grounded (its class-mean target is a genuine state). This ties the E10
  programme together: the SAME L17 direction that transfers backtracking best (E10.1) is the one
  where a bounded intervention is also safest (P2). Gentleness is a property of the grounded causal
  geometry, not of clamping per se.
- **Honesty caveat (sealed into the reading):** "gentle" = avoids the verbatim 4-gram loop
  specifically. das_clamp_b1.0 induces massive backtracking (cue 255/1k vs vanilla 10) and avoids
  loops (collapse 0.44) but boxed 0.00 and hits the 8192 cap ⇒ it prevents the degenerate-loop
  failure mode without preserving useful task completion. Lower repetition-collapse is NOT the same
  as preserved reasoning; report both.
