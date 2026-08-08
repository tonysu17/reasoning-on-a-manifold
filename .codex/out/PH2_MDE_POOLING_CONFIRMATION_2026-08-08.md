# Phase-2 MDE pooling-path confirmation

**Audit date:** 2026-08-08  
**Scope:** read-only audit of the current working-tree `ph2_mde_sim.py`; no simulation or model/API work was run.  
**Audited script SHA-256:** `4a4bdc1a331c42332e66a1b0755e20c9612248f76fc07a3c709219fb4f6d1f63`

## One-line handoff

**Confirmed for the primary sentence-fraction curves, with one code-path qualification:** `ph2_mde_sim.py` uses the arm-correct `single_direction -> energy_matched_random` floor and equal-weights all sample/subspace replicates within `base_task_id` before forming the 49 paired task profiles; its deterministic inputs match `src/delta_floor.py` exactly, and no legacy `_rates` extraction enters the curves. The qualification is that the simulator reimplements this extraction contract rather than directly importing `src.delta_floor.per_task_fraction`.

## Evidence

1. **The authoritative path defines the required contract.**
   - `src/delta_floor.py:16-28` fixes `single_direction -> energy_matched_random` and the task-level resampling unit after pooling `#s`/`#rs` replicates within `base_task_id`.
   - `src/delta_floor.py:67-80` implements that arm-to-floor mapping.
   - `src/delta_floor.py:100-125` begins the authoritative `per_task_fraction` extractor; `src/delta_floor.py:109-114` explicitly requires replicate pooling and unresolved missing annotations.
   - `src/delta_floor.py:324-356` applies the mapped floor, intersects arm/floor task IDs, and forms each task's `floor - arm` contrast.
   - `08_steering_analysis.py:32,169-176` shows that `delta_floor_report.json` is produced through `delta_floor_headline` from `src/delta_floor.py`.

2. **The simulator selects the arm-correct floor from that authoritative report.**
   - `ph2_mde_sim.py:265-276` reads the `backtracking|single_direction` cell and obtains its `alpha` and `floor`; it does not hard-code a legacy generic floor at extraction time.
   - `results/eval/R1-1.5B__E1/delta_floor_report.json:20-26` records `floor = "energy_matched_random"`, `alpha = 1.0`, `n_tasks = 49`, and `delta_floor = 0.05378923565449617`.
   - The generated sizing artifact repeats this selection at `.codex/out/ph2_mde_sim.json:29-36`.

3. **Replicates are pooled within task before profiles or curves are built.**
   - `ph2_mde_sim.py:72-104` collapses records to `base_task_id` (or strips the replicate suffix) while retaining every valid record in a per-task list.
   - `ph2_mde_sim.py:122-129` takes the equal-weight mean of the replicate values within each task.
   - `ph2_mde_sim.py:132-149` then intersects arm/floor task IDs and creates one paired profile per task.
   - Only after those task profiles exist are the primary curves simulated (`ph2_mde_sim.py:288-308`).

4. **The sentence-fraction input is fail-closed against the authoritative result.**
   - `ph2_mde_sim.py:279-286` requires the pooled mean task contrast to equal the authoritative `delta_floor_report.json` value to absolute tolerance `1e-12`, otherwise it raises.
   - A separate deterministic audit (no resampling) compared the simulator's task profiles with `src.delta_floor.per_task_fraction`: 49/49 task IDs matched; maximum absolute arm-value difference was `0.0`; maximum absolute floor-value difference was `0.0`; simulator, authoritative-function, and artifact means were all exactly `0.05378923565449617`.

5. **No `_rates`-style extraction leaked into the primary curves.**
   - Repository search finds `_rates` only in the untracked legacy `ph0_mde_firstcut.py:58-70`. That helper assigns a single row per `base_task_id` and therefore is not the pooling implementation used by `ph2_mde_sim.py`.
   - `ph2_mde_sim.py` neither imports nor calls `ph0_mde_firstcut.py` or `_rates`.
   - Its `legacy_first_cut` calibration receives the already-correct pooled `fraction_profiles` (`ph2_mde_sim.py:230-262,279-286`); only the calibration's variance assumptions are legacy. The calibration is explicitly excluded from the paired-design variance model in `ph2_mde_sim.py:372-380`.

## Qualification and disposition

- **Literal-path deviation:** the simulator does not call `src.delta_floor.per_task_fraction`; it independently implements equivalent extraction and pooling in `load_observations` / `_pooled_value_and_variance`. Current inputs are protected by exact deterministic equivalence and the fail-closed mean check, so this deviation did not alter the current primary sentence-fraction curves. It is nevertheless a maintenance risk if either extractor changes later.
- **Per-1k curve:** it uses the same arm-correct floor, task intersection, and within-task pooling, but `src/delta_floor.py` has no per-1k-token authoritative estimand and the exact `delta_floor_report` check applies only to sentence fraction. The simulator itself labels per-1k as a variance-reduced candidate rather than the E8 headline (`ph2_mde_sim.py:432-434`). Do not describe the per-1k curve as having been validated against the authoritative E8 estimand.
- **Interpretation:** Claude may use the paired-design **sentence-fraction** sizing curve as based on authoritative, arm-correct, task-pooled E8 inputs. Preserve the simulator's stated model-based variance and extrapolation caveats (`ph2_mde_sim.py:417-431`).

## Audit identity

- `src/delta_floor.py` SHA-256: `412aacff285c64d147f264c9a8600cbc4f72a3c3105978f1e69d89089a5d2d2a`
- `delta_floor_report.json` SHA-256: `a492604b572bbf40ca3531ffa5f65e210f11cea8e88e72d330eb75ac92e8e926`
- `.codex/out/ph2_mde_sim.json` SHA-256: `31efc58544c5b31d9000583091258fcc6d06b3903b913b946f3eaae45d7d2e0b`
- Analysis-repository HEAD at audit time: `5fd66637f34981c6e79109c6f0ecbee04d71ff5b` (the two MDE scripts and generated sizing artifacts are currently untracked, so their SHA-256 values—not HEAD alone—identify the audited versions).
