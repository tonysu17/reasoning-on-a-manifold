# J-space R1 pilot — Phase-1 post-completion validation amendment 5

**Protocol marker:** amended

**Scientific-run status:** runner complete; overall gate recorded negative

**Integrity status at sealing:** independent validation incomplete

**Validation continuation status:** prospective/unrun

**Sealed UTC:** 2026-08-10T23:26:55Z

**Parent amendment:** `results/prereg/JSPACE_R1_PHASE1_AMENDMENT_4_2026-08-10.md` (`fbf7787871b70757f418799d49a0eb5f085cf4eb5cd35bfaf4b9d4b6fa64efcd`)

**Scientific run UUID:** `jspace-p1-20260810T224338Z-996077031021`

**Validation attempt UUID:** `jspace-p1v-20260810T231952Z-6ad3de6f3754`

**Scientific runner commit:** `54a6538212091961d906b738f2ccc83c2f58f818`

**Recovery implementation commit:** `d8d2a04c45d04dd8e8e4f280eb19b3520127cab9`

## 1. Completed scientific computation and bounded outcome

The A4 runner completed both 50-prompt fits, numerical/serialization checks, all 20 held-out readouts and their registered null, all 275 eligible external readouts and all registered external nulls. It recorded numerical/serialization pass, held-out-stability pass, external-positive-control non-pass, and overall scientific-gate non-pass. The registered consequence is to stop before Phase 2 and license only: “the fitted lens did not pass the prespecified validity gate.” This does not show that J-space is absent or merge this result with the thesis's geometric estimands.

The runner result is not treated as integrity-valid at sealing because the independent validator terminated before completing its full attestation. The negative gate was visible before this amendment. No scientific rule may be changed in response.

## 2. Validator-only failure and correction

The first validator regenerated all three external null pairs in fixed RNG order and recomputed all three external reports. It then entered a second metadata loop. That loop reused Python's stale `eligible_items` variable from the final multihop iteration, so it compared the association report's correct ordered 98-name list against the multihop 81-name list and stopped. The runner's stored lists exactly match the sealed eligibility manifest for association, typo and multihop.

The correction replaces the stale variable with a slug-indexed lookup of eligible items in frozen manifest order. It changes no model forward, lens, top-25 ID, rank, label, null array, score, threshold, gate or scientific report. Regression tests use disjoint per-slug names, filter ineligible items, preserve order and reject reordered or corrupted names.

## 3. Frozen source artefacts

The validation continuation must treat the following A4 files as immutable and verify their exact inventory and SHA-256 before validation, after validation and again before terminal sealing:

| File | SHA-256 |
|---|---|
| `INDEPENDENT_VALIDATION.json` | `88988d7c717f24db35b0adc02478e0f38fa437295cec75def831d414b4c6f60d` |
| `RUN_ENVIRONMENT.txt` | `10cf842ff55e803d1bf83b0edfc7494dd9d8720beb5b35590c1f6c1704c0e9b7` |
| `external_positive_controls.json` | `f86e83dbaae81cbe748e81d6684f63cd6c106c0911820fc9a768c27156de7ed2` |
| `external_readout_arrays.npz` | `02ee47f173b034db744758ac915460d1bd7ed5c347bd4022ca97dd95760c6289` |
| `heldout_stability.json` | `f2c19d5efd9fba0ed18e9d84adcb8bbef7de75c36fc61fc075afd8b267bf246a` |
| `heldout_stability_arrays.npz` | `e45634c7a611454695d0207d9b4c97280aa217fba21bedbff3ee8053b02585f1` |
| `lenses/fit_a.fp16.pt` | `a3c85e27d8c2d2b4cb669fb17a01e656d66b9655e7d92b7b350d4ecd14c388e8` |
| `lenses/fit_a.fp32.pt` | `6874f0ac5854bb877a4972da95367e523fc349f2f4f6dcf36c459f3a60a50aa7` |
| `lenses/fit_b.fp16.pt` | `75ea473ca2784779afbf38897857295af0a21b9befe4130a4c19cd6bb356de1c` |
| `lenses/fit_b.fp32.pt` | `b00a2b4e339317a7a8aa1e246bf006eed5b1f749f491fc91bd2ac031282195cc` |
| `lenses/merged.fp16.pt` | `0bb262d333927fd3816b1f5c5399ce2347875c11687c73ab8a729dd3365e42c9` |
| `lenses/merged.fp32.pt` | `6b5f1043b3c3fa3fcd8d4d69919772a2d763c145f996477bf715fe4b9b89f672` |
| `numerical_validation.json` | `7497d8430855a752f420ed5cc4afb81f776adc197d9ed2923c514be0a3aaaaea` |
| `phase1_report.json` | `004145b129f289972e740ad23e20084eb0a9109ef0d89383645f3fc269fc35d9` |
| `phase1_runner.log` | `52830b815699a3f4081ebc1f02604792e83c90110374fef8dc237649fb25c828` |

The source runtime's append-only `FAILED.json` is separately pinned to `8c6dccd33b48b55b8ace8ca2a55e45eaf1b55d8d72a9db44e2df1181f9eafb5f`. The failed V1 attestation remains preserved and non-authoritative.

## 4. Validation-only continuation

No fit, model forward, activation capture, unembedding, rank extraction or scientific-output generation is rerun. A separate validation runtime reruns the complete independent validator against the immutable files, including numerical/lens checks, the 1,000-draw CUDA stability replay, held-out report recomputation, all three 1,000-draw external-null replays in fixed order, all external report comparisons, and overall-gate/consequence recomputation.

The complete registered check-key set must be present and true; missing or extra keys fail. The corrected attestation is written append-only at `validation_attempts/jspace-p1v-20260810T231952Z-6ad3de6f3754/INDEPENDENT_VALIDATION.json`. Only this V2 path is authoritative; V1 remains in the bundle as failed-attempt provenance.

The recovery controller may publish `ARTIFACT_MANIFEST.json` and `DONE.json` only after exact pre/post source hashes, source failure identity, fixed implementation hashes, source execution-manifest identity, corrected attestation identity and the complete check set all pass. `DONE.scientific_gate_pass` must remain equal to both the immutable runner report and independent recomputation.

## 5. Frozen recovery files

| File | SHA-256 |
|---|---|
| `jspace_phase1_validate.py` | `f80570ea2f2118b2a21861b07886d22c215f23697ce7a981edbba310a316f590` |
| `jspace_phase1_revalidate.py` | `04b987d71eb5d1f0a1747bd25db6290d413dfc12093094c5c7f94e4ed6fc41c6` |
| `.codex/out/jspace_phase1_finish_watch.py` | `4e30bff98d95f0da9a82331636ad6cf240618e2827698549de88fa93614cb670` |
| `tests/test_jspace_phase1_validation.py` | `11e616895eb896c31bff81abe2b8cd20645ac41133cbb74bf96972bd025b2429` |
| `tests/test_jspace_phase1_revalidate.py` | `abf2b5df555afadc929669ea6408d5c16f8ee98bceb1425ad4ca334c0a5a0453` |
| `tests/test_jspace_phase1_finish_watch.py` | `0e05eb7d68c97697451bdf8f50b48058cbe26de89bfd741f01d62c3405bed664` |

Twenty-seven relevant tests pass under the local `/usr/bin/python3` environment. The watcher must follow the separate validation runtime, pull the original scientific run plus both validation attempts, rerun the corrected semantic validator locally, require Boolean equality of the gate in `DONE`, the runner report and local recomputation, and rehash the complete promoted bundle immediately before termination. It also rechecks the remote manifest and `DONE` hashes and the exact RunPod API identity before the single authorised termination mutation.

The success notification is explicitly titled “Laptop ready to close” and may fire only after local verification and confirmed termination of literal Pod `bghqqjkco3q7r5`. Any failure holds the Pod.
