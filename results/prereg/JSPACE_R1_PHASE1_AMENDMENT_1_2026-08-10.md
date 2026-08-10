# J-space R1 pilot — Phase-1 pre-execution amendment 1

**Protocol marker:** amended  
**Evidence status:** prospective/unrun at sealing  
**Sealed UTC:** 2026-08-10T21:45:46Z  
**Parent protocol:** `results/prereg/JSPACE_R1_STEERING_PILOT_PREREG_2026-08-10.md`  
**Scope:** Phase 1 corpus, fitting, held-out stability, and external positive-control readout only. This amendment does not authorise or alter Phase 2 decomposition or causal interventions.

This amendment resolves execution ambiguities found after Phase 0 and before any Phase-1 fit or readout. It strengthens the validity gate used to decide whether the later L17 decomposition may run. No Phase-1 Jacobian, held-out rank, or external-evaluation rank existed when this document was sealed.

## 1. Frozen pre-execution inputs

| Input | SHA-256 |
|---|---|
| Parent protocol | `fa0011fe80ea03b84656b154b6e799b6d725f0c1fa2d459ed8027103517b67c8` |
| Corpus/fit manifest | `59c8695f4d8bc37e8a6cfb592bba09fc72afaec7f5eca6b415f443cb90defa6e` |
| External-evaluation eligibility manifest | `a193ce15ff18d1852a870703dba104763a01b06740c99ec8fd955e3a15927f7c` |
| Manifest generator | `102715f3c51549a1045b0396ac24d18bf4dc03cfb87c4d65da88d9bd5b33e027` |

The generator used Python 3.11.10, `datasets==4.8.5`, NumPy 1.26.3 and Transformers 5.5.0. It used the pinned R1 tokenizer with `add_bos_token=true`, matching the effective `jlens.from_hf(..., force_bos=True)` configuration. The dataset is `Salesforce/wikitext`, `wikitext-103-raw-v1`, train, revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Of 1,801,350 source rows, 449,301 nonblank rows had at least 128 effective tokens. NumPy `Generator(PCG64(20260810))` selected the registered first 120 permuted eligible source indices: 50 fit A, 50 fit B, and 20 held out.

The first-128-token prefixes are unique across all 120 rows. Before any readout, the generator compared them with the three external prompt sets and the fixed 50 causal tasks using exact token-prefix equality, normalised-text equality, and Jaccard overlap of five-token shingles. The registered near-overlap threshold is 0.80; none was flagged (maximum observed comparison 0.004149377593360996). A future duplicate or flagged overlap would stop execution rather than trigger post hoc resampling.

External eligibility is frozen before ranks. The eligible item counts are 98/102 association, 96/96 typo, and 81/93 multihop (82 of 103 multihop labels). Each scored `intermediates` label is encoded as one leading ASCII space plus the label, with special tokens disabled; exactly one R1 token is required. Ineligible labels are removed, an item remains only if at least one label is eligible, and each evaluation requires at least 50 eligible items. The multihop JSON `target` locates the readout and is not scored. All prompts are raw strings without a chat template and are at most 128 tokens; the readout is their final prompt token.

## 2. Fit and storage contract

- Fit A and B separately with source layers 0--26, target L27, `dim_batch=8`, `max_seq_len=128`, `skip_first=16`, eager attention, no compilation, and deterministic algorithms. Every manifest row's effective first 128 token IDs must equal `HFLensModel.encode` before fitting.
- Use fresh run-UUID-scoped checkpoint paths on container-local scratch, `checkpoint_every=10`, and `resume=False` on this first execution. Any restart is a new authorised run unless a wrapper first proves checkpoint model, corpus, code and estimator identity; the reference checkpoint alone is insufficient.
- Both fits must finish with exactly 50 successful prompts. Merge the in-memory FP32 A and B objects with released `JacobianLens.merge([A,B])`; never merge reloaded FP16 copies.
- Save A, B and merged lenses in both FP32 and FP16 using temporary files followed by load verification, hashing and atomic rename. The authoritative validity and any later decomposition lens is merged FP32. Output roots are unique and non-overwriting.
- For every layer, report relative Frobenius error. The merged FP32 matrix must agree with `(A+B)/2` to a maximum per-layer relative error of `1e-6`. FP32 save/load must be bitwise identical (and therefore at most `1e-6` relative error). FP16 storage is diagnostic and must have maximum per-layer relative error at most `1e-3` against the corresponding in-memory FP32 lens.

## 3. Held-out stability statistic and null

For held-out row (r), source layer (l), and every valid position (p\in\{16,\ldots,126\}), let (Q^A_{rlp}) and (Q^B_{rlp}) be the two top-25 token-ID sets. Top-k is ordered by decreasing logit and then increasing token ID; an exact tie at the rank-25 boundary uses the lower token IDs. Define

\[
 z_{rl}=\operatorname{median}_p\frac{|Q^A_{rlp}\cap Q^B_{rlp}|}{|Q^A_{rlp}\cup Q^B_{rlp}|},
 \qquad T_l=\operatorname{median}_r z_{rl}.
\]

The control has (B=1{,}000) draws from NumPy `Generator(PCG64(20260811))`. Each draw is one global random bijection of the 151,665 vocabulary IDs applied to the B-lens token IDs only; A is left fixed. For any layer vector (T), define the scan-adjusted statistic

\[
 S(T)=\max_{I}\min_{l\in I}T_l,
\]

where (I) ranges over all contiguous source-layer bands of length at least four containing L17. The same complete band scan is applied inside every permutation. The null p95 uses NumPy's `quantile(..., method="higher")`; the finite-sample p-value is \((1+\#\{S_b\geq S_{obs}\})/(B+1)\). The selected observed band breaks score ties by shortest length and then lowest starting layer.

This gate passes only if (S_{obs}>q_{.95}(S_b)), the empirical p-value is at most 0.05, and (S_{obs}\geq0.10). The last condition is the prospective absolute effect-size floor and prevents a near-zero vocabulary null from licensing a negligible overlap. A 10,000-resample row bootstrap with `PCG64(20260813)` reports 95% intervals for (T_l), but is not an additional threshold.

## 4. External positive-control statistic and strengthened L17 gate

The repository-comparable pass@25 remains the mean across eligible items of the within-item fraction of eligible intermediate labels appearing in the top 25 at any source layer. Because the planned decomposition is specifically at L17, the same pass@25 is also computed using L17 alone; the all-layer number cannot by itself validate L17.

For each evaluation, create 1,000 null draws using one NumPy `Generator(PCG64(20260812))` consumed in fixed order: association, typo, multihop. Each draw permutes whole eligible intermediate-label lists among the eligible prompts within that evaluation, preserving each list and using the recipient prompt's readout. The p95 and empirical p-value use the same `higher` and `>=` rules as the stability null. The vanilla logit lens is reported only as a comparator.

An evaluation is a qualifying success only when all of the following hold in that same evaluation:

1. merged all-layer pass@25 exceeds its permutation p95 with empirical (p\leq0.05);
2. merged L17 pass@25 exceeds its L17 permutation p95 with empirical (p\leq0.05);
3. fit A and fit B differ by at most 0.10 absolute pass@25 for the all-layer statistic; and
4. fit A and fit B differ by at most 0.10 absolute pass@25 at L17.

The external gate requires at least two of the three evaluations to be qualifying successes. Requiring the same evaluations removes the parent protocol's possible mismatch between the significance and A/B-agreement subsets.

## 5. Execution, blinding and consequence

The analysis code, its tests, this amendment, both manifests and their hashes must be copied and reverified on the network volume before launch. Unit and synthetic tests may be used while developing the scorer, but no implementation debugging may use the 20 held-out rows or the external readout ranks after the runner hash is sealed. A new code change after launch requires a new run UUID and amendment; outputs from a superseded attempt remain identifiable and are not silently overwritten.

Phase 1 passes only if the numerical/serialization gate, held-out stability gate, and strengthened external gate all pass. Failure of any one stops Phase 2 decomposition and licenses only: “the fitted lens did not pass the prespecified validity gate.” It does not show that J-space is absent, that the thesis's correlation-dimension result is wrong, or that the two estimands coincide.

Phase 0 measured 604.353 seconds for 100 Jacobian calls on the conforming RTX 4090. That is a projection for the fitting calls, not a bound on checkpoint I/O, held-out analysis, or the three external suites.
