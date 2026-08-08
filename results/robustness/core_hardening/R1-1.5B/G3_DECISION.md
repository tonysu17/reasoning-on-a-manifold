# G3 core-geometry decision

Decision freeze: 2026-07-27  
Model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`  
Registered representation: mean-pooled, unclipped, common zero-based L27  
Scientific unit: reasoning chain

## Decision

**G3 classification: narrow outcome.**

The preregistered descriptive low-cdim criterion passes, but the direct cdim
specificity family fails. The composite chain/truncation robustness family
also fails, and the matched pooling/window family is unrun because its six
aligned inputs are absent.

The permitted central wording is:

> Low estimated dimension within annotation-indexed clouds.

Specificity remains unsupported by the direct cdim contract. The result must
not be described as a six-dimensional linear subspace, behaviour-specific
activation geometry, pooling/window robust, or globally flat.

## Core hypotheses

| Gate | Decision | Sealed result |
|---|---|---|
| H1: descriptive low cdim | **PASS** | All four common-L27 equal-chain estimates are finite, positive, and at most 10: backtracking 7.207893; uncertainty estimation 7.364468; example testing 7.040283; adding knowledge 8.251715. |
| H2: direct within-chain specificity | **FAIL** | Raw lower-tail \(p\)-values are 0.842463, 1.000000, 0.894842, and 1.000000; all four Holm-adjusted values are 1.000000. A complete diagnostic-seed repetition gives the same four failures. |
| H3: chain/truncation robustness | **FAIL** | The chain leg passes for three behaviours and fails for adding knowledge. All four category-imbalance triggers fire; every matched truncation leg fails because the complete/truncated PR bands do not overlap. The registered chain/truncation robustness criterion was not met. |
| H4: matched pooling/window robustness | **UNRUN** | The required aligned mean/first/last × clipped/unclipped representation matrices, metadata, and row indices are absent. No replacement activations or reduced-grid substitute were generated. |

The primary H2 family contains 20,000/20,000 valid null draws. Every required
H3 stability component contains 500/500 valid draws.

## Secondary families

- Five-depth direct cdim family: **FAIL**. None of 20
  behaviour-by-depth cells passes Holm correction. The smallest raw
  \(p=.009196\) becomes \(p_{\mathrm{Holm}}=.183926\).
- Three-annotation direct cdim family: **FAIL**. None of 24 cells passes;
  every Holm-adjusted \(p\)-value is 1.000000.
- Registered common-L16 curvature diagnostic: **MIXED**. The bounded negative
  operator passes for uncertainty estimation and example testing. It does not
  pass for backtracking, whose chain-ratio band is entirely above one, or
  adding knowledge, whose band is entirely below one; both departures exceed
  their row-control analogues. This is not a global-flatness conclusion.

## Sealed artefacts

| Artefact | SHA-256 |
|---|---|
| `primary_L27_cdim_null.json` | `99f9aec51af8447bdb576d46513da51db7418ad13354148d53226819f17abbd1` |
| `five_depth_cdim_null.json` | `7122d85cb9b8d3f75f18ba84da3b33bb1c3b473302a4bc61710fd079caf8f369` |
| `three_annotator_cdim_null.json` | `502e00f2862a0fc74f50f8997ce5fab3419ab803572bc21d18740159d442c4ed` |
| `chain_stability.json` | `5d067729be2c55d05ab9ec4848ea9076d8008e22bf4328d51fe437d3d1ceb48f` |
| `truncation_sensitivity.json` | `e0ee2090a2ee4a3a02bb90d62a1c1908d439bc2c7186bdd4297127c14b7fd929` |
| `curvature_L16.json` | `9697654295e8f6a533aa2f523a264d2fdd45ddab2dd13be4adf84cf23a21262f` |
| `provenance.json` | `38a42bd3ddac55fb1c707a5ea8dd7f9ba874f7f436d405e4a04cbb8311fc0e06` |
| preregistration | `27d266e58d84c48946caf64c0ce9312a44528f3f20a9a3a55117128ff21085a5` |
| analysis harness | `753a2a2b75959d27f1c64446b6bc307f1f3a7ebd1cc5055d0148a1263dabc1c2` |
| result-cell schema | `7342997425a708e102ee0f71246f72784736c6770f0e66f16869659fe3d9442b` |
| harness tests | `62b1f98987e551035969c6ace3b56e28d05c0a02acb640bda8621f9be79606a9` |

Individual result envelopes retain their exact execution argv, start/end
times, runtime, peak RSS, input hashes, amendment identifiers, and cell-level
provenance. The aggregate `provenance.json` is a sealed-output reporting
manifest; `code_commit` remains `null` because no git operation was performed.

## Thesis integration

The canonical Abstract, Introduction, Methods, Results, Steering, and
Conclusion now use this hierarchy. The rebuilt 61-page PDF is
`thesis/ucl_msc.pdf`, SHA-256
`675ec9531e25ef4343a2b195458a27e56527155f44cc9b68a73e96a6df3dcf0f`.
The build completed without an overfull box, undefined-reference, or fatal
LaTeX error; the remaining underfull-box warnings occur in pre-existing
Steering, Safety, and bibliography content.
