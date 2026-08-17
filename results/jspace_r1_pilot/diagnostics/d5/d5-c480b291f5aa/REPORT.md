# D5 — FP32 vs BF16 readout margins

Run `d5-c480b291f5aa`; sealed §7 + amendment 1 §2. BF16 forward, FP32 lens/norm/unembed.

| Suite | items | hits gained | hits lost | items w/ hit BF16→FP32 | union BF16→FP32 |
|---|---:|---:|---:|---:|---:|
| lens-eval-association | 98 | 6 | 4 | 4→3 | 0.0408→0.0306 |
| lens-eval-typo | 96 | 110 | 57 | 86→87 | 0.8958→0.9062 |
| lens-eval-multihop | 81 | 6 | 5 | 15→16 | 0.1852→0.1975 |

**Verdict:** {"association_item_delta": -1, "registered_expectation_met": true, "registered_expectation": "association union changes by at most +/-1 item", "consequence": "account (e) precision artefact REJECTED as an explanation of the association failure"}

