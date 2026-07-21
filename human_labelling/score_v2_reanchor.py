#!/usr/bin/env python3
"""Score the H1V2 schema-v2 re-anchor export against the three v2 judges.

Reuses ``score_human.py``'s H1 machinery verbatim (per-DSR-label Cohen kappa,
human vs each judge, plus per-stratum breakdown) against ``_key_v2.json``.

Verdict rule (pre-registered, mirrors the F4 gates): `decision` human-vs-best-
judge kappa >= 0.6 -> v2 validated, P2 may proceed on all three citable labels;
0.4-0.6 -> v2 usable but geometry must replicate across judges' labelings;
< 0.4 -> the v2 LLM consensus does not match the human construct — STOP, the
0.798 was prompt-convergence, not validity.

Usage: python3 human_labelling/score_v2_reanchor.py ~/Downloads/H1V2_dsr_reanchor_human.json
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from score_human import load, score_h1  # noqa: E402


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    exp = load(Path(sys.argv[1]))
    if exp.get("file_id") != "H1V2_dsr_reanchor":
        sys.exit(f"expected file_id H1V2_dsr_reanchor, got {exp.get('file_id')!r}")
    key = json.loads((HERE / "_key_v2.json").read_text())["H1V2"]
    out = score_h1(exp, key)

    dec = out["per_label"]["decision"]["best_judge_kappa"]
    if dec >= 0.6:
        out["VERDICT"] = (f"v2 VALIDATED: decision human-vs-best-judge kappa {dec:.2f} "
                          "-> the 0.798 LLM consensus tracks the human construct; "
                          "P2 may proceed on the three citable labels.")
    elif dec >= 0.4:
        out["VERDICT"] = (f"v2 REPLICATE-TIER: decision kappa {dec:.2f} -> usable, but "
                          "any geometry must replicate across all three judges' labelings.")
    else:
        out["VERDICT"] = (f"v2 NOT VALIDATED: decision kappa {dec:.2f} < 0.4 -> the LLM "
                          "agreement is prompt-convergence, not construct validity; "
                          "STOP before P2.")

    print(json.dumps(out, indent=1))
    print("\nVERDICT:", out["VERDICT"])
    res = HERE / "v2_reanchor_results.json"
    res.write_text(json.dumps(out, indent=1))
    print("wrote", res)


if __name__ == "__main__":
    main()
