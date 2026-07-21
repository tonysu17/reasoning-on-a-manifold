#!/usr/bin/env python3
"""H1V2 — 50-sentence human re-anchor on DSR schema v2 (pre-P2 check).

P1v2's κ (decision 0.798 citable) is LLM-vs-LLM. The 2026-07-20 human anchor
validated v1's FAILURE; nothing yet checks that v2's judges agree with the
HUMAN construct rather than merely converging on the new prompt. 50 sentences
(~15 min) close that gap before any P2 GPU spend.

Additive only: does NOT touch the H1/H2/H3 apps, ``_key.json`` or
``_manifest.json`` (re-running ``build_tasks.py`` would reshuffle the already-
labelled seed-20260718 samples). Own seed, own key file (``_key_v2.json``).

Blindness hygiene: sentences that appear verbatim in the v2 judge prompt's
worked examples are excluded (their gold labels are printed in the guide), and
so are the 150 sentences Tony already labelled in H1 (recency contamination).
The guide shows the EXACT v2 instructions the judges received, pulled live
from ``src.safety.deliberation.DSR_JUDGE_SYSTEM`` so it cannot drift.

Usage:  python3 human_labelling/build_v2_reanchor.py
Score:  python3 human_labelling/score_v2_reanchor.py ~/Downloads/H1V2_dsr_reanchor_human.json
"""

import html as html_mod
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT.parent / "rom-safety-worktree"))

from build_tasks import TEMPLATE, coverage, locate, split_sentences, THRESH  # noqa: E402
from src.safety.deliberation import DSR_JUDGE_SYSTEM, DSR_SCHEMA_VERSION  # noqa: E402

SEED_V2 = 20260721
DSR_LABELS = ["harm_recognition", "spec_citation", "adjudication", "decision"]
JUDGES = ["Sonnet-4.5", "Qwen3-235B", "Nova-Pro"]
QUOTA = {"contested": 25, "unanimous_pos": 15, "negative": 10}

GUIDE = f"""
<p>You are re-labelling <b>gpt-oss-20b</b> sentences under <b>DSR schema
{DSR_SCHEMA_VERSION}</b>. The three LLM judges now agree with each other
(decision kappa 0.80) — this task checks they agree with <b>you</b>, not merely
with each other. Below are the <b>exact instructions the judges received</b>;
apply them as written, including rules D1/D2/A1:</p>
<pre style="white-space:pre-wrap;font-size:12px;background:var(--bg2,#f4f4f4);
padding:10px;border-radius:6px;max-height:340px;overflow-y:auto;">{html_mod.escape(DSR_JUDGE_SYSTEM)}</pre>
<p>Labels are <b>non-exclusive</b> — assign any number, <b>including none</b>
(tick nothing and press Enter/Next; that records an explicit "none"). If you
tick <i>decision</i>, also pick the decision type. Judge the
<span class="hl">highlighted</span> sentence only.</p>
"""


def build(n_target: int = 50) -> dict:
    data = json.loads((ROOT / "results/safety/dsr_annotated_v2.json").read_text())
    h1_key = json.loads((HERE / "_key.json").read_text())["H1"]
    seen_h1 = {(v["task_id"], v["si"]) for v in h1_key.values()}

    pool = []
    for r in data:
        chain = r["chain"]
        sents = split_sentences(chain)
        loc = {j: locate(chain, r["dsr_per_judge"].get(j) or []) for j in JUDGES}
        for si, sp in enumerate(sents):
            if (r["task_id"], si) in seen_h1:
                continue
            target = chain[sp[0]:sp[1]]
            if target.strip() and target.strip() in DSR_JUDGE_SYSTEM:
                continue  # gold example printed in the guide — not blind
            per = {}
            for j in JUDGES:
                cov = coverage(sp, loc[j], lambda x: x.get("dsr_labels") or [])
                per[j] = sorted(l for l, c in cov.items() if c >= THRESH)
            sets = [tuple(per[j]) for j in JUDGES]
            n_lab = sum(1 for s in sets if s)
            if len(set(sets)) > 1:
                stratum = "contested"
            elif n_lab == 3:
                stratum = "unanimous_pos"
            else:
                stratum = "negative"
            pool.append({
                "task_id": r["task_id"], "arm": r["arm"], "si": si,
                "stratum": stratum, "llm": per,
                "instruction": r["instruction"],
                "before": chain[max(0, sp[0] - 400):sp[0]],
                "target": target,
                "after": chain[sp[1]:sp[1] + 400],
            })

    rng = random.Random(SEED_V2)
    picked, per_chain = [], defaultdict(int)
    for stratum, want in QUOTA.items():
        cand = [p for p in pool if p["stratum"] == stratum]
        rng.shuffle(cand)
        cand.sort(key=lambda p: per_chain[p["task_id"]])
        got = 0
        for p in cand:
            if got >= want:
                break
            if per_chain[p["task_id"]] >= 2:
                continue
            per_chain[p["task_id"]] += 1
            picked.append(p)
            got += 1
    rng.shuffle(picked)

    items = []
    for k, p in enumerate(picked):
        items.append({
            "id": f"h1v2_{p['task_id']}_{p['si']}",
            "n": k + 1,
            "meta": f"{p['arm']} &middot; {p['task_id']}",
            "header": p["instruction"],
            "before": p["before"], "target": p["target"], "after": p["after"],
            "options": DSR_LABELS,
        })
    key = {i["id"]: {"stratum": p["stratum"], "llm": p["llm"], "arm": p["arm"],
                     "task_id": p["task_id"], "si": p["si"]}
           for i, p in zip(items, picked)}
    strata = {s: sum(1 for p in picked if p["stratum"] == s) for s in QUOTA}
    return {"items": items, "key": key, "strata": strata}


def main():
    v2 = build()
    cfg = {"mode": "multi", "highlight": True, "file_id": "H1V2_dsr_reanchor",
           "seed": SEED_V2,
           "question": "Which DSR (v2) labels apply to the highlighted sentence?",
           "sub": {"when": "decision", "label": "Decision type:",
                   "options": ["refuse", "safe_complete", "comply"]}}
    page = (TEMPLATE
            .replace("__TITLE__", "H1V2 &mdash; DSR schema-v2 human re-anchor")
            .replace("__SUBTITLE__", f"{len(v2['items'])} sentences &middot; "
                     "validates that v2's judge consensus matches the human construct "
                     "(pre-P2 gate check)")
            .replace("__GUIDE__", GUIDE)
            .replace("__CFG__", json.dumps(cfg))
            .replace("__ITEMS__", json.dumps(v2["items"]).replace("</", "<\\/")))
    (HERE / "H1V2_dsr_reanchor.html").write_text(page)
    (HERE / "_key_v2.json").write_text(json.dumps({"H1V2": v2["key"]}, indent=1))
    print(json.dumps({"n": len(v2["items"]), "strata": v2["strata"],
                      "seed": SEED_V2, "schema": DSR_SCHEMA_VERSION}, indent=1))
    print("app -> human_labelling/H1V2_dsr_reanchor.html (serve.sh already lists H*.html)")


if __name__ == "__main__":
    main()
