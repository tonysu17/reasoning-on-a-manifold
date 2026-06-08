#!/usr/bin/env python3
"""
Quality check on generated reasoning chains.

Reports on:
  1. Structural integrity (missing fields, duplicates, errors)
  2. Category distribution (vs expected 100/category)
  3. Token-length distribution per category
  4. Truncation analysis (chains hitting max_tokens, missing </think>)
  5. Prompt integrity (Phase 4 needs prompt+chain to match generation exactly)
  6. Content anomalies (non-ASCII, repetition loops, language drift)
  7. Sample chain openings (qualitative spot-check)
  8. Token count histogram

Writes a Markdown report to results/quality_reports/<stem>.md and a JSON
summary to results/quality_reports/<stem>.json.

Usage:
  python check_chain_quality.py                              # default: data/chains_R1-1.5B.json
  python check_chain_quality.py --in data/annotated_R1-1.5B.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def quality_check(chains, chain_field="chain", tokens_field="n_tokens",
                  max_tokens=8192, expected_per_category=100):
    """Return a dict with all quality metrics. Pure function, no I/O."""
    n = len(chains)
    out = {"n_chains": n, "max_tokens": max_tokens}

    required = {"task_id", "category", "instruction", "prompt", chain_field,
                "full_text", tokens_field}
    out["integrity"] = {
        "missing_fields":    sum(1 for c in chains if not required.issubset(c)),
        "empty_chain":       sum(1 for c in chains if not c.get(chain_field, "").strip()),
        "empty_prompt":      sum(1 for c in chains if not c.get("prompt", "").strip()),
        "has_error_field":   sum(1 for c in chains if c.get("error")),
        "zero_tokens":       sum(1 for c in chains if c.get(tokens_field, 0) == 0),
        "dup_task_ids":      n - len({c["task_id"] for c in chains}),
        "full_text_mismatch":sum(1 for c in chains
                                  if c["full_text"] != c["prompt"] + c[chain_field]),
    }

    by_cat = Counter(c["category"] for c in chains)
    out["category_distribution"] = {
        cat: {"count": cnt, "delta_vs_expected": cnt - expected_per_category}
        for cat, cnt in by_cat.most_common()
    }

    by_cat_tokens = defaultdict(list)
    for c in chains:
        by_cat_tokens[c["category"]].append(c.get(tokens_field, 0))

    out["token_lengths"] = {}
    for cat in sorted(by_cat_tokens):
        toks = by_cat_tokens[cat]
        n_trunc = sum(1 for t in toks if t >= max_tokens)
        out["token_lengths"][cat] = {
            "mean":   round(statistics.mean(toks)),
            "median": round(statistics.median(toks)),
            "min":    min(toks),
            "max":    max(toks),
            "pct_at_max": round(100 * n_trunc / len(toks), 1),
        }
    all_toks = [c[tokens_field] for c in chains]
    out["token_lengths_overall"] = {
        "mean":   round(statistics.mean(all_toks)),
        "median": round(statistics.median(all_toks)),
        "min":    min(all_toks),
        "max":    max(all_toks),
        "pct_at_max": round(100 * sum(1 for t in all_toks if t >= max_tokens) / n, 1),
    }

    n_trunc       = sum(1 for c in chains if c[tokens_field] >= max_tokens)
    n_closed      = sum(1 for c in chains if "</think>" in c.get(chain_field, ""))
    both          = sum(1 for c in chains
                        if c[tokens_field] >= max_tokens
                        and "</think>" not in c[chain_field])
    trunc_closed  = sum(1 for c in chains
                        if c[tokens_field] >= max_tokens
                        and "</think>" in c[chain_field])
    short_no_close= sum(1 for c in chains
                        if c[tokens_field] < max_tokens
                        and "</think>" not in c[chain_field])
    short_closed  = sum(1 for c in chains
                        if c[tokens_field] < max_tokens
                        and "</think>" in c[chain_field])
    out["truncation"] = {
        "n_at_max":                   n_trunc,
        "pct_at_max":                 round(100 * n_trunc / n, 1),
        "n_with_closing_think":       n_closed,
        "pct_with_closing_think":     round(100 * n_closed / n, 1),
        "n_without_closing_think":    n - n_closed,
        "max_AND_no_think_close":     both,
        "max_AND_think_close":        trunc_closed,
        "short_AND_no_think_close":   short_no_close,
        "short_AND_think_close":      short_closed,
    }

    prompt_lengths = [len(c["prompt"]) for c in chains]
    distinct_headers = Counter(c["prompt"][:50] for c in chains)
    out["prompts"] = {
        "length_mean":   round(statistics.mean(prompt_lengths)),
        "length_median": round(statistics.median(prompt_lengths)),
        "length_min":    min(prompt_lengths),
        "length_max":    max(prompt_lengths),
        "n_distinct_template_headers": len(distinct_headers),
        "n_with_think_tag":            sum(1 for c in chains
                                            if "<think>" in c["prompt"]),
    }

    out["anomalies"] = {
        "non_ascii_in_first_1000":  sum(1 for c in chains
                                         if any(ord(ch) > 127
                                                for ch in c[chain_field][:1000])),
        "contains_cjk":             sum(1 for c in chains
                                         if re.search(r"[一-鿿]",
                                                       c[chain_field])),
        "contains_kana":            sum(1 for c in chains
                                         if re.search(r"[぀-ヿ]",
                                                       c[chain_field])),
        "repetition_loops":         sum(1 for c in chains
                                         if re.search(r"(.{20,100})\1{5,}",
                                                       c[chain_field][:5000])),
    }

    random.seed(42)
    samples = random.sample(chains, min(3, n))
    out["samples"] = [{
        "task_id":     c["task_id"],
        "category":    c["category"],
        "n_tokens":    c[tokens_field],
        "instruction": c["instruction"][:200] + ("..." if len(c["instruction"]) > 200 else ""),
        "chain_start": c[chain_field][:300].replace("\n", " / "),
    } for c in samples]

    edges = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000,
             max_tokens, max_tokens + 1]
    labels = ["<1k", "1-2k", "2-3k", "3-4k", "4-5k", "5-6k", "6-7k", "7-8k",
              f"8-{max_tokens/1000:.1f}k", f"{max_tokens}"]
    hist = [0] * len(labels)
    for t in all_toks:
        for i in range(len(edges) - 1):
            if edges[i] <= t < edges[i + 1]:
                hist[i] += 1
                break
    out["histogram"] = dict(zip(labels, hist))

    return out


def render_markdown(report, source_file):
    """Render the report as a Markdown document for the paper appendix."""
    lines = []
    lines.append(f"# Chain quality report: `{source_file}`")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Total chains: {report['n_chains']}")
    lines.append(f"Max-tokens setting: {report['max_tokens']}")
    lines.append("")

    lines.append("## 1. Structural integrity")
    lines.append("")
    lines.append("| Check | Count |")
    lines.append("|---|---|")
    for k, v in report["integrity"].items():
        flag = "OK" if v == 0 else "WARN"
        lines.append(f"| {k.replace('_', ' ')} | {v} ({flag}) |")
    lines.append("")

    lines.append("## 2. Category distribution")
    lines.append("")
    lines.append("| Category | Count | Delta vs expected |")
    lines.append("|---|---|---|")
    for cat, d in report["category_distribution"].items():
        delta = d["delta_vs_expected"]
        sign = "" if delta == 0 else (f"+{delta}" if delta > 0 else f"{delta}")
        lines.append(f"| {cat} | {d['count']} | {sign} |")
    lines.append("")

    lines.append("## 3. Token-length distribution per category")
    lines.append("")
    lines.append("| Category | mean | median | min | max | % at max |")
    lines.append("|---|---|---|---|---|---|")
    for cat in sorted(report["token_lengths"]):
        t = report["token_lengths"][cat]
        lines.append(f"| {cat} | {t['mean']} | {t['median']} | {t['min']} | "
                     f"{t['max']} | {t['pct_at_max']}% |")
    t = report["token_lengths_overall"]
    lines.append(f"| **OVERALL** | **{t['mean']}** | **{t['median']}** | "
                 f"**{t['min']}** | **{t['max']}** | **{t['pct_at_max']}%** |")
    lines.append("")

    lines.append("## 4. Truncation analysis")
    lines.append("")
    tr = report["truncation"]
    lines.append(f"- Chains at max_tokens: **{tr['n_at_max']} ({tr['pct_at_max']}%)**")
    lines.append(f"- Chains with closing `</think>`: {tr['n_with_closing_think']} ({tr['pct_with_closing_think']}%)")
    lines.append(f"- Chains *without* closing `</think>`: **{tr['n_without_closing_think']}** (likely truncated)")
    lines.append("")
    lines.append("Cross-tabulation:")
    lines.append("")
    lines.append("| Status | Count |")
    lines.append("|---|---|")
    lines.append(f"| Hit max AND no closing think (truncated mid-thinking) | {tr['max_AND_no_think_close']} |")
    lines.append(f"| Hit max AND has closing think (ended right at limit) | {tr['max_AND_think_close']} |")
    lines.append(f"| Short AND no closing think (terminated some other way) | {tr['short_AND_no_think_close']} |")
    lines.append(f"| Short AND has closing think (clean finish) | {tr['short_AND_think_close']} |")
    lines.append("")

    lines.append("## 5. Prompt integrity")
    lines.append("")
    p = report["prompts"]
    lines.append(f"- Prompt length: mean={p['length_mean']}, median={p['length_median']}, range=[{p['length_min']}, {p['length_max']}]")
    lines.append(f"- Distinct template headers: {p['n_distinct_template_headers']}")
    lines.append(f"- Prompts containing `<think>` tag: {p['n_with_think_tag']} of {report['n_chains']}")
    lines.append("")

    lines.append("## 6. Content anomalies")
    lines.append("")
    lines.append("| Check | Count |")
    lines.append("|---|---|")
    for k, v in report["anomalies"].items():
        lines.append(f"| {k.replace('_', ' ')} | {v} |")
    lines.append("")

    lines.append("## 7. Token-count histogram")
    lines.append("")
    lines.append("```")
    max_h = max(report["histogram"].values()) if report["histogram"] else 1
    for lbl, h in report["histogram"].items():
        bar = "#" * int(40 * h / max_h)
        lines.append(f"  {lbl:>7s} ({h:>4d}) {bar}")
    lines.append("```")
    lines.append("")

    lines.append("## 8. Sample chain openings (random seed=42)")
    lines.append("")
    for s in report["samples"]:
        lines.append(f"### `{s['task_id']}`  ({s['category']}, n_tokens={s['n_tokens']})")
        lines.append("")
        lines.append(f"**Instruction:** {s['instruction']}")
        lines.append("")
        lines.append(f"**Chain start:** {s['chain_start']}...")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path,
                        default=Path("data/chains_R1-1.5B.json"))
    parser.add_argument("--out", dest="out_path", type=Path, default=None,
                        help="Markdown output (default: results/quality_reports/<stem>.md)")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--expected-per-category", type=int, default=100)
    parser.add_argument("--chain-field", default="chain")
    parser.add_argument("--tokens-field", default="n_tokens")
    args = parser.parse_args()

    with open(args.in_path) as f:
        chains = json.load(f)

    report = quality_check(
        chains,
        chain_field=args.chain_field,
        tokens_field=args.tokens_field,
        max_tokens=args.max_tokens,
        expected_per_category=args.expected_per_category,
    )

    md = render_markdown(report, source_file=str(args.in_path))

    if args.out_path is None:
        args.out_path = Path("results/quality_reports") / f"{args.in_path.stem}.md"
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(md)

    json_path = args.out_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Markdown: {args.out_path}")
    print(f"JSON:     {json_path}")


if __name__ == "__main__":
    main()
