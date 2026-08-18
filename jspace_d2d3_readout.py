#!/usr/bin/env python3
"""D2 (scorer positive control) and D3 (small-end scale ladder).

Sealed protocol §4-§5 (seal 752dee7) + AMENDMENT 1 (RunPod venue).

Both stages are the same computation on different (model, hosted lens) pairs:
run the three pinned eval suites through OUR Phase-1 scoring path with a
hosted Neuronpedia lens, at the final prompt position, scoring the
``intermediates`` labels under a per-model tokenizer eligibility rule.

D2 pass rule (sealed): scorer exonerated iff >=2 of 3 suites have any-layer
union pass@25 > permutation p95 with p <= 0.05.
D3 is descriptive: does either ~1-2B model clear the same bar on association
or multihop? Its reading is FROZEN if D2 fails (sealed execution order).

Usage:
  python jspace_d2d3_readout.py --stage d2 --model Qwen/Qwen2.5-7B-Instruct \
      --revision <sha> --lens-repo neuronpedia/jacobian-lens \
      --lens-file qwen2.5-7b-it/jlens/Salesforce-wikitext/Qwen2.5-7B-Instruct_jacobian_lens.pt
  python jspace_d2d3_readout.py --stage d3 --model Qwen/Qwen3-1.7B --revision <sha> \
      --lens-path data/jlens_hosted/qwen3-1.7b_jacobian_lens.pt
"""
from __future__ import annotations

import argparse
import json
import os
import time
import uuid

import numpy as np
import torch

import jspace_diag_common as C


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["d2", "d3"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--lens-path", default=None, help="local .pt lens")
    ap.add_argument("--lens-repo", default=None, help="HF repo id hosting the lens")
    ap.add_argument("--lens-file", default=None, help="path inside the HF repo")
    ap.add_argument("--outroot", default="results/jspace_r1_pilot/diagnostics")
    ap.add_argument("--label", default=None, help="short cell name, e.g. qwen3-1.7b")
    ap.add_argument("--expected-layers", type=int, default=None)
    ap.add_argument("--expected-d-model", type=int, default=None)
    ap.add_argument("--expected-head-vocab", type=int, default=None)
    args = ap.parse_args()

    import jlens
    from jlens import JacobianLens
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    label = args.label or args.model.split("/")[-1].lower()
    run_uuid = f"{args.stage}-{label}-" + uuid.uuid4().hex[:12]
    outdir = os.path.join(args.outroot, args.stage, run_uuid)
    os.makedirs(outdir, exist_ok=True)

    # ---- lens ----
    if args.lens_path:
        lens_local = args.lens_path
        lens = JacobianLens.load(lens_local)
    else:
        from huggingface_hub import hf_hub_download
        lens_local = hf_hub_download(args.lens_repo, args.lens_file)
        lens = JacobianLens.load(lens_local)
    lens_sha = C.sha256_file(lens_local)

    # ---- model ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    hf = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
                                              dtype=torch.bfloat16).to(device)
    lens_model = jlens.from_hf(hf, tok)          # force_bos=True, as Phase 1
    source_layers = list(lens.source_layers)

    if args.expected_layers is not None:
        assert lens_model.n_layers == args.expected_layers, (
            f"expected {args.expected_layers} layers, got {lens_model.n_layers}")
    if args.expected_d_model is not None:
        assert lens_model.d_model == args.expected_d_model, (
            f"expected d_model={args.expected_d_model}, got {lens_model.d_model}")
    if args.expected_head_vocab is not None:
        head_vocab = int(lens_model._lm_head.weight.shape[0])
        assert head_vocab == args.expected_head_vocab, (
            f"expected head vocab={args.expected_head_vocab}, got {head_vocab}")

    if lens.d_model != lens_model.d_model:
        raise ValueError(f"lens d_model {lens.d_model} != model {lens_model.d_model}")
    bad = [l for l in source_layers if not 0 <= l < lens_model.n_layers]
    if bad:
        raise ValueError(f"lens source layers out of range for model: {bad}")

    domain = C.vocabulary_domain(lens_model)
    suite_items, suite_hashes = C.fetch_suites(os.path.join(outdir, "eval_files"))
    elig = C.derive_eligibility(lens_model, suite_items, domain)

    jac = C.jacobian_stack(lens, source_layers, device)
    rng = np.random.Generator(np.random.PCG64(C.PERM_SEED))

    results, ties_total = {}, {}
    for slug in C.SUITES:
        items = elig[slug]["items"]
        tops_lens, tops_logit, labels = [], [], []
        ties = {"lens": 0, "logit": 0}
        for i, item in enumerate(items):
            t_l, n1, _ = C.readout_item(lens_model, item["prompt"], jac, source_layers, domain)
            t_g, n2, _ = C.readout_item(lens_model, item["prompt"], None, source_layers, domain)
            tops_lens.append(t_l)
            tops_logit.append(t_g)
            labels.append(item["scored_token_ids"])
            ties["lens"] += n1
            ties["logit"] += n2
            if (i + 1) % 25 == 0:
                print(f"[{args.stage}:{label}] {slug} {i+1}/{len(items)}", flush=True)
        arr_lens = np.stack(tops_lens)
        arr_logit = np.stack(tops_logit)
        res = C.score_suite(arr_lens, labels, rng)
        res["logit_lens_comparator"] = {
            "any_layer_union_pass_at_25": __import__("jspace_phase1_scoring").pass_at_25(
                arr_logit, labels, layers=range(arr_logit.shape[1])),
        }
        res["n_eligible_items"] = elig[slug]["n_eligible_items"]
        res["n_eligible_labels"] = elig[slug]["n_eligible_labels"]
        res["below_minimum_eligible_flag"] = elig[slug]["below_minimum_eligible_flag"]
        results[slug] = res
        ties_total[slug] = ties
        np.savez_compressed(os.path.join(outdir, f"top25__{slug.replace('-', '_')}.npz"),
                            lens=arr_lens, logit=arr_logit,
                            labels=np.array([l[0] for l in labels], dtype=np.int64))
        print(f"[{args.stage}:{label}] {slug}: union={res['any_layer_union_pass_at_25']:.4f} "
              f"p95={res['permutation']['p95_higher']:.4f} p={res['permutation']['empirical_upper_p']:.4f} "
              f"qualifies={res['qualifies_instrument_level']}", flush=True)

    n_qual = sum(1 for r in results.values() if r["qualifies_instrument_level"])
    verdict = {
        "n_qualifying_suites": n_qual,
        "qualifying_suites": [s for s, r in results.items() if r["qualifies_instrument_level"]],
    }
    if args.stage == "d2":
        verdict["scorer_exonerated"] = bool(n_qual >= 2)
        verdict["sealed_pass_rule"] = ">=2 of 3 suites: union > permutation p95 and p <= 0.05"
        verdict["consequence"] = ("account (a) scorer/recipe defect REJECTED; D3 interpretable"
                                  if n_qual >= 2 else
                                  "scorer/recipe SUSPECT; D3 interpretation frozen per sealed order")
    else:
        verdict["note"] = ("descriptive; interpretation conditional on D2 pass. "
                           "Question: does this ~1-2B model clear the instrument-level bar "
                           "on association or multihop?")

    manifest = {
        "schema_version": f"rom-jspace-{args.stage}-hosted-readout-v1",
        "run_uuid": run_uuid, "stage": args.stage, "cell_label": label,
        "protocol": "results/prereg/JSPACE_R1_DIAGNOSTIC_PROTOCOL_2026-08-16.md",
        "seal_commit": "752dee7", "amendment": "JSPACE_DIAG_AMENDMENT_1_2026-08-17.md",
        "model": args.model, "model_revision": args.revision, "model_dtype": "bfloat16",
        "lens_source": args.lens_path or f"{args.lens_repo}::{args.lens_file}",
        "lens_sha256": lens_sha, "lens_n_prompts": lens.n_prompts,
        "lens_source_layers": source_layers, "lens_d_model": lens.d_model,
        "model_n_layers": lens_model.n_layers,
        "vocabulary_domain": domain,
        "tokenizer_add_bos_token": getattr(lens_model.tokenizer, "add_bos_token", None),
        "readout_position": "final_prompt_token", "max_seq_len": C.MAX_SEQ_LEN,
        "permutation_seed": C.PERM_SEED, "n_permutations": C.N_PERM,
        "eval_file_sha256_at_fetch": suite_hashes, "boundary_ties": ties_total,
        "scoring_module_sha256": C.sha256_file("jspace_phase1_scoring.py"),
        "readout_script_sha256": C.sha256_file("jspace_d2d3_readout.py"),
        "common_module_sha256": C.sha256_file("jspace_diag_common.py"),
        "jlens_commit": C.JLENS_COMMIT,
        "eligibility_counts": {s: {k: v for k, v in elig[s].items() if k != "items"}
                               for s in C.SUITES},
        "execution_run_id": os.environ.get("ROM_RUN_ID", ""),
        "execution_protocol_amendment": os.environ.get("ROM_PROTOCOL_AMENDMENT"),
        "env": C.environment(), "git_commit": C.git_head(),
        "finished_utc": C.utc_now(), "wall_seconds": round(time.time() - t0, 1),
    }
    report = {"results": results, "verdict": verdict}

    # Retain exact per-model item order and token eligibility.  Pairwise
    # item-overlap analyses must not assume two tokenizers share token IDs.
    json.dump(elig, open(os.path.join(outdir, "eligibility.json"), "w"),
              indent=1, sort_keys=True)

    md = [f"# {args.stage.upper()} — {label}", "",
          f"Run `{run_uuid}`; sealed protocol §{'4' if args.stage=='d2' else '5'}; "
          f"seal 752dee7 + amendment 1. Diagnostic-only, non-gating.", "",
          f"Model `{args.model}` @ `{args.revision}`; lens `{args.lens_path or args.lens_file}` "
          f"(n_prompts={lens.n_prompts}, layers {source_layers[0]}–{source_layers[-1]}); "
          f"scored vocab {domain['scored_vocab_size']} of head {domain['head_vocab_size']}.", "",
          "| Suite | n items | union pass@25 | perm p95 | p | qualifies | logit-lens union |",
          "|---|---:|---:|---:|---:|:--:|---:|"]
    for s, r in results.items():
        md.append(f"| {s} | {r['n_eligible_items']} | {r['any_layer_union_pass_at_25']:.4f} | "
                  f"{r['permutation']['p95_higher']:.4f} | "
                  f"{r['permutation']['empirical_upper_p']:.4f} | "
                  f"{'YES' if r['qualifies_instrument_level'] else 'no'} | "
                  f"{r['logit_lens_comparator']['any_layer_union_pass_at_25']:.4f} |")
    md += ["", f"**Verdict:** {json.dumps(verdict)}", ""]
    C.write_bundle(outdir, manifest, report, "\n".join(md) + "\n")
    print(f"[{args.stage}:{label}] DONE {run_uuid} -> {outdir}", flush=True)
    print(json.dumps(verdict, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
