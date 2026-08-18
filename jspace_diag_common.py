#!/usr/bin/env python3
"""Shared utilities for J-space diagnostics D2-D5.

Sealed protocol: results/prereg/JSPACE_R1_DIAGNOSTIC_PROTOCOL_2026-08-16.md
(seal commit 752dee7) + AMENDMENT 1 (RunPod venue, D5 mechanics).

Scoring primitives are imported UNMODIFIED from jspace_phase1_scoring (the
Phase-1 fixed input, sha256 6f610e61...). This module only generalises the
Phase-1 readout path across models: per-model tokenizer eligibility, per-model
vocabulary domain, and final-prompt-position readout.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import subprocess
import urllib.request
from typing import Any

import numpy as np
import torch

import jspace_phase1_scoring as scoring

JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
RAW_BASE = f"https://raw.githubusercontent.com/anthropics/jacobian-lens/{JLENS_COMMIT}/data/evaluations"
SUITES = ("lens-eval-association", "lens-eval-typo", "lens-eval-multihop")
MAX_SEQ_LEN = 128          # Phase-1 fixed input
PERM_SEED = 20260817       # sealed protocol §4/§5
N_PERM = 1000
K = 25
MIN_ELIGIBLE_ITEMS = 50    # sealed reporting flag (not a gate)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    # Lightweight pod payloads intentionally omit .git.  The launcher binds
    # the payload to its clean source commit through ROM_GIT_COMMIT.
    return head or os.environ.get("ROM_GIT_COMMIT", "")


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def fetch_suites(cache_dir: str) -> tuple[dict[str, list[dict]], dict[str, str]]:
    """Download the three pinned eval files; return items and sha256-at-fetch."""
    os.makedirs(cache_dir, exist_ok=True)
    items, hashes = {}, {}
    for slug in SUITES:
        fn = f"{slug}.json"
        path = os.path.join(cache_dir, fn)
        if os.path.exists(path):
            raw = open(path, "rb").read()
        else:
            raw = urllib.request.urlopen(f"{RAW_BASE}/{fn}", timeout=60).read()
            open(path, "wb").write(raw)
        hashes[fn] = hashlib.sha256(raw).hexdigest()
        items[slug] = json.loads(raw)["items"]
    return items, hashes


def vocabulary_domain(lens_model: Any) -> dict[str, int]:
    """Registered vocabulary domain for a model: head rows vs tokenizer IDs.

    The scored domain is the contiguous tokenizer prefix of the output head
    (Phase-1 convention; Qwen-family heads carry padding rows, and some
    tokenizers report one more entry than the head provides).
    """
    head = int(lens_model._lm_head.weight.shape[0])
    tok_len = int(len(lens_model.tokenizer))
    scored = min(tok_len, head)
    if scored <= K:
        raise ValueError(f"scored vocabulary too small: {scored}")
    return {"head_vocab_size": head, "tokenizer_len": tok_len,
            "scored_vocab_size": scored,
            "head_rows_excluded": head - scored,
            "tokenizer_ids_beyond_head": max(0, tok_len - head)}


def derive_eligibility(lens_model: Any, suite_items: dict[str, list[dict]],
                       domain: dict[str, int]) -> dict[str, Any]:
    """Re-derive Phase-1 label eligibility under this model's tokenizer.

    Contract (Phase-1 scoring_contract, unchanged): scored field is
    ``intermediates``; scored form is one leading ASCII space + label encoded
    with ``add_special_tokens=False``; eligible iff exactly one token, that ID
    lies in the scored vocabulary domain, and it is not a special token.
    """
    tok = lens_model.tokenizer
    specials = set(int(i) for i in (tok.all_special_ids or []))
    scored_max = domain["scored_vocab_size"]
    out: dict[str, Any] = {}
    for slug, items in suite_items.items():
        eligible_items = []
        for item in items:
            labels = []
            for label in item["intermediates"]:
                ids = tok(" " + label, add_special_tokens=False)["input_ids"]
                ok = (len(ids) == 1 and int(ids[0]) < scored_max
                      and int(ids[0]) not in specials)
                labels.append({"label": label, "scored_form": " " + label,
                               "token_ids": [int(i) for i in ids],
                               "eligible": bool(ok)})
            keep = [lb for lb in labels if lb["eligible"]]
            if keep:
                eligible_items.append({"name": item["name"], "prompt": item["prompt"],
                                       "labels": labels,
                                       "scored_token_ids": [lb["token_ids"][0] for lb in keep]})
        out[slug] = {
            "items": eligible_items,
            "n_eligible_items": len(eligible_items),
            "n_eligible_labels": sum(len(i["scored_token_ids"]) for i in eligible_items),
            "n_source_items": len(items),
            "below_minimum_eligible_flag": len(eligible_items) < MIN_ELIGIBLE_ITEMS,
        }
    return out


def top25_from_logits(full_logits: torch.Tensor, domain: dict[str, int]) -> tuple[np.ndarray, int]:
    """Sealed deterministic top-25 over the scored vocabulary domain."""
    logits = scoring.tokenizer_vocabulary_logits(
        full_logits, head_vocab_size=domain["head_vocab_size"],
        tokenizer_vocab_size=domain["scored_vocab_size"])
    ids, ties = scoring.deterministic_topk_ids(logits, k=K)
    return ids.to(dtype=torch.int32).cpu().numpy(), ties


def capture_final_residual(lens_model: Any, prompt: str, source_layers: list[int]
                           ) -> tuple[torch.Tensor, list[int]]:
    """Residuals at the final prompt position: [n_source_layers, 1, d_model]."""
    from jlens.hooks import ActivationRecorder

    input_ids = lens_model.encode(prompt, max_length=MAX_SEQ_LEN)
    with torch.no_grad(), ActivationRecorder(lens_model.layers, at=source_layers) as rec:
        lens_model.forward(input_ids)
        acts = torch.stack([rec.activations[l][0].detach() for l in source_layers], dim=0)
    pos = input_ids.shape[1] - 1
    return acts[:, pos:pos + 1, :], input_ids[0].cpu().tolist()


def readout_item(lens_model: Any, prompt: str, jac_stack: torch.Tensor | None,
                 source_layers: list[int], domain: dict[str, int],
                 unembed=None) -> tuple[np.ndarray, int, list[int]]:
    """Top-25 IDs per source layer at the final prompt position.

    ``jac_stack`` None gives the logit-lens comparator (no transport).
    ``unembed`` overrides the model's unembedding (D5 FP32 path).
    """
    residual, ids = capture_final_residual(lens_model, prompt, source_layers)
    x = residual.float()
    if jac_stack is not None:
        x = torch.bmm(x, jac_stack.transpose(1, 2))
    emb = unembed if unembed is not None else lens_model.unembed
    logits = emb(x.reshape(-1, x.shape[-1]))
    top, ties = top25_from_logits(logits, domain)
    return top.reshape(len(source_layers), K), ties, ids


def jacobian_stack(lens: Any, source_layers: list[int], device: str) -> torch.Tensor:
    return torch.stack([lens.jacobians[l].to(device=device, dtype=torch.float32)
                        for l in source_layers], dim=0)


def score_suite(top25: np.ndarray, label_lists: list[list[int]],
                rng: np.random.Generator) -> dict[str, Any]:
    """Instrument-level scoring: any-layer union + layer profile + permutation.

    Uses only sealed primitives. Deliberately omits the Phase-1 L17 criterion:
    D2/D3 are instrument-level by design (sealed protocol §4).
    """
    n_layers = top25.shape[1]
    union = scoring.pass_at_25(top25, label_lists, layers=range(n_layers))
    profile = [scoring.pass_at_25(top25, label_lists, layers=[l]) for l in range(n_layers)]
    all_null, _ = scoring.external_permutation_null(top25, label_lists, rng=rng,
                                                    n_perm=N_PERM)
    q95 = scoring.higher_quantile(all_null, 0.95)
    p = scoring.empirical_upper_p(all_null, union)
    return {
        "any_layer_union_pass_at_25": union,
        "layer_profile_pass_at_25": profile,
        "peak_layer": int(np.argmax(profile)),
        "peak_layer_pass_at_25": float(np.max(profile)),
        "permutation": {"n": N_PERM, "p95_higher": q95, "empirical_upper_p": p},
        "beats_p95": bool(union > q95),
        "p_at_most_0.05": bool(p <= 0.05),
        "qualifies_instrument_level": bool(union > q95 and p <= 0.05),
    }


def environment() -> dict[str, Any]:
    import transformers
    env = {
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "numpy": np.__version__,
        "platform": platform.platform(), "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["gpu_total_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    return env


def write_bundle(outdir: str, manifest: dict[str, Any], report: dict[str, Any],
                 markdown: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    json.dump(manifest, open(os.path.join(outdir, "manifest.json"), "w"),
              indent=1, sort_keys=True, default=str)
    json.dump(report, open(os.path.join(outdir, "report.json"), "w"),
              indent=1, sort_keys=True, default=str)
    open(os.path.join(outdir, "REPORT.md"), "w").write(markdown)
