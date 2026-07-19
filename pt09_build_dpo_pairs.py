#!/usr/bin/env python3
r"""Post-training as entropy reduction — R3: build DPO preference datasets.

Builds TRL-DPO-format JSON (a list of ``{"prompt", "chosen", "rejected"}``
records, plus provenance metadata) for two arms:

  (a) SAFETY   — prompt = STAR-1 question; chosen = STAR-1's deliberative-safety
                 completion (reasoning + answer); rejected = R1-1.5B's OWN
                 generation for that prompt. This is the off-policy-contrastive
                 safety point (primer §3, R3).

  (b) CONTROL  — prompt = a math/reasoning task; chosen = a CORRECT chain,
                 rejected = an INCORRECT chain for the SAME task. Correctness is
                 judge-free (primer's "verifiable, no judge"): see the DATA GAP
                 note below for how correctness is decided given that the repo
                 has no reference answers.

Modes
-----
  --mock       deterministic, offline, no torch/network — builds a tiny valid
               DPO JSON for either arm (used by the unit tests).
  --generate   load the model on a pod and generate the missing side (safety:
               the rejected base response; control: K samples/task for the
               correctness pairing). Writes the DPO JSON (+ for control, a
               ``*_pseudo_references.json`` for pt11's math reward).
  (neither)    "assemble" mode — read a generation cache produced by a prior
               --generate run (``--gen-cache``) and (re)emit the DPO JSON
               without regenerating. Errors with a hint if no cache is present.

DATA GAP (control / math correctness)
-------------------------------------
``data/tasks_final.json`` has NO reference answers, and only ~174/1000 greedy
chains emit a ``\boxed{...}`` (the tasks are open-ended proofs). There is one
greedy chain per task, so same-prompt chosen/rejected pairs cannot be assembled
from the existing corpus at all. Resolution, in priority order:

  1. ``--reference-answers PATH`` supplied  -> correctness = boxed exact-match
     against the reference (the tolerant loader in ``rl_rewards``).
  2. otherwise (default)                     -> SELF-CONSISTENCY: sample K
     completions/task, take the majority ``\boxed{}`` answer as a pseudo-
     reference; a sample is "correct" iff its boxed answer == majority. A task
     yields a pair only if it has BOTH a majority-agreeing sample and a
     disagreeing (or answer-less) sample. Tasks that never box out are skipped.
     Documented as a proxy, not ground truth; the majority answers are written
     to ``*_pseudo_references.json`` so pt11's GRPO-math arm can reuse them.

Examples
--------
    # tests / offline:
    python pt09_build_dpo_pairs.py --arm safety  --mock --out data/dpo_safety_mock.json
    python pt09_build_dpo_pairs.py --arm control --mock --out data/dpo_control_mock.json

    # real (pod):
    python pt09_build_dpo_pairs.py --arm safety  --generate \
        --safety-data data/safety_star1_sft.json --out data/dpo_safety.json
    python pt09_build_dpo_pairs.py --arm control --generate --k 8 --temperature 0.8 \
        --tasks data/tasks_final.json --out data/dpo_control.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.safety_posttrain import contrastive as C  # noqa: E402  (torch-free)
from src.safety_posttrain import rl_rewards as R  # noqa: E402  (torch-free)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("pt09")

# Manual DeepSeek prompt scaffold for the offline (no-tokenizer) mock path.
_MANUAL_PROMPT = C._DEEPSEEK_MANUAL  # "<｜begin▁of▁sentence｜><｜User｜>{instruction}<｜Assistant｜><think>\n"


# ── prompt formatting ────────────────────────────────────────────────────────

def _format_prompt(instruction: str, tokenizer=None, family: str = "deepseek") -> str:
    """Chat-templated prompt ending in the model's generation prefix (``<think>\\n``).

    Uses the real tokenizer's template when available (generate mode), else the
    manual DeepSeek scaffold (mock mode) — same discipline as the SFT builder.
    """
    if tokenizer is not None:
        from src.model_adapters import format_prompt as _fmt
        return _fmt(tokenizer, instruction, family=family)
    return _MANUAL_PROMPT.format(instruction=instruction)


def _dpo_record(prompt: str, chosen: str, rejected: str, **meta) -> dict:
    rec = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
    rec.update(meta)
    return rec


# ── SAFETY arm ───────────────────────────────────────────────────────────────

def build_safety_mock(n_pairs: int = 6) -> list[dict]:
    """Deterministic safety DPO pairs: chosen = a refusal-style deliberation,
    rejected = a canned 'over-compliant' completion. No model needed."""
    recs = C.mock_dataset(n_pairs, with_responses=True)
    harmful = [r for r in recs if r["label"] == "harmful"]
    out = []
    for r in harmful:
        prompt = _format_prompt(r["prompt"], tokenizer=None)
        chosen = C.assemble_completion(r["reasoning"], r["answer"])
        rejected = C.assemble_completion(
            "The user asked a direct question; I'll just answer it straight away.",
            "Sure — here is exactly what you asked for, step by step: ...",
        )
        out.append(_dpo_record(prompt, chosen, rejected, id=r["id"],
                               arm="safety", category=r.get("category")))
    return out


def build_safety_generate(safety_recs, model_id, dtype, *, n, max_new_tokens,
                          temperature, seed) -> list[dict]:
    """Real safety pairs: chosen = STAR-1 completion, rejected = base R1-1.5B's
    own generation for the same prompt."""
    from src.chain_gen import load_model, generate_chain
    from src.model_adapters import family_of

    model, tok = load_model(model_id, dtype=dtype)
    fam = family_of(model_id)
    subset = safety_recs[:n] if n else safety_recs
    out, cache = [], []
    for i, r in enumerate(subset):
        prompt = _format_prompt(r["prompt"], tokenizer=tok, family=fam)
        chosen = C.assemble_completion(r.get("reasoning", ""), r["answer"])
        gen = generate_chain(model, tok, r["prompt"], max_new_tokens=max_new_tokens,
                             temperature=temperature, seed=seed, model_id=model_id)
        rejected = gen["chain"].strip()
        if not rejected:
            log.warning("empty base generation for %s; skipping", r.get("id"))
            continue
        out.append(_dpo_record(prompt, chosen, rejected, id=r.get("id"),
                               arm="safety", category=r.get("category")))
        cache.append({"id": r.get("id"), "prompt": r["prompt"], "rejected": rejected})
        if (i + 1) % 25 == 0:
            log.info("safety: %d/%d generated", i + 1, len(subset))
    return out, cache


# ── CONTROL (math) arm ───────────────────────────────────────────────────────

def build_control_mock(n_pairs: int = 6) -> list[dict]:
    """Deterministic control DPO pairs with fabricated correct/incorrect boxed
    chains, so the tests exercise the boxed-answer pairing logic offline."""
    out = []
    for i in range(n_pairs):
        a, b = 2 + i, 3 + i
        instruction = f"Compute {a} + {b}. Put the final answer in \\boxed{{}}."
        prompt = _format_prompt(instruction, tokenizer=None)
        chosen = C.assemble_completion(
            f"Adding {a} and {b} gives {a + b}.", f"The answer is \\boxed{{{a + b}}}.")
        rejected = C.assemble_completion(
            f"Adding {a} and {b}... I think it's {a + b + 1}.",
            f"The answer is \\boxed{{{a + b + 1}}}.")
        out.append(_dpo_record(prompt, chosen, rejected, id=f"MOCK_{i:03d}",
                               arm="control", category="mock_arith"))
    return out


def _label_samples(samples, task_id, refmap):
    """Given decoded samples for one task, return (correct_idx, incorrect_idx,
    pseudo_reference). Uses reference exact-match if available, else majority-vote
    self-consistency on the boxed answer."""
    boxed = [R.extract_boxed(s) for s in samples]
    ref = refmap.get(str(task_id)) if refmap else None

    if ref is not None:
        correct = [i for i, b in enumerate(boxed) if b and R.answers_match(b, ref)]
        incorrect = [i for i, b in enumerate(boxed) if not (b and R.answers_match(b, ref))]
        return correct, incorrect, R.normalize_answer(ref)

    # self-consistency
    have = [R.normalize_answer(b) for b in boxed if b]
    counts = Counter(x for x in have if x)
    if not counts:
        return [], [], None
    majority, _ = counts.most_common(1)[0]
    correct = [i for i, b in enumerate(boxed)
               if b and R.normalize_answer(b) == majority]
    incorrect = [i for i, b in enumerate(boxed)
                 if (not b) or R.normalize_answer(b) != majority]
    return correct, incorrect, majority


def build_control_generate(tasks, model_id, dtype, *, n, k, max_new_tokens,
                           temperature, seed, refmap) -> tuple[list[dict], dict]:
    """Real control pairs via K-sample self-consistency (or reference match)."""
    from src.chain_gen import load_model, generate_chain
    from src.model_adapters import family_of

    model, tok = load_model(model_id, dtype=dtype)
    fam = family_of(model_id)
    subset = tasks[:n] if n else tasks
    out, pseudo_refs = [], {}
    n_skipped_nobox = n_skipped_nopair = 0
    for ti, t in enumerate(subset):
        instruction = t.get("instruction") or t["prompt"]
        tid = t.get("id") or t.get("task_id") or f"task_{ti}"
        samples = [
            generate_chain(model, tok, instruction, max_new_tokens=max_new_tokens,
                           temperature=temperature, seed=seed + s, model_id=model_id)["chain"]
            for s in range(k)
        ]
        correct, incorrect, pseudo = _label_samples(samples, tid, refmap)
        if pseudo is None:
            n_skipped_nobox += 1
            continue
        if not correct or not incorrect:
            n_skipped_nopair += 1
            continue
        prompt = _format_prompt(instruction, tokenizer=tok, family=fam)
        out.append(_dpo_record(prompt, samples[correct[0]].strip(),
                               samples[incorrect[0]].strip(), id=tid,
                               arm="control", category=t.get("category")))
        pseudo_refs[str(tid)] = pseudo
        if (ti + 1) % 20 == 0:
            log.info("control: %d/%d tasks (%d pairs so far)", ti + 1, len(subset), len(out))
    log.info("control: %d pairs; skipped %d (no boxed answer) + %d (no correct/incorrect split)",
             len(out), n_skipped_nobox, n_skipped_nopair)
    return out, pseudo_refs


# ── assemble-from-cache (no regeneration) ────────────────────────────────────

def assemble_safety_from_cache(safety_recs, cache_path, tokenizer=None) -> list[dict]:
    cache = {c["id"]: c for c in json.loads(Path(cache_path).read_text())}
    out = []
    for r in safety_recs:
        c = cache.get(r.get("id"))
        if not c:
            continue
        prompt = _format_prompt(r["prompt"], tokenizer=tokenizer)
        chosen = C.assemble_completion(r.get("reasoning", ""), r["answer"])
        out.append(_dpo_record(prompt, chosen, c["rejected"], id=r.get("id"),
                               arm="safety", category=r.get("category")))
    return out


# ── IO ───────────────────────────────────────────────────────────────────────

def write_dpo(records: list[dict], path: Path, extra_meta: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # keep only DPO keys in a clean copy; retain meta keys too (pt10 selects).
    path.write_text(json.dumps(records, indent=1, ensure_ascii=False))
    log.info("wrote %d DPO pairs -> %s", len(records), path)
    try:
        from src.config import provenance
        meta = dict(extra_meta or {})
        meta.update({"n_pairs": len(records),
                     "format": "trl-dpo (prompt/chosen/rejected)"})
        (path.with_suffix(".meta.json")).write_text(
            json.dumps({**provenance(), **meta}, indent=2))
    except Exception as exc:  # pragma: no cover
        log.warning("provenance stamp failed: %s", exc)


def _default_out(arm: str, mock: bool) -> str:
    tag = "_mock" if mock else ""
    return f"data/dpo_{arm}{tag}.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=["safety", "control"])
    ap.add_argument("--mock", action="store_true", help="offline deterministic build (no torch)")
    ap.add_argument("--generate", action="store_true", help="generate the missing side (pod)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default="1.5b", help="cli_alias from config.yaml")
    ap.add_argument("--n", type=int, default=None, help="cap #prompts (default all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="sampling temperature for generation (safety rejected / control samples)")
    # safety
    ap.add_argument("--safety-data", default="data/safety_star1_sft.json")
    ap.add_argument("--gen-cache", default=None,
                    help="assemble mode: cached base generations from a prior --generate run")
    # control
    ap.add_argument("--tasks", default="data/tasks_final.json")
    ap.add_argument("--k", type=int, default=8, help="samples/task for self-consistency")
    ap.add_argument("--reference-answers", default=None,
                    help="optional reference-answers file; if given, correctness = boxed match")
    ap.add_argument("--mock-n", type=int, default=6, help="#pairs in --mock mode")
    args = ap.parse_args()

    out = Path(args.out or _default_out(args.arm, args.mock))

    # ── MOCK ────────────────────────────────────────────────────────────────
    if args.mock:
        recs = (build_safety_mock(args.mock_n) if args.arm == "safety"
                else build_control_mock(args.mock_n))
        write_dpo(recs, out, {"arm": args.arm, "mode": "mock"})
        return

    # ── GENERATE ────────────────────────────────────────────────────────────
    if args.generate:
        from src.config import model_tuple
        model_id, short, dtype = model_tuple(args.model)
        log.info("base model: %s (%s, %s)", model_id, short, dtype)

        if args.arm == "safety":
            safety_recs = json.loads(Path(args.safety_data).read_text())
            recs, cache = build_safety_generate(
                safety_recs, model_id, dtype, n=args.n,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                seed=args.seed)
            write_dpo(recs, out, {"arm": "safety", "mode": "generate", "model": model_id})
            (out.with_suffix(".gencache.json")).write_text(
                json.dumps(cache, indent=1, ensure_ascii=False))
        else:
            tasks = json.loads(Path(args.tasks).read_text())
            refmap = R.load_reference_answers(args.reference_answers) if args.reference_answers else {}
            if refmap:
                log.info("using %d reference answers (exact-match correctness)", len(refmap))
            else:
                log.info("no reference answers — using majority-vote self-consistency (proxy)")
            recs, pseudo = build_control_generate(
                tasks, model_id, dtype, n=args.n, k=args.k,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                seed=args.seed, refmap=refmap)
            write_dpo(recs, out, {"arm": "control", "mode": "generate", "model": model_id,
                                  "correctness": "reference" if refmap else "self_consistency",
                                  "k": args.k})
            (out.with_name(out.stem + "_pseudo_references.json")).write_text(
                json.dumps(pseudo, indent=1, ensure_ascii=False))
            log.info("wrote %d pseudo-references -> %s", len(pseudo),
                     out.with_name(out.stem + "_pseudo_references.json"))
        return

    # ── ASSEMBLE FROM CACHE ─────────────────────────────────────────────────
    if args.arm == "safety" and args.gen_cache:
        safety_recs = json.loads(Path(args.safety_data).read_text())
        recs = assemble_safety_from_cache(safety_recs, args.gen_cache)
        write_dpo(recs, out, {"arm": "safety", "mode": "assemble"})
        return

    raise SystemExit(
        "Nothing to do: pass --mock (offline test), --generate (run the model on a "
        "pod), or --gen-cache PATH (assemble from a prior --generate run). "
        "Same-prompt DPO pairs cannot be built from the existing single-chain "
        "corpus without generation — see the DATA GAP note in this file's docstring."
    )


if __name__ == "__main__":
    main()
