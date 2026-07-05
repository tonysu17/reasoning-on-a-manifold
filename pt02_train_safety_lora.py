#!/usr/bin/env python3
"""Post-training spillover — Step 2: LoRA safety SFT (dose-response + control).

Fine-tunes a reasoning model (default R1-1.5B) on the contrastive dataset with a
completion-only loss. Produces one LoRA adapter (optionally a merged full model
for the extraction pipeline) per requested DOSE so the geometric shift can be
read as a trajectory rather than a single point.

Runs on the DGX Spark (CUDA, bf16). Requires: pip install .[gpu] peft

Examples
--------
    # full dose-response on the safety data, save merged checkpoints for extraction:
    python pt02_train_safety_lora.py --data data/safety_contrastive.json \
        --dose 100,300,all --merge --out-dir checkpoints/r1_1.5b_safety

    # size-matched NON-SAFETY control (isolate "safety" from "any SFT"):
    python pt02_train_safety_lora.py --data data/control_generic_sft.json \
        --dose all --merge --out-dir checkpoints/r1_1.5b_control
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.config import model_tuple
from src.safety_posttrain import contrastive as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pt02")


def _parse_doses(spec: str, n: int) -> list[int]:
    out = []
    for tok in spec.split(","):
        tok = tok.strip().lower()
        if tok in ("all", "full", ""):
            out.append(n)
        else:
            out.append(min(int(tok), n))
    return sorted(set(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="contrastive dataset JSON (pt01 output)")
    ap.add_argument("--model", default="1.5b", help="cli_alias from config.yaml (default R1-1.5B)")
    ap.add_argument("--out-dir", default="checkpoints/r1_1.5b_safety")
    ap.add_argument("--dose", default="all",
                    help="comma list of example counts, e.g. '100,300,all'")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--merge", action="store_true",
                    help="also save a merged full model per dose (for 04_extract_activations)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Heavy import deferred so --help / arg errors don't require torch+peft.
    from src.safety_posttrain import sft as S

    model_id, short, dtype = model_tuple(args.model)
    log.info("base model: %s (%s, %s)", model_id, short, dtype)

    records = C.load_dataset(args.data)
    # Need responses for SFT.
    records = [r for r in records if r.get("answer")]
    if not records:
        raise SystemExit(
            f"{args.data} has no records with an 'answer' field — generate the "
            f"dataset WITH responses (pt01 without --no-responses)."
        )
    log.info("loaded %d SFT-ready records from %s", len(records), args.data)

    # Build SFT text with the real tokenizer's family (so the chat template matches).
    from src.chain_gen import load_model
    _, tokenizer = load_model(model_id, dtype=dtype)
    from src.model_adapters import family_of
    sft_all = C.records_to_sft(records, tokenizer=tokenizer, family=family_of(model_id))

    doses = _parse_doses(args.dose, len(sft_all))
    log.info("dose-response levels: %s", doses)

    out_root = Path(args.out_dir)
    summary = {"base_model": model_id, "data": args.data, "runs": []}
    for dose in doses:
        subset = sft_all[:dose]
        run_dir = out_root / f"dose_{dose}"
        log.info("=== training dose=%d -> %s ===", dose, run_dir)
        result = S.train_lora(
            subset, model_id, run_dir,
            dtype=dtype, epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, grad_accum=args.grad_accum,
            max_len=args.max_len, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, seed=args.seed, merge=args.merge,
        )
        summary["runs"].append({"dose": dose, **result})

    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("done — %d runs; summary -> %s", len(doses), out_root / "training_summary.json")


if __name__ == "__main__":
    main()
