#!/usr/bin/env python3
"""Post-training spillover — Step 2c: FULL-parameter safety SFT (no LoRA).

Closes the "LoRA-vs-full-FT regime" caveat on the pt02 arms by training every
weight, matching the published STAR-1 recipe the repo declares
(METHODOLOGY_SAFETY_SPILLOVER_2026-07-03.md §4 / runpod_runB_remote.sh):
full-parameter SFT, bf16, **5 epochs, LR 1e-5 cosine + 5% warmup, AdamW
0.9/0.95, wd 1e-4, effective batch 128, completion-only loss**; seq capped at
4096 like the LoRA arms (dataset p99 << 4096). Same input format as pt02
(pt01/pt01b/pt01c record JSON: prompt / reasoning / answer / ...).

Fits a 24 GB RTX 4090 for a 1.5B model: bf16 weights+grads (~7 GB) + gradient
checkpointing + micro-batch 2 x grad-accum 64 + one memory-lean optimizer
route — 8-bit Adam when bitsandbytes is installed (CUDA), else Adafactor
(fp32 AdamW states alone would be ~14 GB and not fit). The chosen route is
logged and recorded in training_summary.json.

Saves a plain ``from_pretrained``-loadable checkpoint into --out-dir, directly
consumable by ``04_extract_activations.py --model-path <out-dir>``.

Examples
--------
    # the real STAR-1 recipe reproduction (RunPod 4090):
    python pt02c_train_full_ft.py --data data/safety_star1_sft.json \
        --seed 42 --out-dir checkpoints/r1_1.5b_safety_fullft_s42

    # off-policy non-safety control arm (pt01c output):
    python pt02c_train_full_ft.py --data data/control_offpolicy_sft.json \
        --seed 42 --out-dir checkpoints/r1_1.5b_control_offpolicy_fullft_s42
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.config import model_tuple
from src.safety_posttrain import contrastive as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pt02c")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset JSON (pt01/pt01b/pt01c output)")
    ap.add_argument("--model", default="1.5b", help="cli_alias from config.yaml (default R1-1.5B)")
    ap.add_argument("--out-dir", default="checkpoints/r1_1.5b_safety_fullft")
    ap.add_argument("--epochs", type=float, default=5.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=2,
                    help="micro-batch per device (24GB 4090: keep at 1-2)")
    ap.add_argument("--grad-accum", type=int, default=64,
                    help="batch-size * grad-accum = effective batch (STAR-1: 128)")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap optimizer steps (smoke tests only)")
    args = ap.parse_args(argv)

    # Heavy import deferred so --help / arg errors don't require torch.
    from src.safety_posttrain import sft as S

    model_id, short, dtype = model_tuple(args.model)
    # config.yaml's dtype is an INFERENCE setting (float16 for the 1.5b alias);
    # training must run bf16: fp16 weights put Trainer on the GradScaler path,
    # which cannot unscale half-precision gradients, and the declared STAR-1
    # recipe is bf16 anyway.
    if dtype != "bfloat16":
        log.info("overriding config dtype %s -> bfloat16 for full-FT training", dtype)
        dtype = "bfloat16"
    log.info("base model: %s (%s, %s)", model_id, short, dtype)
    eff = args.batch_size * args.grad_accum
    if eff != 128:
        log.warning("effective batch %d != 128 (declared STAR-1 recipe)", eff)

    records = C.load_dataset(args.data)
    records = [r for r in records if r.get("answer")]
    if not records:
        raise SystemExit(f"{args.data} has no records with an 'answer' field")
    log.info("loaded %d SFT-ready records from %s", len(records), args.data)

    # Tokenizer only (chat template for the SFT text); unlike pt02 we avoid
    # load_model here so the full model is resident exactly once (full FT
    # holds weights+grads+optimizer state — no room for a stray second copy).
    from transformers import AutoTokenizer

    from src.model_adapters import family_of
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    sft_all = C.records_to_sft(records, tokenizer=tokenizer, family=family_of(model_id))

    out_dir = Path(args.out_dir)
    result = S.train_full(
        sft_all, model_id, out_dir,
        dtype=dtype, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, grad_accum=args.grad_accum,
        max_len=args.max_len, warmup_ratio=args.warmup_ratio,
        seed=args.seed, max_steps=args.max_steps,
    )

    summary = {"base_model": model_id, "data": args.data, "seed": args.seed,
               "recipe": "STAR-1 full SFT (5ep, lr1e-5 cosine, eff.batch 128, "
                         "completion-only loss, seq<=4096)",
               **result}
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("done — %d steps, loss %.4f, optim %s; checkpoint + summary -> %s",
             result["steps"], result["train_loss"], result["optim"], out_dir)


if __name__ == "__main__":
    main()
