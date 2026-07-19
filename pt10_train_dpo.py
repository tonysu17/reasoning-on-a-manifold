#!/usr/bin/env python3
r"""Post-training as entropy reduction — R3: LoRA DPO training (TRL).

Trains a LoRA adapter (rank 16, matching the SFT arms) with TRL's ``DPOTrainer``
on the preference pairs from ``pt09_build_dpo_pairs.py``. LoRA-only so the new
arm is internally comparable to the executed SFT/DPO/GRPO arms (the intruder-
dimension confound of full-FT direction comparisons, primer §3).

KL checkpoints
--------------
The primer's dose axis is measured KL(π‖π₀) on the frozen corpus, checkpointed at
~2 levels so methods compare at MATCHED KL, not matched steps. Measuring KL
online is complex, so this trainer instead **saves the adapter at fixed training-
step fractions {0.5, 1.0}** (``--kl-fractions``). KL is then measured POST-HOC by
the entropy battery (primer R1 / "pt08") on each checkpoint. Each checkpoint dir
is ``<out>/ckpt_frac_<f>/`` (adapter; + ``merged/`` when ``--merge``).

Reference model: with a LoRA ``peft_config`` TRL uses the adapter-disabled base
as the frozen reference (no separate ref model needed / loaded).

Examples
--------
    # safety DPO arm, beta 0.1, save adapter+merged at 50%/100%:
    python pt10_train_dpo.py --data data/dpo_safety.json --beta 0.1 --merge \
        --out-dir checkpoints/r1_1.5b_dpo_safety

    # control (math) DPO arm:
    python pt10_train_dpo.py --data data/dpo_control.json --beta 0.1 --merge \
        --out-dir checkpoints/r1_1.5b_dpo_control
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("pt10")

# LoRA targets — same as the SFT arms (Qwen2 / R1-Distill projection names).
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ── data ──────────────────────────────────────────────────────────────────────

def load_pairs(path) -> "list[dict]":
    """Load pt09 output and keep only the TRL-DPO columns."""
    rows = json.loads(Path(path).read_text())
    pairs = [{"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
             for r in rows if r.get("prompt") and r.get("chosen") and r.get("rejected")]
    if not pairs:
        raise SystemExit(f"{path}: no valid prompt/chosen/rejected triples")
    return pairs


# ── KL-fraction checkpoint callback ──────────────────────────────────────────

def _make_checkpoint_callback(fractions, out_dir, tokenizer):
    from transformers import TrainerCallback

    class FractionCheckpoint(TrainerCallback):
        """Save the LoRA adapter at global steps closest to the given fractions
        of total training. Adapter-only (safe mid-training); merged models are
        exported post-hoc so ``merge_and_unload`` never mutates the live model."""

        def __init__(self):
            self.fractions = list(fractions)
            self.targets: dict[int, float] = {}
            self.saved: dict[float, str] = {}

        def on_train_begin(self, args, state, control, **kw):
            total = max(1, state.max_steps)
            for f in self.fractions:
                step = min(total, max(1, round(f * total)))
                self.targets[step] = f
            log.info("KL checkpoints at steps %s (of %d)",
                     sorted(self.targets), total)

        def _save(self, model, f):
            d = Path(out_dir) / f"ckpt_frac_{f:g}"
            d.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(d))
            tokenizer.save_pretrained(str(d))
            self.saved[f] = str(d)
            log.info("saved KL checkpoint frac=%.2f -> %s", f, d)

        def on_step_end(self, args, state, control, model=None, **kw):
            f = self.targets.get(state.global_step)
            if f is not None and f not in self.saved and model is not None:
                self._save(model, f)

    return FractionCheckpoint()


# ── merge (post-hoc, from saved adapters) ────────────────────────────────────

def merge_adapter(base_model_id, adapter_dir, out_dir, dtype: str = "bfloat16"):
    """Load base + adapter, merge, and save a full model (for the extraction
    pipeline). Done outside the training loop so it can't corrupt training."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                   "float32": torch.float32}[dtype]
    base = AutoModelForCausalLM.from_pretrained(base_model_id, dtype=torch_dtype)
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    out = Path(adapter_dir) / "merged"
    merged.save_pretrained(str(out))
    AutoTokenizer.from_pretrained(str(adapter_dir)).save_pretrained(str(out))
    log.info("merged model -> %s", out)
    return str(out)


# ── training ──────────────────────────────────────────────────────────────────

def train_dpo(
    pairs,
    *,
    model_id: str,
    out_dir,
    beta: float = 0.1,
    dtype: str = "bfloat16",
    epochs: float = 1.0,
    lr: float = 5e-6,
    batch_size: int = 2,
    grad_accum: int = 8,
    max_len: int = 1024,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    seed: int = 42,
    kl_fractions=(0.5, 1.0),
    merge: bool = False,
    max_steps: int = -1,
) -> dict:
    """Run one LoRA DPO training and save checkpoints. ``model_id`` may be a HF
    id or a local model dir (the tests pass a tiny local Qwen2)."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_cuda = torch.cuda.is_available()

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                   "float32": torch.float32}[dtype]
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype)

    ds = Dataset.from_list(pairs)
    lora = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=DEFAULT_TARGET_MODULES, bias="none", task_type="CAUSAL_LM",
    )

    cfg = DPOConfig(
        output_dir=str(out_dir),
        beta=beta,
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_length=max_len,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        seed=seed,
        bf16=(use_cuda and dtype == "bfloat16"),
        fp16=(use_cuda and dtype == "float16"),
        gradient_checkpointing=use_cuda,
    )

    ckpt_cb = _make_checkpoint_callback(kl_fractions, out_dir, tok)
    trainer = DPOTrainer(
        model=model,
        ref_model=None,          # LoRA: adapter-disabled base is the reference
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=lora,
        callbacks=[ckpt_cb],
    )
    train_out = trainer.train()

    # always save a final adapter (frac 1.0) even if the callback didn't fire
    final_dir = out_dir / "ckpt_frac_1"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(final_dir))
    tok.save_pretrained(str(final_dir))
    ckpt_cb.saved.setdefault(1.0, str(final_dir))

    merged = {}
    if merge:
        del model, trainer
        if use_cuda:
            torch.cuda.empty_cache()
        for f, adir in ckpt_cb.saved.items():
            merged[f] = merge_adapter(model_id, adir, adir, dtype=dtype)

    summary = {
        "model_id": model_id,
        "out_dir": str(out_dir),
        "beta": beta,
        "n_pairs": len(pairs),
        "lora_r": lora_r,
        "kl_checkpoints": ckpt_cb.saved,
        "merged": merged,
        "train_loss": float(getattr(train_out, "training_loss", float("nan"))),
        "epochs": epochs,
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="pt09 DPO JSON")
    ap.add_argument("--model", default="1.5b", help="cli_alias from config.yaml")
    ap.add_argument("--model-path", default=None,
                    help="load base from a local dir instead of the config alias (tests)")
    ap.add_argument("--out-dir", default="checkpoints/r1_1.5b_dpo")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--kl-fractions", default="0.5,1.0",
                    help="comma list of step-fractions to checkpoint at")
    ap.add_argument("--merge", action="store_true", help="also export merged models")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()

    if args.model_path:
        model_id, dtype = args.model_path, "float32"
    else:
        from src.config import model_tuple
        model_id, _short, dtype = model_tuple(args.model)
    fractions = [float(x) for x in args.kl_fractions.split(",") if x.strip()]

    pairs = load_pairs(args.data)
    log.info("loaded %d DPO pairs from %s", len(pairs), args.data)
    summary = train_dpo(
        pairs, model_id=model_id, out_dir=args.out_dir, beta=args.beta, dtype=dtype,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        grad_accum=args.grad_accum, max_len=args.max_len, lora_r=args.lora_r,
        lora_alpha=args.lora_alpha, seed=args.seed, kl_fractions=fractions,
        merge=args.merge, max_steps=args.max_steps,
    )
    log.info("done — checkpoints: %s", list(summary["kl_checkpoints"]))


if __name__ == "__main__":
    main()
