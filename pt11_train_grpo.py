#!/usr/bin/env python3
r"""Post-training as entropy reduction — R4: LoRA GRPO / RLVR-lite (TRL).

On-policy tilting toward reward-approved trajectories, the sharpening point on
the translation-vs-contraction plane (primer §2-3, R4). TRL ``GRPOTrainer``
(installed TRL exposes GRPOTrainer natively, so no PPO fallback is needed — see
``--reward`` below), group size 8, LoRA rank 16.

Two judge-free reward modes (``--reward``), both from ``src.safety_posttrain.rl_rewards``:

  math            extract ``\boxed{...}`` from the completion and exact-match it
                  against a reference answer (the canonical RLVR). Needs a
                  reference-answers file (``--reference-answers``); the loader is
                  tolerant (id->answer dict, id->record dict, or list of records).
                  Since the repo ships NO reference answers, mine them first with
                  ``pt09 --arm control --generate`` (self-consistency), which
                  writes ``*_pseudo_references.json`` — feed that here.

  refusal-format  rule-based rubric: reward refusal-style completions on
                  harmful-tagged prompts and non-refusal on benign prompts
                  (documented in rl_rewards.refusal_format_reward). Prompts carry
                  a ``harmful`` bool (or ``label`` in {harmful,benign}).

Prompts come from a small JSON the script accepts (``--prompts``): a list of
records with an instruction (``prompt``/``instruction``) and, per mode, either a
reference (math; inline or via --reference-answers) or a harmful flag (refusal).

Examples
--------
    # math RLVR (control), references mined by pt09:
    python pt11_train_grpo.py --reward math --prompts data/tasks_final.json \
        --reference-answers data/dpo_control_pseudo_references.json \
        --group-size 8 --out-dir checkpoints/r1_1.5b_grpo_math

    # refusal-format (safety), harmful/benign prompt pool:
    python pt11_train_grpo.py --reward refusal-format --prompts data/grpo_refusal_prompts.json \
        --group-size 8 --out-dir checkpoints/r1_1.5b_grpo_refusal
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.safety_posttrain import rl_rewards as R  # noqa: E402  (torch-free)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("pt11")

DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

_HARMFUL_TRUE = {"harmful", "unsafe", "refuse", "true", "1", "yes"}


# ── reward functions (TRL signature: reward(completions, **cols) -> list[float]) ─

def make_math_reward():
    def reward_math(completions, reference=None, **kw):
        refs = reference if reference is not None else [None] * len(completions)
        return [R.math_reward(c, r) for c, r in zip(completions, refs)]
    reward_math.__name__ = "reward_math"
    return reward_math


def make_refusal_reward():
    def reward_refusal(completions, harmful=None, **kw):
        flags = harmful if harmful is not None else [False] * len(completions)
        return [R.refusal_format_reward(c, h) for c, h in zip(completions, flags)]
    reward_refusal.__name__ = "reward_refusal_format"
    return reward_refusal


# ── dataset construction ─────────────────────────────────────────────────────

def _is_harmful(row) -> bool:
    if "harmful" in row and isinstance(row["harmful"], bool):
        return row["harmful"]
    v = row.get("harmful", row.get("label", row.get("safety_label", "")))
    return str(v).strip().lower() in _HARMFUL_TRUE


def build_dataset(prompts_path, reward_mode, *, tokenizer=None, family="deepseek",
                  reference_path=None, n=None):
    """Build a HF Dataset with a formatted ``prompt`` column and the reward's
    side-columns (``reference`` for math, ``harmful`` for refusal-format)."""
    from datasets import Dataset

    def _fmt(instruction):
        if tokenizer is not None:
            from src.model_adapters import format_prompt as f
            return f(tokenizer, instruction, family=family)
        from src.safety_posttrain.contrastive import _DEEPSEEK_MANUAL
        return _DEEPSEEK_MANUAL.format(instruction=instruction)

    rows = json.loads(Path(prompts_path).read_text())
    if isinstance(rows, dict):
        rows = [{"id": k, **(v if isinstance(v, dict) else {"prompt": v})}
                for k, v in rows.items()]
    refmap = R.load_reference_answers(reference_path) if reference_path else {}

    out = []
    n_skipped = 0
    for i, row in enumerate(rows):
        instruction = row.get("instruction") or row.get("prompt")
        if not instruction:
            continue
        tid = str(row.get("id") or row.get("task_id") or f"row_{i}")
        rec = {"prompt": _fmt(instruction), "task_id": tid}
        if reward_mode == "math":
            ref = refmap.get(tid)
            if ref is None:
                ref = R._first(row, R._REF_KEYS) if hasattr(R, "_first") else None
            if ref is None:
                n_skipped += 1
                continue
            rec["reference"] = str(ref)
        else:  # refusal-format
            rec["harmful"] = _is_harmful(row)
        out.append(rec)
        if n and len(out) >= n:
            break
    if reward_mode == "math" and n_skipped:
        log.info("math: skipped %d prompts with no reference answer", n_skipped)
    if not out:
        raise SystemExit(
            "no usable prompts — for --reward math every prompt needs a reference "
            "(supply --reference-answers, e.g. pt09's *_pseudo_references.json)")
    return Dataset.from_list(out)


# ── training ──────────────────────────────────────────────────────────────────

def train_grpo(
    dataset,
    reward_func,
    *,
    model_id: str,
    out_dir,
    dtype: str = "bfloat16",
    group_size: int = 8, micro_batch: int | None = None,
    epochs: float = 1.0,
    lr: float = 1e-6,
    grad_accum: int = 4,
    max_prompt_len: int = 640,
    max_completion_len: int = 512,
    temperature: float = 0.9,
    beta: float = 0.04,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    seed: int = 42,
    max_steps: int = -1,
) -> dict:
    """One LoRA GRPO run. ``model_id`` may be a HF id or a local dir (tests)."""
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_cuda = torch.cuda.is_available()
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                   "float32": torch.float32}[dtype]

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype)

    lora = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=DEFAULT_TARGET_MODULES, bias="none", task_type="CAUSAL_LM",
    )

    # generation batch = per_device_train_batch_size * steps_per_generation must
    # be divisible by group_size (num_generations); the simplest always-valid
    # choice is one unique prompt per micro-step -> bs == group_size.
    micro_batch = micro_batch or group_size
    cfg = GRPOConfig(
        output_dir=str(out_dir),
        num_generations=group_size,
        per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=grad_accum,
        # steps_per_generation defaults to grad_accum: generation_batch =
        # group_size * grad_accum, always divisible by num_generations (group_size).
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        # NB: TRL 1.8's GRPOConfig dropped `max_prompt_length` (prompts are no
        # longer pre-truncated by the config); left-truncate long prompts at
        # build time if needed. Only completion length is capped here.
        max_completion_length=max_completion_len,
        temperature=temperature,
        beta=beta,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=seed,
        bf16=(use_cuda and dtype == "bfloat16"),
        fp16=(use_cuda and dtype == "float16"),
        gradient_checkpointing=use_cuda,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=cfg,
        train_dataset=dataset,
        processing_class=tok,
        peft_config=lora,
    )
    train_out = trainer.train()

    final_dir = out_dir / "final_adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(final_dir))
    tok.save_pretrained(str(final_dir))

    summary = {
        "model_id": model_id, "out_dir": str(out_dir), "reward": reward_func.__name__,
        "group_size": group_size, "n_prompts": len(dataset), "lora_r": lora_r,
        "beta": beta, "adapter": str(final_dir),
        "train_loss": float(getattr(train_out, "training_loss", float("nan"))),
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reward", required=True, choices=["math", "refusal-format"])
    ap.add_argument("--prompts", required=True, help="prompt pool JSON")
    ap.add_argument("--reference-answers", default=None,
                    help="reference answers for --reward math (tolerant loader)")
    ap.add_argument("--model", default="1.5b", help="cli_alias from config.yaml")
    ap.add_argument("--model-path", default=None, help="local base dir (tests)")
    ap.add_argument("--out-dir", default="checkpoints/r1_1.5b_grpo")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--micro-batch", type=int, default=None,
                    help="per-device batch (< group-size to cut logit-buffer memory)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-prompt-len", type=int, default=640)
    ap.add_argument("--max-completion-len", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--n", type=int, default=None, help="cap #prompts")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()

    if args.model_path:
        model_id, dtype = args.model_path, "float32"
        family = "deepseek"
    else:
        from src.config import model_tuple
        from src.model_adapters import family_of
        model_id, _short, dtype = model_tuple(args.model)
        family = family_of(model_id)

    # tokenizer for prompt formatting (cheap, reused by the trainer)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)

    ds = build_dataset(args.prompts, args.reward, tokenizer=tok, family=family,
                       reference_path=args.reference_answers, n=args.n)
    log.info("built %d prompts for reward=%s", len(ds), args.reward)

    reward_func = make_math_reward() if args.reward == "math" else make_refusal_reward()
    summary = train_grpo(
        ds, reward_func, model_id=model_id, out_dir=args.out_dir, dtype=dtype,
        group_size=args.group_size, micro_batch=args.micro_batch, epochs=args.epochs, lr=args.lr,
        grad_accum=args.grad_accum, max_prompt_len=args.max_prompt_len,
        max_completion_len=args.max_completion_len, temperature=args.temperature,
        beta=args.beta, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        seed=args.seed, max_steps=args.max_steps,
    )
    log.info("done — adapter: %s", summary["adapter"])


if __name__ == "__main__":
    main()
