"""Supervised fine-tuning (LoRA and full-parameter) for the safety post-training
intervention.

Trains a reasoning model (default R1-1.5B) on the contrastive dataset with a
**completion-only** loss (prompt tokens masked to -100), distribution-matched to
how the model generates (prompt ends with ``<think>\\n``; the completion carries
the reasoning, ``</think>``, and the answer). Supports dose-response runs and a
size-matched non-safety control. ``train_lora`` is the pt02 path;
``train_full`` (pt02c) removes the LoRA-vs-full-FT regime approximation by
matching the published STAR-1 recipe (full-parameter SFT).

Heavy dependencies (torch / transformers / peft) are imported lazily inside the
functions so importing this module stays cheap (e.g. for unit tests of the
tokenisation/masking logic, which need only a tokenizer).

Runs on the DGX Spark (CUDA, bf16). For a 1.5B model with LoRA this fits in a
few GB and a STAR-1-scale (~1-2k example) run completes in minutes
(see ``README.md``). Requires ``pip install .[gpu] peft``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# LoRA adapters on attention + MLP projections (Qwen2 naming, shared by R1-Distill).
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ── Tokenisation with prompt masking ─────────────────────────────────────────

def tokenize_example(
    prompt_text: str,
    completion_text: str,
    tokenizer,
    max_len: int = 1024,
) -> dict:
    """Tokenise one (prompt, completion) into input_ids/labels with the prompt
    masked (-100) and an EOS appended to the completion.

    ``add_special_tokens=False`` because the chat-template prompt string already
    contains the special tokens (adding them again would duplicate BOS).
    Returns python lists (the collator tensorises).
    """
    p_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    c_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token_id
    if eos is not None:
        c_ids = c_ids + [eos]

    input_ids = p_ids + c_ids
    labels = [-100] * len(p_ids) + list(c_ids)

    if len(input_ids) > max_len:
        # Keep the head (prompt + as much completion as fits). Completions are
        # short for this task, so truncation should be rare; warn if it bites.
        logger.warning("example exceeds max_len=%d (%d toks); truncating tail",
                       max_len, len(input_ids))
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]
    return {"input_ids": input_ids, "labels": labels}


class SFTCollator:
    """Right-pad a batch of {input_ids, labels} to the batch max length."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        import torch
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            n = len(b["input_ids"])
            pad = maxlen - n
            input_ids.append(b["input_ids"] + [self.pad_token_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append([1] * n + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


# ── Model construction ───────────────────────────────────────────────────────

def build_peft_model(
    model_id: str,
    *,
    dtype: str = "bfloat16",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[list[str]] = None,
):
    """Load the base model + tokenizer and wrap it with a LoRA adapter."""
    from peft import LoraConfig, get_peft_model
    from src.chain_gen import load_model

    model, tokenizer = load_model(model_id, dtype=dtype)
    model.train()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()  # needed when base is frozen + grad ckpt
    if getattr(model, "config", None) is not None:
        model.config.use_cache = False

    cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules or DEFAULT_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)
    try:
        model.print_trainable_parameters()
    except Exception:  # pragma: no cover
        pass
    return model, tokenizer


# ── Training ─────────────────────────────────────────────────────────────────

def train_lora(
    sft_examples: list[dict],
    model_id: str,
    output_dir,
    *,
    dtype: str = "bfloat16",
    epochs: float = 3.0,
    lr: float = 2e-4,
    batch_size: int = 4,
    grad_accum: int = 4,
    max_len: int = 1024,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    seed: int = 42,
    merge: bool = False,
) -> dict:
    """Run one LoRA SFT and save the adapter (and optionally a merged model).

    ``sft_examples`` is the output of ``contrastive.records_to_sft`` (built with
    THIS tokenizer's family). Returns a small summary dict.
    """
    import torch
    from transformers import Trainer, TrainingArguments

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = build_peft_model(
        model_id, dtype=dtype, lora_r=lora_r, lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    tokenised = [
        tokenize_example(ex["prompt_text"], ex["completion_text"], tokenizer, max_len)
        for ex in sft_examples
    ]

    use_cuda = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=seed,
        bf16=(use_cuda and dtype == "bfloat16"),
        fp16=(use_cuda and dtype == "float16"),
        gradient_checkpointing=use_cuda,
        remove_unused_columns=False,
    )

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenised,
        data_collator=SFTCollator(pad_id),
    )
    train_out = trainer.train()

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("saved LoRA adapter -> %s", output_dir)

    merged_dir = None
    if merge:
        merged_dir = output_dir / "merged"
        merged = model.merge_and_unload()
        merged.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))
        logger.info("saved merged model -> %s", merged_dir)

    return {
        "output_dir": str(output_dir),
        "merged_dir": str(merged_dir) if merged_dir else None,
        "n_examples": len(sft_examples),
        "train_loss": float(getattr(train_out, "training_loss", float("nan"))),
        "epochs": epochs,
        "model_id": model_id,
    }


# ── Full-parameter SFT (pt02c) ───────────────────────────────────────────────

def select_full_ft_optim() -> str:
    """Pick the ONE memory-lean optimizer route for full-parameter FT.

    fp32 AdamW states alone are ~14 GB for a 1.5B model, so a 24 GB RTX 4090
    can't run the textbook recipe. Route: **8-bit Adam** (2 bytes/param state)
    whenever it can actually run (CUDA + bitsandbytes importable — bnb kernels
    are CUDA-only), else **Adafactor** (sub-linear state; also the CPU/unit-test
    path). Both are driven through ``TrainingArguments(optim=...)``.
    """
    import importlib.util

    import torch

    if torch.cuda.is_available() and importlib.util.find_spec("bitsandbytes") is not None:
        return "adamw_bnb_8bit"
    return "adafactor"


def train_full(
    sft_examples: list[dict],
    model_id: str,
    output_dir,
    *,
    dtype: str = "bfloat16",
    epochs: float = 5.0,
    lr: float = 1e-5,
    batch_size: int = 2,
    grad_accum: int = 64,
    max_len: int = 4096,
    warmup_ratio: float = 0.05,
    weight_decay: float = 1e-4,
    adam_beta2: float = 0.95,
    seed: int = 42,
    max_steps: Optional[int] = None,
    device_map: str = "auto",
) -> dict:
    """Full-parameter SFT (no LoRA) matching the published STAR-1 recipe.

    Recipe (METHODOLOGY_SAFETY_SPILLOVER_2026-07-03.md §4): full SFT, bf16,
    5 epochs, LR 1e-5 cosine + 5% warmup, AdamW betas 0.9/0.95, wd 1e-4,
    effective batch 128, completion-only loss; seq capped at 4096 like the
    LoRA arms (dataset p99 << 4096). Defaults reproduce it with
    ``batch_size * grad_accum = 128``.

    24 GB budget (1.5B): bf16 weights+grads ~7 GB, 8-bit Adam states ~3.5 GB
    (or Adafactor less), gradient checkpointing for activations at seq 4096.
    ``max_steps`` (optional) caps optimizer steps for smoke tests. Saves a
    plain ``from_pretrained``-loadable checkpoint into ``output_dir`` (what
    ``04_extract_activations.py --model-path`` expects).
    """
    import torch
    from transformers import Trainer, TrainingArguments

    from src.chain_gen import load_model

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(model_id, dtype=dtype, device_map=device_map)
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    if getattr(model, "config", None) is not None:
        model.config.use_cache = False

    tokenised = [
        tokenize_example(ex["prompt_text"], ex["completion_text"], tokenizer, max_len)
        for ex in sft_examples
    ]

    use_cuda = torch.cuda.is_available()
    optim = select_full_ft_optim()
    logger.info("full-FT optimizer route: %s (effective batch %d = %d x %d)",
                optim, batch_size * grad_accum, batch_size, grad_accum)
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        max_steps=max_steps if max_steps is not None else -1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        adam_beta1=0.9,
        adam_beta2=adam_beta2,
        optim=optim,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        seed=seed,
        bf16=(use_cuda and dtype == "bfloat16"),
        fp16=(use_cuda and dtype == "float16"),
        gradient_checkpointing=use_cuda,
        # full-FT targets CUDA pods; the non-CUDA path (unit tests) runs on
        # plain CPU — MPS is deliberately excluded (no bnb, memory-unsafe).
        use_cpu=not use_cuda,
        remove_unused_columns=False,
    )

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenised,
        data_collator=SFTCollator(pad_id),
    )
    train_out = trainer.train()

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("saved full-FT checkpoint -> %s", output_dir)

    return {
        "output_dir": str(output_dir),
        "n_examples": len(sft_examples),
        "steps": int(trainer.state.global_step),
        "train_loss": float(getattr(train_out, "training_loss", float("nan"))),
        "epochs": epochs,
        "effective_batch": batch_size * grad_accum,
        "optim": optim,
        "model_id": model_id,
    }
