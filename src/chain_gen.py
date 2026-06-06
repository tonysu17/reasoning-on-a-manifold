"""
Phase 2: Reasoning chain generation with DeepSeek-R1-Distill.

Generates one reasoning chain per task using greedy decoding (temperature=0),
following Venhoff et al.'s protocol.  Each record stores the prompt template
alongside the chain text so that Phase 4 can reconstruct exact token positions
during activation extraction.

Requires: torch, transformers, accelerate  (pip install .[gpu])
GPU:       RTX 4090 / A100 recommended; 1.5B model fits in ~4 GB VRAM.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from src.model_adapters import (
    DEEPSEEK, GPT_OSS, family_of,
    format_prompt as _adapter_format_prompt, split_reasoning_final,
)

logger = logging.getLogger(__name__)


def _seed_torch(seed: int) -> None:
    """Set torch RNG state for reproducible sampling.

    Only meaningful when `do_sample=True`. Greedy decoding (T=0) ignores
    the RNG state entirely.

    Used by `generate_chain` to support the M4.5 multi-seed regeneration
    path (synthesis §M4.5).
    """
    import torch
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


# ── Prompt template ───────────────────────────────────────────────────────────
# Prompt formatting is delegated to src/model_adapters so the same pipeline can
# drive DeepSeek-R1-Distill (<think> CoT), gpt-oss-20b (harmony analysis channel)
# and non-thinking base models. The default (family="deepseek") reproduces the
# original behaviour exactly, including the manual <think> fallback.


def format_prompt(
    tokenizer,
    instruction: str,
    *,
    family: str = DEEPSEEK,
    model_id: "Optional[str]" = None,
    reasoning_effort: str = "high",
) -> str:
    """Return the full prompt string (ready to tokenise and feed to the model)."""
    fam = family_of(model_id) if model_id else family
    return _adapter_format_prompt(
        tokenizer, instruction, family=fam, reasoning_effort=reasoning_effort
    )


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(
    model_id: str,
    dtype: str = "float16",
    use_4bit: bool = False,
    device_map: str = "auto",
    cache_dir: Optional[str] = None,
    attn_implementation: Optional[str] = None,
    model_kwargs: Optional[dict] = None,
):
    """Load a causal-LM and tokenizer.

    Works for DeepSeek-R1-Distill, Qwen base/instruct, and gpt-oss-20b. For
    gpt-oss on a non-Hopper GPU (e.g. the DGX Spark) pass dtype="bfloat16" and,
    if the learned attention sinks need it, attn_implementation="eager" and/or
    model_kwargs={"use_kernels": True}; these are no-ops for the DeepSeek path.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # On Apple Silicon, bfloat16 is more stable than float16 under MPS
    if torch.backends.mps.is_available() and dtype == "float16":
        dtype = "bfloat16"
        logger.info("Apple Silicon detected — using bfloat16 for MPS stability")

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                   "float32": torch.float32}[dtype]

    # Set device explicitly — "auto" is unreliable with some accelerate versions
    if device_map == "auto":
        if torch.cuda.is_available():
            device_map = {"": "cuda:0"}
        elif torch.backends.mps.is_available():
            device_map = {"": "mps"}
        else:
            device_map = {"": "cpu"}

    quant_cfg = None
    if use_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, cache_dir=cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    extra = dict(model_kwargs or {})
    if attn_implementation:
        extra["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=quant_cfg,
        cache_dir=cache_dir,
        **extra,
    )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    device = next(model.parameters()).device
    logger.info(f"Loaded {model_id}: {n_params:.1f}B params on {device}")
    return model, tokenizer


# ── Single-chain generation ───────────────────────────────────────────────────

def generate_chain(
    model,
    tokenizer,
    instruction: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    seed: int = 0,
    *,
    family: str = DEEPSEEK,
    model_id: "Optional[str]" = None,
    reasoning_effort: str = "high",
) -> dict:
    """
    Generate one reasoning chain for an instruction.

    `family` / `model_id` select the prompt format and how the completion is
    split into reasoning vs answer. The default (deepseek) is unchanged; for
    gpt-oss the harmony `analysis` channel becomes `chain` and the `final`
    channel is stored as `final_answer`.

    `seed` controls torch RNG state and is only meaningful when
    `temperature > 0` (do_sample=True). Greedy decoding (T=0) is
    deterministic regardless of seed.

    Returns a dict with:
        instruction, prompt, chain, full_text, n_tokens, seed, temperature
        (+ final_answer, reasoning_effort, family for gpt-oss)
    """
    import torch

    _seed_torch(seed)

    fam = family_of(model_id) if model_id else family
    prompt = format_prompt(
        tokenizer, instruction, family=fam, reasoning_effort=reasoning_effort
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = outputs[0][prompt_len:]

    if fam == GPT_OSS:
        # Keep special tokens so the harmony channel markers survive the decode,
        # then split analysis (reasoning we annotate) from final (the answer).
        decoded = tokenizer.decode(new_ids, skip_special_tokens=False)
        chain, final_answer = split_reasoning_final(decoded, family=GPT_OSS)
    else:
        chain = tokenizer.decode(new_ids, skip_special_tokens=True)
        final_answer = ""

    record = {
        "instruction": instruction,
        "prompt": prompt,
        "chain": chain,
        "full_text": prompt + chain,
        "n_tokens": len(new_ids),
        "seed": int(seed),
        "temperature": float(temperature),
    }
    if fam == GPT_OSS:
        record["final_answer"] = final_answer
        record["reasoning_effort"] = reasoning_effort
        record["family"] = fam
    return record


# ── Batch generation ──────────────────────────────────────────────────────────

def generate_chains(
    model,
    tokenizer,
    tasks: list[dict],
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    save_path: Optional[Path] = None,
    checkpoint_every: int = 10,
    seed: int = 0,
    dedup_keys: tuple = ("task_id",),
    *,
    family: str = DEEPSEEK,
    model_id: Optional[str] = None,
    reasoning_effort: str = "high",
) -> list[dict]:
    """
    Generate reasoning chains for a list of tasks.

    Checkpoints every `checkpoint_every` tasks so you can resume after
    interruption by re-running the same command.

    `seed` controls torch RNG state for sampled generation. Only
    meaningful when `temperature > 0`. The seed is recorded on every
    chain record so multi-seed re-generation (synthesis §M4.5) can mark
    runs.

    Args:
        model / tokenizer: loaded via load_model()
        tasks:             list of {id, prompt, category, difficulty}
        save_path:         JSON file for output (and checkpoints)
        seed:              torch RNG seed for sampled decoding
        dedup_keys:        tuple of dict-keys identifying a chain for resume.
                           Default ("task_id",) is the original behaviour;
                           pass ("task_id", "seed") for multi-seed runs.

    Returns:
        List of chain records: {task_id, category, instruction, prompt,
                                chain, full_text, n_tokens, seed, temperature}
    """
    chains: list[dict] = []
    save_path = Path(save_path) if save_path else None

    if save_path and save_path.exists():
        from src.config import backup_existing
        backup_existing(save_path)  # snapshot prior chains before this run touches them
        with open(save_path) as f:
            chains = json.load(f)
        logger.info(f"Resuming from checkpoint: {len(chains)}/{len(tasks)} done")

    def _chain_key(rec: dict) -> tuple:
        return tuple(rec.get(k) for k in dedup_keys)

    completed = {_chain_key(c) for c in chains}

    for task in tqdm(tasks, initial=len(chains), total=len(tasks),
                     desc=f"Generating chains (seed={seed}, T={temperature})"):
        tid = task["id"]
        if _chain_key({"task_id": tid, "seed": seed}) in completed:
            continue

        try:
            result = generate_chain(
                model, tokenizer, task["prompt"],
                max_new_tokens=max_new_tokens,
                temperature=temperature, seed=seed,
                family=family, model_id=model_id,
                reasoning_effort=reasoning_effort,
            )
            record = {
                "task_id": tid,
                "category": task.get("category", "unknown"),
                "instruction": task["prompt"],
                "prompt": result["prompt"],
                "chain": result["chain"],
                "full_text": result["full_text"],
                "n_tokens": result["n_tokens"],
                "seed": int(seed),
                "temperature": float(temperature),
            }
            # gpt-oss carries the harmony final answer + effort alongside the CoT.
            if "final_answer" in result:
                record["final_answer"] = result["final_answer"]
                record["reasoning_effort"] = result.get("reasoning_effort")
                record["family"] = result.get("family")
        except Exception as exc:
            logger.warning(f"Task {tid} failed: {exc}")
            record = {
                "task_id": tid,
                "category": task.get("category", "unknown"),
                "instruction": task["prompt"],
                "prompt": "",
                "chain": "",
                "full_text": "",
                "n_tokens": 0,
                "seed": int(seed),
                "temperature": float(temperature),
                "error": str(exc),
            }

        chains.append(record)
        completed.add(_chain_key(record))

        if save_path and len(chains) % checkpoint_every == 0:
            _save_json(chains, save_path)
            logger.info(f"  checkpoint: {len(chains)}/{len(tasks)}")

    if save_path:
        _save_json(chains, save_path)

    success = sum(1 for c in chains if c["n_tokens"] > 0)
    tokens = [c["n_tokens"] for c in chains if c["n_tokens"] > 0]
    if tokens:
        logger.info(
            f"Done: {success}/{len(chains)} chains generated  "
            f"[mean={sum(tokens)/len(tokens):.0f}, "
            f"min={min(tokens)}, max={max(tokens)} tokens]"
        )
    return chains


def load_chains(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _save_json(data, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.rename(path)
