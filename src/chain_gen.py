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

logger = logging.getLogger(__name__)


# ── Prompt template ───────────────────────────────────────────────────────────
# DeepSeek-R1-Distill models need <think>\n appended to the assistant turn to
# force the model into chain-of-thought mode.  We prefer apply_chat_template
# when the tokenizer supports it; otherwise fall back to the manual format.

_MANUAL_TEMPLATE = (
    "<|begin▁of▁sentence|>"
    "<|User|>{instruction}"
    "<|Assistant|><think>\n"
)


def format_prompt(tokenizer, instruction: str) -> str:
    """Return the full prompt string (ready to tokenise and feed to the model)."""
    try:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
        # Append <think>\n to enter reasoning mode
        if not text.rstrip().endswith("<think>"):
            text = text.rstrip() + "<think>\n"
        return text
    except Exception:
        return _MANUAL_TEMPLATE.format(instruction=instruction)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(
    model_id: str,
    dtype: str = "float16",
    use_4bit: bool = False,
    device_map: str = "auto",
    cache_dir: Optional[str] = None,
):
    """Load a DeepSeek-R1-Distill model and tokenizer."""
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

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=quant_cfg,
        cache_dir=cache_dir,
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
) -> dict:
    """
    Generate one reasoning chain for an instruction.

    Returns a dict with:
        instruction, prompt, chain, full_text, n_tokens
    """
    import torch

    prompt = format_prompt(tokenizer, instruction)
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
    chain = tokenizer.decode(new_ids, skip_special_tokens=True)

    return {
        "instruction": instruction,
        "prompt": prompt,
        "chain": chain,
        "full_text": prompt + chain,
        "n_tokens": len(new_ids),
    }


# ── Batch generation ──────────────────────────────────────────────────────────

def generate_chains(
    model,
    tokenizer,
    tasks: list[dict],
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    save_path: Optional[Path] = None,
    checkpoint_every: int = 50,
) -> list[dict]:
    """
    Generate reasoning chains for a list of tasks.

    Checkpoints every `checkpoint_every` tasks so you can resume after
    interruption by re-running the same command.

    Args:
        model / tokenizer: loaded via load_model()
        tasks:             list of {id, prompt, category, difficulty}
        save_path:         JSON file for output (and checkpoints)

    Returns:
        List of chain records: {task_id, category, instruction,
                                prompt, chain, full_text, n_tokens}
    """
    chains: list[dict] = []
    save_path = Path(save_path) if save_path else None

    # Resume from checkpoint
    start = 0
    if save_path and save_path.exists():
        with open(save_path) as f:
            chains = json.load(f)
        start = len(chains)
        logger.info(f"Resuming from checkpoint: {start}/{len(tasks)} done")

    completed_ids = {c["task_id"] for c in chains}

    for task in tqdm(tasks[start:], initial=start, total=len(tasks),
                     desc="Generating chains"):
        tid = task["id"]
        if tid in completed_ids:
            continue

        try:
            result = generate_chain(model, tokenizer, task["prompt"],
                                    max_new_tokens, temperature)
            record = {
                "task_id": tid,
                "category": task.get("category", "unknown"),
                "instruction": task["prompt"],
                "prompt": result["prompt"],
                "chain": result["chain"],
                "full_text": result["full_text"],
                "n_tokens": result["n_tokens"],
            }
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
                "error": str(exc),
            }

        chains.append(record)

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
