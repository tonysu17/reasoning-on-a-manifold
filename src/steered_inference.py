"""
Phase 7: Steered model inference.

Applies a steering vector to the residual stream at a specified layer during
autoregressive generation, implementing Huang et al. Equation 3:

    h' = h − α · (r^T h) · r      [subtract mode — reduce the behaviour]
    h' = h + α · (r^T h) · r      [add mode     — amplify the behaviour]

(The hook applies to every position, prompt prefill included, matching Huang;
vectors from src/steering.py are unit-norm, so α is the full scale knob.)

Conditions compared per behaviour:
  - vanilla:            unsteered baseline — generated ONCE per task and
                        shared across behaviours/α (greedy decoding makes
                        per-α regeneration byte-identical; the old per-α
                        vanilla arm multiplied generation AND re-annotation
                        cost ~8× for zero information)
  - single_direction:   Venhoff-style difference-of-means vector (α > 0)
  - manifold_projected: our PCA-subspace-projected vector at auto_k (α > 0)
  - random_direction:   norm-matched random unit vector, fixed seed per
                        behaviour (α > 0) — the control that licenses causal
                        language: without it, "manifold beats single
                        direction" can't be separated from "any perturbation
                        of this magnitude changes behaviour fractions"

Default max_new_tokens follows configs/config.yaml generation.max_new_tokens
(8192). The previous 2048 default silently truncated steered chains — the
exact mistake memorialised by data/chains_R1-1.5B_BAD_2048cap.json — which
confounds α effects with truncation effects.

Requires: torch, transformers  (pip install .[gpu])
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

#: Pseudo-behaviour key for the shared unsteered baseline records.
SHARED_BASELINE = "shared"


def default_max_new_tokens() -> int:
    """chains.max_new_tokens from config.yaml (8192), with a safe fallback.

    (The config section is ``chains:`` — an earlier version read a nonexistent
    ``generation:`` key and always silently used the fallback.)"""
    try:
        from src.config import load_config
        cfg = load_config()
        v = (cfg.get("chains", {}) or {}).get("max_new_tokens")
        if v is None:
            v = (cfg.get("generation", {}) or {}).get("max_new_tokens", 8192)
        return int(v)
    except Exception:
        return 8192


def random_direction_like(reference: np.ndarray, seed_key: str) -> np.ndarray:
    """Norm-matched random control vector, reproducible across runs.

    Seeded from a stable digest of *seed_key* (NOT the salted builtin hash),
    drawn isotropically and rescaled to ||reference||. Note this matches the
    VECTOR norm, not the delivered energy: the hook applies α·(r·h)·r, and a
    random r has smaller |r·h| than a behaviour-aligned direction — so this
    arm is a generic-perturbation floor, not an energy-matched twin. Describe
    it as such in any writeup.
    """
    seed = int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(reference.shape[0]).astype(np.float64)
    v /= np.linalg.norm(v)
    ref_norm = float(np.linalg.norm(reference))
    if ref_norm <= 0:
        ref_norm = 1.0
    return (v * ref_norm).astype(np.asarray(reference).dtype, copy=False)


class SteeredModel:
    """
    Wraps a HuggingFace model to apply one steering vector during generation.

    The hook is active only inside generate() to avoid polluting other calls.
    """

    def __init__(
        self,
        model,
        tokenizer,
        vector: np.ndarray,
        layer: int,
        alpha: float = 1.0,
        mode: str = "subtract",
    ):
        import torch
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.alpha = alpha
        self.mode = mode
        self._hook_handle = None

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        self._r = torch.tensor(vector, dtype=torch.float32).to(device)

    def _hook_fn(self, module, input, output):
        import torch
        # transformers <5 decoder layers return a tuple (hidden, ...); 5.x can
        # return the bare hidden-state tensor. On a bare tensor, output[0]
        # would silently index the first BATCH element — handle both shapes.
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        h = hidden.float()             # (batch, seq, hidden)
        r = self._r                    # (hidden,)
        proj = torch.einsum("bsd,d->bs", h, r).unsqueeze(-1)   # (batch, seq, 1)
        delta = self.alpha * proj * r.view(1, 1, -1)
        if self.mode == "subtract":
            h = h - delta
        else:
            h = h + delta
        h = h.to(hidden.dtype)
        return ((h,) + output[1:]) if is_tuple else h

    def generate(
        self,
        instruction: str,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> dict:
        """
        Generate a steered reasoning chain.

        Returns:
            {instruction, chain, n_tokens, alpha, mode, layer}
        """
        import torch
        from src.chain_gen import format_prompt

        if max_new_tokens is None:
            max_new_tokens = default_max_new_tokens()
        prompt = format_prompt(self.tokenizer, instruction)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs.input_ids.shape[1]

        hook = self.model.model.layers[self.layer].register_forward_hook(self._hook_fn)
        try:
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=(temperature > 0),
                    temperature=temperature if temperature > 0 else 1.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        finally:
            hook.remove()

        new_ids = out[0][prompt_len:]
        chain = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return {
            "instruction": instruction,
            "chain": chain,
            "n_tokens": len(new_ids),
            "alpha": self.alpha,
            "mode": self.mode,
            "layer": self.layer,
        }


def run_steering_experiment(
    model,
    tokenizer,
    tasks: list[dict],
    steering_vectors: dict,
    alpha_values: list[float],
    max_new_tokens: Optional[int] = None,
    save_path: Optional[Path] = None,
    include_random_control: bool = True,
) -> list[dict]:
    """
    Run the main comparison experiment.

    Arms (see module docstring):
      - vanilla: ONE unsteered generation per task, recorded once under
        behaviour=SHARED_BASELINE / alpha=0.0. Greedy decoding makes per-α
        copies byte-identical, so the old per-(behaviour, α) vanilla sweep
        only multiplied generation + re-annotation cost.
      - single_direction / manifold_projected (auto_k) / random_direction
        (norm-matched control, fixed per-behaviour seed) at every α > 0.

    Checkpoints to *save_path* after every (behaviour, α, method) sweep so
    the experiment can be resumed after interruption. Vanilla records from
    pre-hoist checkpoints (behaviour-specific, per-α) are left untouched but
    not regenerated.

    Returns:
        List of result dicts: {behaviour, method, alpha, task_id, chain,
                               n_tokens, layer}
    """
    import torch
    from src.chain_gen import generate_chain

    if max_new_tokens is None:
        max_new_tokens = default_max_new_tokens()
    if max_new_tokens < 8192:
        logger.warning(f"max_new_tokens={max_new_tokens} < corpus cap 8192 — "
                       f"steered chains will truncate harder than the corpus "
                       f"did, confounding α effects with truncation "
                       f"(cf. chains_R1-1.5B_BAD_2048cap.json)")

    results: list[dict] = []
    save_path = Path(save_path) if save_path else None

    if save_path and save_path.exists():
        with open(save_path) as f:
            results = json.load(f)
        logger.info(f"Resuming: {len(results)} results already saved")
        # Pre-hoist checkpoints carry per-(behaviour, α) vanilla records that no
        # loop regenerates but that WOULD be re-annotated and would fake a
        # vanilla-vs-α curve in aggregation. Drop them on load.
        n_legacy = sum(1 for r in results
                       if r["method"] == "vanilla" and r["behaviour"] != SHARED_BASELINE)
        if n_legacy:
            results = [r for r in results
                       if not (r["method"] == "vanilla"
                               and r["behaviour"] != SHARED_BASELINE)]
            logger.warning(f"Dropped {n_legacy} legacy per-behaviour vanilla "
                           f"records (pre-hoist schema) from the resume file")

    done = {
        (r["behaviour"], r["method"], r["alpha"], r["task_id"])
        for r in results
    }

    # ── Shared unsteered baseline: one generation per task ────────────────
    for task in tqdm(tasks, desc="vanilla (shared baseline)", leave=False):
        key = (SHARED_BASELINE, "vanilla", 0.0, task["id"])
        if key in done:
            continue
        r = generate_chain(model, tokenizer, task["prompt"], max_new_tokens)
        results.append({
            "behaviour": SHARED_BASELINE,
            "method": "vanilla",
            "alpha": 0.0,
            "task_id": task["id"],
            "chain": r["chain"],
            "n_tokens": r["n_tokens"],
            "layer": None,
        })
        done.add(key)
    if save_path:
        _save_json(results, save_path)

    # ── Steered arms ───────────────────────────────────────────────────────
    steered_alphas = [a for a in alpha_values if a > 0]
    if len(steered_alphas) < len(alpha_values):
        logger.info("α=0 entries are covered by the shared vanilla baseline "
                    "(subtract-mode steering at α=0 is the identity)")

    for beh, vecs in steering_vectors.items():
        layer = vecs["layer"]

        # vecs["manifold_projected"] is keyed by k values 1, 3, 5, 10 and the
        # literal string "auto" (= projection at auto_k); "auto" is canonical.
        arms = [
            ("single_direction", vecs["single_direction"]),
            ("manifold_projected", vecs["manifold_projected"]["auto"]),
        ]
        if include_random_control:
            arms.append(("random_direction",
                         random_direction_like(vecs["single_direction"],
                                               f"random_direction|{beh}|L{layer}")))

        for alpha in steered_alphas:
            for method_name, vec in arms:
                for task in tqdm(
                    tasks,
                    desc=f"{beh[:4]} α={alpha:.1f} {method_name}",
                    leave=False,
                ):
                    key = (beh, method_name, alpha, task["id"])
                    if key in done:
                        continue

                    steered = SteeredModel(model, tokenizer, vec, layer,
                                           alpha=alpha, mode="subtract")
                    r = steered.generate(task["prompt"], max_new_tokens)

                    results.append({
                        "behaviour": beh,
                        "method": method_name,
                        "alpha": alpha,
                        "task_id": task["id"],
                        "chain": r["chain"],
                        "n_tokens": r["n_tokens"],
                        "layer": layer,
                    })
                    done.add(key)

                if save_path:
                    _save_json(results, save_path)

    logger.info(f"Steering experiment complete: {len(results)} total results")
    return results


def _save_json(data, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.rename(path)
