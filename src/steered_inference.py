"""
Phase 7: Steered model inference.

Applies a steering vector to the residual stream at a specified layer during
autoregressive generation, implementing Huang et al. Equation 3:

    h' = h − α · (r^T h) · r      [subtract mode — reduce the behaviour]
    h' = h + α · (r^T h) · r      [add mode     — amplify the behaviour]

Three conditions are compared for each (behaviour, α):
  - vanilla:             no steering (α = 0 baseline)
  - single_direction:   Venhoff-style difference-of-means vector
  - manifold_projected: our manifold-projected vector (at auto_k)

Requires: torch, transformers  (pip install .[gpu])
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


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
        h = output[0].float()          # (batch, seq, hidden)
        r = self._r                    # (hidden,)
        proj = torch.einsum("bsd,d->bs", h, r).unsqueeze(-1)   # (batch, seq, 1)
        delta = self.alpha * proj * r.view(1, 1, -1)
        if self.mode == "subtract":
            h = h - delta
        else:
            h = h + delta
        return (h.to(output[0].dtype),) + output[1:]

    def generate(
        self,
        instruction: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> dict:
        """
        Generate a steered reasoning chain.

        Returns:
            {instruction, chain, n_tokens, alpha, mode, layer}
        """
        import torch
        from src.chain_gen import format_prompt

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
    max_new_tokens: int = 2048,
    save_path: Optional[Path] = None,
) -> list[dict]:
    """
    Run the main comparison experiment.

    For every (behaviour, α, method, task) combination:
      - vanilla:             unsteered generation (only at α=0)
      - single_direction:    Venhoff vector
      - manifold_projected:  our vector (auto_k)

    Checkpoints to *save_path* after every (behaviour, α, method) sweep so
    the experiment can be resumed after interruption.

    Returns:
        List of result dicts: {behaviour, method, alpha, task_id, chain,
                               n_tokens, layer}
    """
    import torch
    from src.chain_gen import generate_chain

    results: list[dict] = []
    save_path = Path(save_path) if save_path else None

    if save_path and save_path.exists():
        with open(save_path) as f:
            results = json.load(f)
        logger.info(f"Resuming: {len(results)} results already saved")

    done = {
        (r["behaviour"], r["method"], r["alpha"], r["task_id"])
        for r in results
    }

    for beh, vecs in steering_vectors.items():
        layer = vecs["layer"]
        auto_k = vecs["auto_k"]

        for alpha in alpha_values:
            methods = [("vanilla", None)]
            if alpha > 0:
                methods += [
                    ("single_direction", vecs["single_direction"]),
                    ("manifold_projected", vecs["manifold_projected"][auto_k]),
                ]

            for method_name, vec in methods:
                for task in tqdm(
                    tasks,
                    desc=f"{beh[:4]} α={alpha:.1f} {method_name}",
                    leave=False,
                ):
                    key = (beh, method_name, alpha, task["id"])
                    if key in done:
                        continue

                    if vec is None:
                        r = generate_chain(model, tokenizer, task["prompt"],
                                           max_new_tokens)
                    else:
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
