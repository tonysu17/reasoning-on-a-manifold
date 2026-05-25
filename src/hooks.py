"""
Residual-stream activation capture via PyTorch forward hooks.

Hooks attach to the output of each transformer layer (the residual stream
post-LayerNorm, same convention as Huang et al. and Venhoff et al.).
Activations are immediately detached and moved to CPU to avoid GPU OOM when
processing many long chains.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ActivationCache:
    """
    Context manager that captures residual-stream activations for specified layers.

    Usage:
        with ActivationCache(model, layers=[12, 20, 27]) as cache:
            with torch.no_grad():
                model(**inputs)
        acts_layer12 = cache[12]  # shape: (batch, seq_len, hidden_dim)
    """

    def __init__(self, model: nn.Module, layers: Optional[list[int]] = None):
        self._model = model
        # Both Qwen and Llama DeepSeek-R1-Distill models expose model.model.layers
        self._all_layers: nn.ModuleList = model.model.layers
        n = len(self._all_layers)
        self.layers: list[int] = layers if layers is not None else list(range(n))
        self._cache: dict[int, torch.Tensor] = {}
        self._hooks: list = []

    # ── Context manager ──────────────────────────────────────────────────

    def __enter__(self) -> "ActivationCache":
        self._register()
        return self

    def __exit__(self, *_) -> None:
        self._remove()

    # ── Public API ───────────────────────────────────────────────────────

    def __getitem__(self, layer_idx: int) -> torch.Tensor:
        """Return cached activation for *layer_idx*. Shape: (batch, seq, hidden)."""
        return self._cache[layer_idx]

    def mean_at_positions(
        self,
        layer_idx: int,
        positions: list[int],
        batch_idx: int = 0,
    ) -> torch.Tensor:
        """
        Mean-pool activations at *positions* for a given layer.

        Returns shape: (hidden_dim,)
        """
        acts = self._cache[layer_idx][batch_idx, positions, :]  # (n_pos, hidden)
        return acts.mean(dim=0)

    def clear(self) -> None:
        """Clear cached tensors (keep hooks registered)."""
        self._cache.clear()

    # ── Internal ─────────────────────────────────────────────────────────

    def _make_hook(self, idx: int):
        def _hook(module, input, output):
            # HF transformer blocks may return either a tuple
            # (hidden_states, present_kv, ...) or just the hidden_states tensor
            # depending on the version. Handle both — we want the full 3D
            # (batch, seq_len, hidden_dim) tensor in the cache.
            h = output[0] if isinstance(output, tuple) else output
            self._cache[idx] = h.detach().cpu().float()
        return _hook

    def _register(self) -> None:
        self._remove()
        for idx in self.layers:
            h = self._all_layers[idx].register_forward_hook(self._make_hook(idx))
            self._hooks.append(h)

    def _remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
