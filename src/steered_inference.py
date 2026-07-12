"""
Phase 7: Steered model inference.

Applies a steering vector to the residual stream at a specified layer during
autoregressive generation, implementing Huang et al. Equation 3:

    h' = h − α · (r^T h) · r      [subtract mode — reduce the behaviour]
    h' = h + α · (r^T h) · r      [add mode     — amplify the behaviour]

(The hook applies to every position, prompt prefill included, matching Huang;
vectors from src/steering.py are unit-norm, so α is the full scale knob.)

Conditions compared per behaviour (method label in brackets):
  - vanilla:            unsteered baseline — generated ONCE per task and
                        shared across behaviours/α (greedy decoding makes
                        per-α regeneration byte-identical).
  - single_direction:   Venhoff-style difference-of-means vector (α > 0).
  - manifold_k{1,3,5,10} / manifold_auto:  HEADLINE k-sweep — the diff-of-means
                        projected onto the behaviour's top-k PCA subspace, at
                        EVERY built k (not just auto, where manifold≈single,
                        cos 0.95–0.97). The granularity question is the fix the
                        adversarial review demanded.
  - random_subspace_k{k}:  single direction projected onto a RANDOM k-dim
                        orthonormal subspace (QR of a seeded Gaussian, matched
                        k, renormalised identically), R seeded replicates
                        averaged. The control that isolates "the behaviour's
                        PCA subspace matters" from "any k-dim projection +
                        renorm".
  - random_direction:   norm-matched random unit vector (α > 0) — now the
                        SANITY FLOOR, not the causal baseline: it injects ~19×
                        less energy (|rᵀh| ≈ 4.9 vs 94.6).
  - energy_matched_random:  random direction rescaled per-behaviour so its mean
                        |rᵀh| matches the behaviour arm's — the real generic-
                        energy floor the manifold/single arms must beat.
  - orthogonal_complement:  the OFF-subspace component (I−P_k)r_single alone,
                        to test whether the discarded component is pure
                        collateral (the mechanism behind any manifold advantage).

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


def _seed_from(seed_key: str) -> int:
    """Stable 64-bit seed from a string digest (NOT the salted builtin hash)."""
    return int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:8], "little")


def random_direction_like(reference: np.ndarray, seed_key: str) -> np.ndarray:
    """Norm-matched random control vector, reproducible across runs.

    Seeded from a stable digest of *seed_key* (NOT the salted builtin hash),
    drawn isotropically and rescaled to ||reference||. Note this matches the
    VECTOR norm, not the delivered energy: the hook applies α·(r·h)·r, and a
    random r has smaller |r·h| than a behaviour-aligned direction — so this
    arm is a generic-perturbation floor, not an energy-matched twin. Describe
    it as such in any writeup.
    """
    rng = np.random.default_rng(_seed_from(seed_key))
    v = rng.standard_normal(reference.shape[0]).astype(np.float64)
    v /= np.linalg.norm(v)
    ref_norm = float(np.linalg.norm(reference))
    if ref_norm <= 0:
        ref_norm = 1.0
    return (v * ref_norm).astype(np.asarray(reference).dtype, copy=False)


def random_subspace_projection(
    reference: np.ndarray, k: int, seed_key: str
) -> np.ndarray:
    """Project *reference* onto a RANDOM k-dim orthonormal subspace, renormalised.

    The QR of a seeded (hidden_dim × k) Gaussian gives a Haar-random orthonormal
    basis Q (hidden_dim × k); we return ``(Q Qᵀ r) / ‖Q Qᵀ r‖`` — the unit-norm
    projection of *reference* onto that subspace. Matched in k and renormalised
    IDENTICALLY to the manifold-projected vector (`manifold_projected_vector`),
    so the ONLY difference from the manifold arm is *which* k-dim subspace was
    used: the behaviour's own top-k PCA subspace vs a random one. This isolates
    "the behaviour's PCA subspace matters" from "any k-dim projection + renorm".

    Falls back to the norm-direction (unit-normalised *reference*) when k spans
    the ambient dimension or the projection collapses to ~0.
    """
    r = np.asarray(reference, dtype=np.float64)
    d = r.shape[0]
    k = int(max(1, min(k, d)))
    rng = np.random.default_rng(_seed_from(seed_key))
    G = rng.standard_normal((d, k))
    Q, _ = np.linalg.qr(G)              # (d, k), orthonormal columns
    coords = Q.T @ r                    # (k,)
    r_proj = Q @ coords                 # (d,) — projection back into ambient
    norm = float(np.linalg.norm(r_proj))
    if norm < 1e-10:
        v = r / (np.linalg.norm(r) or 1.0)
        return v.astype(np.asarray(reference).dtype, copy=False)
    return (r_proj / norm).astype(np.asarray(reference).dtype, copy=False)


def orthogonal_complement_vector(
    single_direction: np.ndarray, subspace_vector: np.ndarray
) -> np.ndarray:
    """The off-subspace component of *single_direction*, unit-normalised.

    Given the single direction r and the manifold-projected (in-subspace) unit
    vector p = P_k r / ‖P_k r‖, returns ``(r − (r·p)p) / ‖·‖`` — i.e. the part
    of r ORTHOGONAL to the behaviour's top-k subspace. Steering this alone tests
    whether the component the manifold projection DISCARDS is pure collateral
    (little on-target effect, much damage) — the mechanism behind any manifold
    advantage. Falls back to the in-subspace direction if r already lies in the
    subspace (zero complement).
    """
    r = np.asarray(single_direction, dtype=np.float64)
    p = np.asarray(subspace_vector, dtype=np.float64)
    p = p / (np.linalg.norm(p) or 1.0)
    r_perp = r - float(r @ p) * p
    norm = float(np.linalg.norm(r_perp))
    if norm < 1e-10:
        logger.warning("Orthogonal complement is ~0 (r lies in the subspace) — "
                       "falling back to the in-subspace direction")
        return p.astype(np.asarray(single_direction).dtype, copy=False)
    return (r_perp / norm).astype(np.asarray(single_direction).dtype, copy=False)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (0 if either is ~0)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float((a @ b) / (na * nb))


def retained_energy(single_direction: np.ndarray, subspace_vector: np.ndarray) -> float:
    """‖P_k r‖ / ‖r‖ — the fraction of the single direction's norm retained by
    the projection onto the behaviour's top-k subspace.

    *subspace_vector* is the UNIT-norm manifold-projected vector p = P_k r/‖P_k r‖.
    Since p has unit norm, ‖P_k r‖ = |r·p| (the projection coefficient), so the
    retained-energy ratio is |r·p| / ‖r‖. This BOUNDS how large any manifold
    effect can be relative to the single direction (a value near 1 means the
    discarded complement is tiny, so manifold≈single)."""
    r = np.asarray(single_direction, dtype=np.float64)
    p = np.asarray(subspace_vector, dtype=np.float64)
    p = p / (np.linalg.norm(p) or 1.0)
    nr = float(np.linalg.norm(r))
    if nr < 1e-12:
        return 0.0
    return float(abs(r @ p) / nr)


def energy_matched_scale(
    behaviour_mean_abs_proj: float, random_mean_abs_proj: float
) -> float:
    """Multiplicative gain that equalises injected energy across two arms.

    The hook injects ``α·scale·(v·h)·v``; with unit-norm v the per-position
    energy is ``α·scale·|v·h|``. To make a RANDOM direction deliver the same
    mean energy as the behaviour arm at the same α, scale it by
    ``E_behaviour / E_random`` where each E is the mean |v·h| over positions
    (measured on the SAME unperturbed stream — see ``measure_mean_abs_proj``).

    A random unit vector has smaller |v·h| than a behaviour-aligned one, so the
    gain is > 1 (it rescales the random direction UP). Returns 1.0 if the random
    energy is ~0 (degenerate; falls back to norm-matched behaviour)."""
    if random_mean_abs_proj is None or random_mean_abs_proj < 1e-12:
        return 1.0
    if behaviour_mean_abs_proj is None:
        return 1.0
    return float(behaviour_mean_abs_proj / random_mean_abs_proj)


def measure_mean_abs_proj(
    model, tokenizer, vector: np.ndarray, layer: int,
    tasks: list[dict], max_new_tokens: int = 64,
) -> float:
    """Forward-only probe of mean |vᵀh| for *vector* on *tasks* (no steering).

    Runs the model in ``mode="measure"`` (the hook records the projection on the
    UNPERTURBED stream and passes hidden states through untouched), so the
    behaviour and random arms are both measured against IDENTICAL activations —
    the well-defined common reference for ``energy_matched_scale``. A short
    ``max_new_tokens`` keeps calibration cheap; the projection statistic is
    stable over a few dozen positions. Requires the model (not unit-tested)."""
    probe = SteeredModel(model, tokenizer, vector, layer, alpha=0.0, mode="measure")
    sums, counts = 0.0, 0
    for task in tasks:
        out = probe.generate(task["prompt"], max_new_tokens=max_new_tokens)
        e = out.get("mean_abs_proj")
        if e is not None and probe._abs_proj_count:
            sums += probe._abs_proj_sum
            counts += probe._abs_proj_count
    return (sums / counts) if counts else 0.0


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
        energy_scale: float = 1.0,
        clamp_value: float = 0.0,
        clamp_gain: float = 1.0,
    ):
        import torch
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.alpha = alpha
        self.mode = mode
        #: Target coordinate for mode="clamp": h' = h + β·(c − rᵀh)·r moves the
        #: r-coordinate a fraction β (clamp_gain) of the way to the constant c at
        #: every position. c is data-derived (a class-mean coordinate the model
        #: actually exhibits); β<1 makes the bounded clamp DOSE-ABLE so its
        #: realized displacement can be swept to overlap the projective arm's
        #: (E10 P2 redesign — the matched-displacement fix).
        self.clamp_value = float(clamp_value)
        self.clamp_gain = float(clamp_gain)
        #: Extra multiplicative gain on the perturbation. =1.0 for behaviour /
        #: norm-matched arms; the energy-matched-random arm sets it so the
        #: delivered |αᵀh| matches the behaviour arm's mean projection energy
        #: (a random unit r has smaller |r·h| than a behaviour-aligned one, so
        #: this rescales it UP to inject equal energy). See energy_matched_scale.
        self.energy_scale = float(energy_scale)
        self._hook_handle = None
        #: Running accumulators for the mean |rᵀh| this vector saw during the
        #: last generate() — the measured per-position projection energy used to
        #: calibrate the energy-matched control. Reset at the start of generate().
        self._abs_proj_sum = 0.0
        self._abs_proj_count = 0
        #: Realized per-position coordinate displacement |Δ(rᵀh)| delivered by
        #: the intervention — the state-level on-target measure that lets clamp
        #: and projective arms be compared at matched displacement (E10 P2).
        self._disp_sum = 0.0
        self._disp_count = 0

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
        # Record the per-position projection energy |rᵀh| (pre-scale) so the
        # energy-matched control can be calibrated against the behaviour arm.
        self._abs_proj_sum += float(proj.abs().sum().item())
        self._abs_proj_count += int(proj.numel())
        if self.mode == "measure":
            # Probe-only: record |rᵀh| on the UNPERTURBED stream and pass the
            # hidden state through untouched. Used to calibrate the energy-
            # matched control against the behaviour arm on identical activations.
            return output
        if self.mode == "clamp":
            # Move the r-coordinate a fraction β toward the class-mean value: the
            # data-bounded intervention (interchange-swap analogue). β=1 is a full
            # clamp; β<1 doses it so displacement can be matched to the projective arm.
            coord_delta = self.clamp_gain * (self.clamp_value - proj)  # (b, s, 1)
        elif self.mode == "subtract":
            coord_delta = -(self.alpha * self.energy_scale) * proj
        else:  # "add"
            coord_delta = (self.alpha * self.energy_scale) * proj
        self._disp_sum += float(coord_delta.abs().sum().item())
        self._disp_count += int(coord_delta.numel())
        h = h + coord_delta * r.view(1, 1, -1)
        h = h.to(hidden.dtype)
        return ((h,) + output[1:]) if is_tuple else h

    def mean_abs_displacement(self) -> Optional[float]:
        """Mean realized |Δ(rᵀh)| the intervention delivered during the last
        generate() (None if unused). The matching variable for clamp-vs-
        projective comparisons (E10 P2)."""
        if self._disp_count == 0:
            return None
        return self._disp_sum / self._disp_count

    def mean_abs_projection(self) -> Optional[float]:
        """Mean |rᵀh| observed during the last generate() (None if unused).

        This is the UNSCALED projection magnitude (the energy a single
        unit-gain application would deliver), so it is comparable across vectors
        regardless of each arm's energy_scale."""
        if self._abs_proj_count == 0:
            return None
        return self._abs_proj_sum / self._abs_proj_count

    def generate(
        self,
        instruction: str,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> dict:
        """
        Generate a steered reasoning chain.

        ``temperature``>0 turns on sampling (``do_sample=True``); ``seed`` sets the
        torch RNG immediately before ``model.generate`` so each sample is
        reproducible AND — with a distinct seed per sample — genuinely different.
        At ``temperature==0`` (greedy) the seed is inert (decoding is
        deterministic). Mirrors ``src.chain_gen.generate_chain``'s seed handling.

        Returns:
            {instruction, chain, n_tokens, alpha, mode, layer, energy_scale,
             mean_abs_proj, temperature, seed}
        """
        import torch
        from src.chain_gen import format_prompt, _seed_torch

        if max_new_tokens is None:
            max_new_tokens = default_max_new_tokens()
        prompt = format_prompt(self.tokenizer, instruction)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs.input_ids.shape[1]

        # Seed BEFORE generate so sampled draws are reproducible/distinct-per-seed.
        # Greedy (temperature==0) ignores the RNG, so this is a harmless no-op there.
        if temperature > 0:
            _seed_torch(seed)

        self._abs_proj_sum = 0.0      # reset per-generation energy probe
        self._abs_proj_count = 0
        self._disp_sum = 0.0
        self._disp_count = 0
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
            "energy_scale": self.energy_scale,
            "mean_abs_proj": self.mean_abs_projection(),
            "temperature": float(temperature),
            "seed": int(seed),
        }

    def generate_batch(
        self,
        instructions: list,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> list:
        """Batched generation with a SELF-HEALING OOM fallback: if a batch
        runs out of GPU memory it empties the cache, splits in half, and retries
        — recursively, down to batch-1 — so an unattended run can never crash on
        OOM (worst case it degrades to slower, smaller batches). Delegates the
        real work to ``_generate_batch_impl``. On an OOM split under sampling
        the sub-batches re-seed with the same *seed* (different draws than the
        unsplit batch would have given — acceptable, because batch composition
        is already part of the sampling contract; see ``_generate_batch_impl``).
        """
        import torch
        if not instructions:
            return []
        try:
            return self._generate_batch_impl(instructions, max_new_tokens,
                                             temperature=temperature, seed=seed)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(instructions) <= 1:
                raise
            mid = len(instructions) // 2
            logger.warning(f"CUDA OOM at batch={len(instructions)} → auto-split to "
                           f"{mid}+{len(instructions) - mid} and retry (no crash)")
            return (self.generate_batch(instructions[:mid], max_new_tokens,
                                        temperature=temperature, seed=seed)
                    + self.generate_batch(instructions[mid:], max_new_tokens,
                                          temperature=temperature, seed=seed))

    def _generate_batch_impl(
        self,
        instructions: list,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> list:
        """BATCHED ``generate`` — throughput on an under-utilised GPU.

        Left-pads the prompts (decoder-only models must left-pad so generated
        tokens align across the batch), runs ONE ``model.generate`` over the batch
        under the steering hook (whose ``einsum("bsd,d->bs")`` already spans the
        batch dim), then decodes each sequence trimmed at its first EOS for the
        true token count. Padding positions are attention-masked, so steering them
        cannot affect the real sequences' tokens (verified by validate_batch.py:
        batched greedy == batch-1 greedy).

        SAMPLING (``temperature`` > 0): the batch is seeded ONCE (``_seed_torch``)
        before ``generate``, so the reproducibility contract is BATCH-level, not
        per-sequence — the same (batch composition, seed) reproduces the same
        draws, while a different ``batch_size`` yields different, equally valid
        draws from the same temperature distribution. The per-chain path keeps
        its per-(task, sample) seeding; downstream analysis pools samples per
        cell, so which valid draw arrived is immaterial. Callers group batches
        by sample index j and pass ``seed = sample_seed_base + j`` (added for
        E9.1, where per-chain T>0 generation was the wall-clock bottleneck).
        Returns one dict per input, in order, with ``generate``'s schema
        (``mean_abs_proj`` is a batch-level mean here — a diagnostic only, not
        used in the headline).
        """
        import torch
        from src.chain_gen import format_prompt, _seed_torch
        if not instructions:
            return []
        if max_new_tokens is None:
            max_new_tokens = default_max_new_tokens()
        tok = self.tokenizer
        prompts = [format_prompt(tok, ins) for ins in instructions]
        prev_side = tok.padding_side
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        inputs = tok(prompts, return_tensors="pt", padding=True).to(self.model.device)
        tok.padding_side = prev_side
        plen = inputs.input_ids.shape[1]

        if temperature > 0:           # batch-level seeding (see docstring)
            _seed_torch(seed)

        self._abs_proj_sum = 0.0      # reset per-generation energy probe
        self._abs_proj_count = 0
        self._disp_sum = 0.0
        self._disp_count = 0
        hook = self.model.model.layers[self.layer].register_forward_hook(self._hook_fn)
        try:
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=(temperature > 0),
                    temperature=temperature if temperature > 0 else 1.0,
                    pad_token_id=tok.eos_token_id,
                )
        finally:
            hook.remove()

        eos = tok.eos_token_id
        batch_proj = self.mean_abs_projection()
        results = []
        for i in range(out.shape[0]):
            gen_ids = out[i][plen:].tolist()
            n_tok = gen_ids.index(eos) if eos in gen_ids else len(gen_ids)
            chain = tok.decode(out[i][plen:plen + n_tok], skip_special_tokens=True)
            results.append({
                "instruction": instructions[i],
                "chain": chain,
                "n_tokens": int(n_tok),
                "alpha": self.alpha,
                "mode": self.mode,
                "layer": self.layer,
                "energy_scale": self.energy_scale,
                "mean_abs_proj": batch_proj,
                "temperature": float(temperature),
                "seed": int(seed),
            })
        return results


#: Manifold method label for a given k (k is 1/3/5/10 or the string "auto").
def manifold_method_label(k) -> str:
    return f"manifold_{'auto' if k == 'auto' else f'k{int(k)}'}"


def effective_task_id(base_task_id: str, replicate=None, sample: int = 0) -> str:
    """Compose a UNIQUE per-record task id from the base task + optional
    random-subspace replicate index + multi-sample index.

    Downstream (annotation dedup, ``aggregate_results``' ``ann_index``) all key on
    ``task_id``; records sharing ``(behaviour, method, alpha)`` are pooled into one
    cell and averaged. So to make the R random-subspace replicates and the N
    temperature samples each a DISTINCT scored draw inside the same cell (rather
    than colliding to one), we encode them into the task id and keep the true task
    in ``base_task_id`` for provenance. Suffix order is ``#rs{rep}`` then ``#s{j}``
    so a replicate's samples stay grouped under that replicate. ``sample==0`` with
    no replicate yields the bare ``base_task_id`` (back-compatible single-sample
    keys; pre-existing checkpoints resume unchanged)."""
    tid = str(base_task_id)
    if replicate is not None:
        tid += f"#rs{replicate}"
    if sample:
        tid += f"#s{sample}"
    return tid


def _build_arms(
    beh: str,
    vecs: dict,
    layer: int,
    include_random_control: bool = True,
    include_random_subspace: bool = True,
    include_energy_matched: bool = True,
    include_orthogonal_complement: bool = True,
    n_random_subspaces: int = 3,
) -> tuple[list[dict], dict]:
    """Construct every steering arm for one behaviour (pure — no model).

    Returns ``(arms, geometry)`` where each *arm* is a dict
    ``{method, vector, energy_scale, subspace_replicate}`` (``energy_scale`` is a
    placeholder 1.0 here — the energy-matched arm is calibrated against the model
    later) and *geometry* is the per-(behaviour, k) bound metadata
    ``{k: {"cos_single_manifold", "retained_energy"}}``.

    Arms (labels are the ``method`` field consumed by the analysis):
      - ``single_direction``                  — Venhoff diff-of-means (unit norm).
      - ``manifold_k{1,3,5,10}`` / ``manifold_auto`` — HEADLINE k-sweep: the diff-
        of-means projected onto the behaviour's top-k PCA subspace, renormalised.
        Previously only auto_k ran, where manifold≈single (cos 0.95–0.97).
      - ``random_subspace_k{k}`` (× ``n_random_subspaces`` seeded replicates) —
        single direction projected onto a RANDOM k-dim subspace, renormalised
        identically. Isolates "the behaviour's PCA subspace matters" from "any
        k-dim projection + renorm". Replicates averaged downstream.
      - ``random_direction``                   — norm-matched random (SANITY FLOOR,
        not the causal baseline: injects ~19× less energy).
      - ``energy_matched_random``              — random direction rescaled so its
        mean |rᵀh| matches the behaviour arm's (the real energy-control floor).
      - ``orthogonal_complement``              — the OFF-subspace component
        (I−P_k)r alone, to test whether the discarded part is pure collateral.

    ``include_*`` flags gate the optional control families so a generation-first
    smoke run can stay cheap.
    """
    r_single = np.asarray(vecs["single_direction"])
    manifold = vecs["manifold_projected"]           # {k: unit vector}
    # Canonical k for the orthogonal-complement / energy-match reference subspace.
    ref_k = "auto" if "auto" in manifold else sorted(
        (k for k in manifold if k != "auto"), key=lambda x: int(x))[-1]

    arms: list[dict] = [
        {"method": "single_direction", "vector": r_single,
         "energy_scale": 1.0, "subspace_replicate": None},
    ]

    geometry: dict = {}
    # k-sweep manifold arms + their geometry bounds.
    for k, vec in manifold.items():
        vec = np.asarray(vec)
        arms.append({"method": manifold_method_label(k), "vector": vec,
                     "energy_scale": 1.0, "subspace_replicate": None})
        kk = "auto" if k == "auto" else int(k)
        geometry[kk] = {
            "cos_single_manifold": cosine(r_single, vec),
            "retained_energy": retained_energy(r_single, vec),
        }

    # Random-subspace-of-equal-k control: one method label per k, R replicates.
    if include_random_subspace:
        for k in manifold:
            kk = "auto" if k == "auto" else int(k)
            # auto's k is the behaviour's auto_k (number of PCs), recorded in vecs.
            k_dim = int(vecs.get("auto_k", 1)) if k == "auto" else int(k)
            for rep in range(max(1, n_random_subspaces)):
                v = random_subspace_projection(
                    r_single, k_dim,
                    seed_key=f"random_subspace|{beh}|L{layer}|k{kk}|rep{rep}")
                arms.append({"method": f"random_subspace_k{kk}", "vector": v,
                             "energy_scale": 1.0, "subspace_replicate": rep})

    # Norm-matched random (sanity floor).
    if include_random_control:
        arms.append({
            "method": "random_direction",
            "vector": random_direction_like(
                r_single, f"random_direction|{beh}|L{layer}"),
            "energy_scale": 1.0, "subspace_replicate": None})

    # Energy-matched random (scale calibrated against the model later; the
    # VECTOR is the same isotropic random direction, distinct seed from the floor).
    if include_energy_matched:
        arms.append({
            "method": "energy_matched_random",
            "vector": random_direction_like(
                r_single, f"energy_matched_random|{beh}|L{layer}"),
            "energy_scale": 1.0, "subspace_replicate": None})

    # Orthogonal-complement (off-subspace component of the single direction).
    if include_orthogonal_complement:
        arms.append({
            "method": "orthogonal_complement",
            "vector": orthogonal_complement_vector(r_single, np.asarray(manifold[ref_k])),
            "energy_scale": 1.0, "subspace_replicate": None,
            "complement_of_k": ("auto" if ref_k == "auto" else int(ref_k))})

    return arms, geometry


def run_steering_experiment(
    model,
    tokenizer,
    tasks: list[dict],
    steering_vectors: dict,
    alpha_values: list[float],
    max_new_tokens: Optional[int] = None,
    save_path: Optional[Path] = None,
    include_random_control: bool = True,
    include_random_subspace: bool = True,
    include_energy_matched: bool = True,
    include_orthogonal_complement: bool = True,
    n_random_subspaces: int = 3,
    geometry_path: Optional[Path] = None,
    n_samples: int = 1,
    temperature: float = 0.0,
    sample_seed_base: int = 0,
    batch_size: int = 1,
    steer_mode: str = "subtract",
) -> list[dict]:
    """
    Run the main comparison experiment.

    ``steer_mode``: "subtract" (suppression — every executed run before E9.1b)
    or "add" (amplification, E9.1b: h' = h + α(rᵀh)r, over-expressing the
    component). All arms including the floors run in the same mode so the
    matched-perturbation logic is preserved; the shared vanilla baseline is a
    zero-vector identity in either mode. Run different modes into DIFFERENT
    out-dirs — the resume key space does not encode the mode.

    Arms (see ``_build_arms`` and the module docstring):
      - ``vanilla``: ONE unsteered generation per task, recorded once under
        behaviour=SHARED_BASELINE / alpha=0.0 (greedy ⇒ per-α copies identical).
      - ``single_direction``.
      - HEADLINE k-sweep ``manifold_k{1,3,5,10}`` / ``manifold_auto`` (every built
        k, not just auto — at auto manifold≈single, cos 0.95–0.97).
      - ``random_subspace_k{k}`` (R seeded replicates) — equal-k random-subspace
        control isolating "the behaviour's subspace matters" from "any k-projection".
      - ``random_direction`` (norm-matched SANITY FLOOR), ``energy_matched_random``
        (rescaled to match the behaviour arm's mean |rᵀh| — the real energy floor),
        ``orthogonal_complement`` ((I−P_k)r alone).

    Writes per-(behaviour, k) ``cos(single, manifold_k)`` and retained-energy
    ``‖P_k r‖/‖r‖`` to *geometry_path* (defaults next to *save_path*) — these
    BOUND how large any manifold effect can be.

    Multi-sample (``n_samples`` > 1): for each (task, arm, α) — INCLUDING the
    shared vanilla baseline — draw ``n_samples`` chains at ``temperature`` > 0,
    each with a distinct seed (``sample_seed_base`` + j). Samples are keyed
    distinctly by folding the sample index into ``task_id`` (``#s{j}``; see
    ``effective_task_id``) with the true task in ``base_task_id``, so resume,
    annotation dedup, and ``aggregate_results`` treat the N samples as N
    independent draws pooled into one (behaviour, method, α) cell — NO downstream
    change. ``n_samples`` > 1 with ``temperature`` ≤ 0 is rejected (greedy draws
    are byte-identical, so it would fake N points from one). The vanilla baseline
    is sampled the SAME N times so the on-target Δ-vs-vanilla compares like with
    like (N steered vs N vanilla, not N vs 1). Default ``n_samples=1`` /
    ``temperature=0.0`` reproduces the prior greedy single-sample behaviour and
    key scheme exactly.

    Checkpoints to *save_path* after every (behaviour, α, method) sweep so the
    experiment can be resumed after interruption.

    Returns:
        List of result dicts: {behaviour, method, alpha, task_id, base_task_id,
                               chain, n_tokens, layer, energy_scale,
                               subspace_replicate, sample, temperature, seed,
                               mean_abs_proj}
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

    # ── Multi-sample guard ───────────────────────────────────────────────────
    # N samples only makes sense under sampling. Greedy (T=0) is deterministic,
    # so N greedy draws are byte-identical and would manufacture N data points
    # from one — a silent variance fraud. Fail loud rather than mislead.
    if steer_mode not in ("subtract", "add"):
        raise ValueError(f"steer_mode must be 'subtract' or 'add', got {steer_mode!r}")
    if steer_mode == "add":
        logger.info("AMPLIFY mode (E9.1b): h' = h + α(rᵀh)r on every arm incl. "
                    "floors; use a dedicated out-dir (resume keys don't encode mode)")
    n_samples = int(n_samples)
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if n_samples > 1 and temperature <= 0:
        raise ValueError(
            f"n_samples={n_samples} requires temperature>0 (greedy decoding is "
            f"deterministic, so multiple samples would be byte-identical and "
            f"would fake {n_samples} independent draws from one). Pass e.g. "
            f"temperature=0.7.")
    if n_samples == 1 and temperature > 0:
        logger.info(f"n_samples=1 with temperature={temperature}: one sampled "
                    f"draw per cell (set n_samples>1 for a sampling distribution)")
    if n_samples > 1:
        logger.info(f"Multi-sample: {n_samples} draws per (task, arm, α) at "
                    f"temperature={temperature} (seeds {sample_seed_base}.."
                    f"{sample_seed_base + n_samples - 1}); samples keyed #s{{j}} "
                    f"into task_id, pooled per cell downstream")

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

    # Dedup key includes subspace_replicate (None for single-vector arms) so the
    # R random-subspace replicates at the same (beh, method, α, task) don't
    # collide. Legacy 4-tuple records (no replicate field) read as rep=None.
    done = {
        (r["behaviour"], r["method"], r["alpha"], r["task_id"],
         r.get("subspace_replicate"))
        for r in results
    }

    # Prompt token lengths for LENGTH-SORTED batching: grouping similar-length
    # prompts into a batch minimises left-padding, which minimises the
    # floating-point perturbation that flips near-tie greedy tokens (batched vs
    # batch-1). Computed once; keyed by base task id.
    from src.chain_gen import format_prompt as _format_prompt
    _plen = {t["id"]: len(tokenizer(_format_prompt(tokenizer, t["prompt"])).input_ids)
             for t in tasks}
    def _by_len(item):   # item = (task, j, eff_task_id)
        return _plen.get(item[0]["id"], 0)

    # ── Shared unsteered baseline: N generations per task ─────────────────
    # Sampled the SAME N times (distinct seeds) as the steered arms so the
    # Δ-vs-vanilla comparison is N-vs-N, not N-vs-1. With n_samples=1 the
    # sample suffix is empty → the legacy bare-task_id key is preserved, so
    # pre-multisample checkpoints resume without re-generating vanilla.
    van_pending = []
    for task in tasks:
        for j in range(n_samples):
            eff_task_id = effective_task_id(task["id"], replicate=None, sample=j)
            if (SHARED_BASELINE, "vanilla", 0.0, eff_task_id, None) in done:
                continue
            van_pending.append((task, j, eff_task_id))

    def _van_record(task, j, eff_task_id, r):
        results.append({
            "behaviour": SHARED_BASELINE, "method": "vanilla", "alpha": 0.0,
            "task_id": eff_task_id, "base_task_id": task["id"],
            "chain": r["chain"], "n_tokens": r["n_tokens"], "layer": None,
            "sample": j, "temperature": float(temperature),
            "seed": int(sample_seed_base + j),
        })
        done.add((SHARED_BASELINE, "vanilla", 0.0, eff_task_id, None))

    if batch_size > 1:
        # Unsteered baseline via a zero-vector α=0 SteeredModel — an EXACT identity
        # (delta = 0·proj·0 = 0; validated == generate_chain) — so we reuse the
        # validated batched path. Vanilla chains run to completion (the
        # LONGEST chains), so this also stress-tests batch_size on the worst-case
        # memory before committing the 1600-chain arm loop to it. Under sampling
        # the batches are grouped by sample index j and seeded per group
        # (batch-level reproducibility — see _generate_batch_impl).
        import numpy as _np
        van_model = SteeredModel(
            model, tokenizer, _np.zeros(model.config.hidden_size, dtype=_np.float32),
            0, alpha=0.0, mode="subtract", energy_scale=0.0)
        for jg in sorted({j for _, j, _ in van_pending}):
            grp = [p for p in van_pending if p[1] == jg]
            grp.sort(key=_by_len)            # length-sorted batches → minimal padding
            for b0 in tqdm(range(0, len(grp), batch_size),
                           desc=f"vanilla (shared baseline) s{jg}", leave=False):
                chunk = grp[b0:b0 + batch_size]
                rs = van_model.generate_batch(
                    [t["prompt"] for t, j, e in chunk], max_new_tokens,
                    temperature=temperature, seed=sample_seed_base + jg)
                for (task, j, eff_task_id), r in zip(chunk, rs):
                    _van_record(task, j, eff_task_id, r)
                if save_path:
                    _save_json(results, save_path)   # save after every batch (atomic)
    else:
        for (task, j, eff_task_id) in tqdm(van_pending, desc="vanilla (shared baseline)",
                                           leave=False):
            r = generate_chain(model, tokenizer, task["prompt"], max_new_tokens,
                               temperature=temperature, seed=sample_seed_base + j)
            _van_record(task, j, eff_task_id, r)
            if save_path:        # checkpoint after EVERY vanilla chain so a crash
                _save_json(results, save_path)   # or restart loses 0 generated output

    # ── Steered arms ───────────────────────────────────────────────────────
    steered_alphas = [a for a in alpha_values if a > 0]
    if len(steered_alphas) < len(alpha_values):
        logger.info("α=0 entries are covered by the shared vanilla baseline "
                    "(subtract-mode steering at α=0 is the identity)")

    geometry_meta: dict = {}
    for beh, vecs in steering_vectors.items():
        layer = vecs["layer"]

        arms, geometry = _build_arms(
            beh, vecs, layer,
            include_random_control=include_random_control,
            include_random_subspace=include_random_subspace,
            include_energy_matched=include_energy_matched,
            include_orthogonal_complement=include_orthogonal_complement,
            n_random_subspaces=n_random_subspaces,
        )
        geometry_meta[beh] = geometry

        # ── Calibrate the energy-matched-random arm ──────────────────────────
        # Match its delivered energy to the SINGLE direction (the behaviour
        # baseline). Both means are read on the SAME unperturbed stream of a few
        # calibration tasks (mode="measure"), so the ratio is well-defined.
        # Skip the probe entirely on a fully-resumed run (every energy-matched
        # cell already generated) so resume stays cheap.
        em_arms = [a for a in arms if a["method"] == "energy_matched_random"]
        em_pending = any(
            (beh, "energy_matched_random", alpha,
             effective_task_id(task["id"], replicate=None, sample=j), None) not in done
            for alpha in steered_alphas for task in tasks
            for j in range(n_samples)
        ) if em_arms else False
        if em_arms and em_pending:
            calib_tasks = tasks[: min(5, len(tasks))]
            e_beh = measure_mean_abs_proj(
                model, tokenizer, vecs["single_direction"], layer, calib_tasks)
            for a in em_arms:
                e_rand = measure_mean_abs_proj(
                    model, tokenizer, a["vector"], layer, calib_tasks)
                a["energy_scale"] = energy_matched_scale(e_beh, e_rand)
            geometry_meta[beh]["energy_matched_random_scale"] = em_arms[0]["energy_scale"]
            logger.info(f"  {beh}: energy_matched_random scale="
                        f"{em_arms[0]['energy_scale']:.2f} "
                        f"(E_single={e_beh:.3f})")

        if geometry_path or save_path:
            gpath = Path(geometry_path) if geometry_path else \
                Path(save_path).with_name("steering_geometry.json")
            _save_json(geometry_meta, gpath)

        for alpha in steered_alphas:
            for arm in arms:
                method_name = arm["method"]
                vec = arm["vector"]
                rep = arm["subspace_replicate"]
                escale = arm["energy_scale"]
                rep_tag = "" if rep is None else f" r{rep}"
                steered = SteeredModel(model, tokenizer, vec, layer,
                                       alpha=alpha, mode=steer_mode,
                                       energy_scale=escale)

                # Random-subspace REPLICATES (rep) and temperature SAMPLES (j) both
                # share (beh, method, α) but must be distinct points for annotation
                # + aggregation. Fold BOTH into task_id (downstream keys on it), so
                # evaluation.py treats them as independent draws pooled into one
                # cell — NO evaluation.py change. base_task_id keeps the true task;
                # the dedup key carries the FULL eff_task_id so each resumes apart.
                pending = []
                for task in tasks:
                    for j in range(n_samples):
                        eff_task_id = effective_task_id(task["id"], replicate=rep, sample=j)
                        if (beh, method_name, alpha, eff_task_id, rep) in done:
                            continue
                        pending.append((task, j, eff_task_id))

                def _record(task, j, eff_task_id, r):
                    results.append({
                        "behaviour": beh, "method": method_name, "alpha": alpha,
                        "task_id": eff_task_id, "base_task_id": task["id"],
                        "chain": r["chain"], "n_tokens": r["n_tokens"],
                        "layer": layer, "energy_scale": escale,
                        "subspace_replicate": rep, "sample": j,
                        "temperature": float(temperature),
                        "seed": int(sample_seed_base + j),
                        "mode": steer_mode,
                        "mean_abs_proj": r.get("mean_abs_proj"),
                    })
                    done.add((beh, method_name, alpha, eff_task_id, rep))

                desc = f"{beh[:4]} α={alpha:.1f} {method_name}{rep_tag}"
                if batch_size > 1:
                    # Batched generation fills the GPU; checkpoint per batch
                    # (finer crash-safety than the old per-sweep save). Under
                    # sampling, batches are grouped by sample index j and seeded
                    # per group (batch-level reproducibility — see
                    # _generate_batch_impl); under greedy this is one group (j=0)
                    # and reduces exactly to the validated greedy path.
                    for jg in sorted({j for _, j, _ in pending}):
                        grp = [p for p in pending if p[1] == jg]
                        grp.sort(key=_by_len)   # length-sorted batches → minimal padding
                        for b0 in tqdm(range(0, len(grp), batch_size),
                                       desc=f"{desc} s{jg}", leave=False):
                            chunk = grp[b0:b0 + batch_size]
                            rs = steered.generate_batch(
                                [t["prompt"] for t, j, e in chunk], max_new_tokens,
                                temperature=temperature, seed=sample_seed_base + jg)
                            for (task, j, eff_task_id), r in zip(chunk, rs):
                                _record(task, j, eff_task_id, r)
                            if save_path:
                                _save_json(results, save_path)
                else:
                    for (task, j, eff_task_id) in tqdm(pending, desc=desc, leave=False):
                        r = steered.generate(task["prompt"], max_new_tokens,
                                             temperature=temperature,
                                             seed=sample_seed_base + j)
                        _record(task, j, eff_task_id, r)
                        if save_path:        # checkpoint after EVERY chain
                            _save_json(results, save_path)

    logger.info(f"Steering experiment complete: {len(results)} total results "
                f"({len(geometry_meta)} behaviours, geometry bounds written)")
    return results


def _save_json(data, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.rename(path)
