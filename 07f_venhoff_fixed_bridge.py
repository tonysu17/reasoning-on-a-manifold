#!/usr/bin/env python3
"""Generation-only Venhoff steering bridge for R1-Distill-Qwen-1.5B.

This runner deliberately isolates the intervention used by Venhoff et al.:

    h <- h - alpha * v

at their published per-behaviour layers, with alpha=1 by default.  It does not
run the repository's projective intervention ``h <- h-alpha*(v.T@h)*v``, and it
does not construct manifold or random-control arms.  It also never calls an
annotation API: the output is a resumable generation artefact plus the cheap
annotation-free damage metrics.

Three vector-source contracts are supported and kept explicit in every record:

``user_direction_published_norm`` (default)
    Keep our direction but rescale it to the corresponding released Venhoff
    vector norm.  This is a useful magnitude sensitivity, not an exact released
    vector.

``user_direction_own_row_pooled_proxy``
(``--scale-convention own-row-pooled-proxy``)
    Rescale to a row-pooled overall-activation norm reconstructed from our
    cached activations.  This is a mixed-vintage proxy (clip-window conventions
    differ across cached inputs, and rows rather than author responses are
    weighted equally), not an exact own-data analogue of Venhoff's scale.

``released_vector`` (pass ``--released-mean-vectors``)
    Derive Venhoff's exact feature vector from a user-supplied copy of the
    released ``mean_vectors_deepseek-r1-distill-qwen-1.5b.pt`` file:
    ``mean_behaviour - mean_overall``, rescaled to ``||mean_overall||``.  The
    file is not bundled and no temporary checkout path is assumed.

By default the bridge runs all four published Qwen-1.5B behaviours at their
published layers and write norms.  Its hybrid user-vector asset preserves the
current thesis E1 direction bytes for backtracking and example-testing; the two
L18 directions are separately reconstructed mixed-vintage sensitivities.  The
evaluation set remains this repository's 50-task stratified hold-out, so even
the released-vector arm is not a full paper replication (the tasks/prompts and
later annotator remain different).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.chain_gen import format_prompt, generate_chain, load_model
from src.config import provenance
from src.evaluation import aggregate_results, save_summary
from src.steered_inference import SteeredModel


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_SHORT = "R1-1.5B"
MODEL_REVISION = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
MODEL_WEIGHT_SHA256 = (
    "58858233513d76b8703e72eed6ce16807b523328188e13329257fb9594462945"
)

# Published Qwen-1.5B steering layers (Venhoff released steering_config).
VENHOFF_LAYERS = {
    "backtracking": 17,
    "uncertainty-estimation": 18,
    "example-testing": 15,
    "adding-knowledge": 18,
}
DEFAULT_BEHAVIOURS = tuple(VENHOFF_LAYERS)
DEFAULT_VECTORS_DIR = Path(
    "results/steering_vectors/R1-1.5B__venhoff_bridge_hybrid"
)

# Norms obtained by applying the released code's exact normalization to the
# official mean-vector file at the layers above.  The feature-vector norm is the
# corresponding ||mean_overall[layer]||.  These constants make the default
# operator bridge match the published write magnitude without pretending our
# unit directions are the authors' vectors.
RELEASED_VECTOR_NORMS = {
    "backtracking": 70.868286132812,
    "uncertainty-estimation": 79.720710754395,
    "example-testing": 56.323959350586,
    "adding-knowledge": 79.720710754395,
}

# Row-pooled proxy from cached, hold-out-excluded six-label activations.  This
# is explicitly NOT an exact own-data Venhoff convention: the cached labels
# mix clip-window vintages and weight activation rows rather than responses.
OWN_ROW_POOLED_PROXY_NORMS = {
    "backtracking": 75.5932,
    "example-testing": 55.9128,
}

RELEASED_MEAN_VECTORS_SHA256 = (
    "bbf7bcac3758df3236d485c41dcf3033a53fd6333a4abd555290e8ef71419412"
)
RELEASED_REPOSITORY = "https://github.com/cvenhoff/steering-thinking-llms"
RELEASED_REPOSITORY_COMMIT = "93259bc3410c99293351df41141cd16b4110422a"
RELEASED_VECTOR_FILE_COMMIT = "8c8f0a2d2ee41b1173877d488e0ae8565b4de2cc"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(data: object) -> str:
    payload = json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _default_pinned_snapshot() -> Path:
    """Return the immutable local Hugging Face snapshot for the audited revision."""
    hf_home = Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    )
    return (
        hf_home
        / "hub"
        / "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B"
        / "snapshots"
        / MODEL_REVISION
    )


def _verify_model_snapshot(path: Path) -> dict:
    """Fail closed unless *path* is the pinned, byte-verified model snapshot."""
    path = path.expanduser().resolve()
    required = [
        path / "config.json",
        path / "tokenizer.json",
        path / "tokenizer_config.json",
        path / "model.safetensors",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Pinned model snapshot is incomplete. Missing: " + ", ".join(missing)
        )
    weight_sha = _sha256(path / "model.safetensors")
    if weight_sha != MODEL_WEIGHT_SHA256:
        raise ValueError(
            "Pinned model weight SHA-256 mismatch: "
            f"expected {MODEL_WEIGHT_SHA256}, got {weight_sha}"
        )
    return {
        "public_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "local_snapshot": str(path),
        "model_weight_sha256": weight_sha,
    }


def _atomic_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def _normalise_and_scale(direction: np.ndarray, norm: float, behaviour: str) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float32).reshape(-1)
    direction_norm = float(np.linalg.norm(direction))
    if not np.isfinite(direction_norm) or direction_norm < 1e-10:
        raise ValueError(
            f"{behaviour}: user direction has invalid norm {direction_norm!r}"
        )
    return (direction / direction_norm * float(norm)).astype(np.float32)


def _load_user_single_vectors(
    vectors_dir: Path,
) -> tuple[dict[str, dict], dict, Path]:
    """Load only the single vectors needed by this no-manifold runner.

    ``src.steering.load_steering_vectors`` also imports the PCA build stack and
    reads every manifold file.  This narrow loader intentionally avoids both:
    the bridge has one fixed-vector arm and should not require scikit-learn at
    generation time.
    """
    metadata_path = vectors_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Vector metadata not found: {metadata_path}")
    with metadata_path.open() as f:
        metadata = json.load(f)

    vectors: dict[str, dict] = {}
    for behaviour, meta in metadata.items():
        if behaviour.startswith("_"):
            continue
        single_path = vectors_dir / f"{behaviour}_single.npy"
        if not single_path.is_file():
            raise FileNotFoundError(f"Single-direction vector not found: {single_path}")
        vectors[behaviour] = {
            "layer": int(meta["layer"]),
            "single_direction": np.load(single_path),
            "path": single_path.resolve(),
            "sha256": _sha256(single_path),
        }
    return vectors, metadata, metadata_path


def load_user_directions(
    vectors_dir: Path,
    behaviours: list[str],
    *,
    scale_convention: str,
) -> tuple[dict[str, dict], dict]:
    """Load the audited bridge directions and apply an explicit norm convention."""
    vectors, metadata, metadata_path = _load_user_single_vectors(vectors_dir)
    if scale_convention == "own-row-pooled-proxy":
        norm_map = OWN_ROW_POOLED_PROXY_NORMS
        vector_kind = "user_direction_own_row_pooled_proxy"
        scale_description = (
            "mixed-vintage row-pooled proxy from cached holdout-excluded "
            "six-label activations; rows, not author responses, are equally weighted"
        )
    elif scale_convention == "published-norm":
        norm_map = RELEASED_VECTOR_NORMS
        vector_kind = "user_direction_published_norm"
        scale_description = (
            "released Venhoff Qwen-1.5B vector norm applied to the user direction"
        )
    else:  # guarded by argparse; retained for direct function callers
        raise ValueError(f"Unknown scale convention: {scale_convention!r}")

    selected: dict[str, dict] = {}
    for behaviour in behaviours:
        if behaviour not in vectors:
            raise ValueError(
                f"{behaviour!r} is absent from {vectors_dir}; available: "
                f"{sorted(vectors)}"
            )
        actual_layer = int(vectors[behaviour]["layer"])
        expected_layer = VENHOFF_LAYERS[behaviour]
        if actual_layer != expected_layer:
            raise ValueError(
                f"{behaviour}: vector metadata says layer {actual_layer}, but "
                f"Venhoff's Qwen-1.5B layer is {expected_layer}"
            )
        if behaviour not in norm_map:
            raise ValueError(
                f"No {scale_convention} norm has been established for "
                f"{behaviour!r}. The generation bridge currently supports "
                f"{sorted(norm_map)} under that convention."
            )
        target_norm = norm_map[behaviour]
        selected[behaviour] = {
            "layer": expected_layer,
            "vector": _normalise_and_scale(
                vectors[behaviour]["single_direction"], target_norm, behaviour
            ),
            "target_norm": target_norm,
            "source_path": str(vectors[behaviour]["path"]),
            "source_sha256": vectors[behaviour]["sha256"],
        }
        selected[behaviour]["derived_vector_sha256"] = hashlib.sha256(
            selected[behaviour]["vector"].tobytes()
        ).hexdigest()

    build_provenance = metadata.get("_provenance", {})
    contract = {
        "kind": vector_kind,
        "description": (
            "repository hybrid exact-layer unit directions: the current E1 "
            "directions are retained exactly for backtracking/example-testing, "
            "while uncertainty-estimation/adding-knowledge use the separately "
            "reconstructed mixed-vintage L18 directions"
        ),
        "vectors_dir": str(vectors_dir.resolve()),
        "metadata": {
            "path": str(metadata_path.resolve()),
            "sha256": _sha256(metadata_path),
            "build_provenance": build_provenance,
            "provenance_status": build_provenance.get("provenance_status")
            or (
                "unresolved vector-build provenance"
                if build_provenance.get("git_commit") is None
                else "recorded"
            ),
        },
        "unit_vectors": {
            b: {
                "path": selected[b]["source_path"],
                "sha256": selected[b]["source_sha256"],
                "derived_scaled_vector_sha256": selected[b][
                    "derived_vector_sha256"
                ],
            }
            for b in behaviours
        },
        "scale": {
            "convention": scale_convention,
            "description": scale_description,
            "values": {b: norm_map[b] for b in behaviours},
        },
    }
    if scale_convention == "own-row-pooled-proxy":
        contract["scale"]["source"] = (
            "cached holdout-excluded six-label activations; mixed clip-window "
            "vintages and row weighting make this a sensitivity proxy only"
        )
    else:
        contract["scale"]["source"] = {
            "repository": RELEASED_REPOSITORY,
            "repository_commit_audited": RELEASED_REPOSITORY_COMMIT,
            "vector_file_commit": RELEASED_VECTOR_FILE_COMMIT,
            "mean_vector_file_sha256": RELEASED_MEAN_VECTORS_SHA256,
        }
    return selected, contract


def load_released_vectors(
    mean_vectors_path: Path,
    behaviours: list[str],
    *,
    allow_unverified: bool = False,
) -> tuple[dict[str, dict], dict]:
    """Derive the exact released vectors from a supplied official .pt file."""
    import torch

    if not mean_vectors_path.is_file():
        raise FileNotFoundError(f"Released mean-vector file not found: {mean_vectors_path}")
    file_sha = _sha256(mean_vectors_path)
    if file_sha != RELEASED_MEAN_VECTORS_SHA256 and not allow_unverified:
        raise ValueError(
            "The supplied --released-mean-vectors file does not match the "
            "audited official Qwen-1.5B file. "
            f"expected sha256={RELEASED_MEAN_VECTORS_SHA256}, got {file_sha}. "
            "Use --allow-unverified-released-vectors only if this is deliberate."
        )

    try:
        means = torch.load(mean_vectors_path, map_location="cpu", weights_only=True)
    except TypeError:  # Older torch without weights_only; the SHA check limits risk.
        means = torch.load(mean_vectors_path, map_location="cpu")

    if "overall" not in means or "mean" not in means["overall"]:
        raise ValueError("Released vector file lacks overall['mean']")
    overall = means["overall"]["mean"].float()
    selected: dict[str, dict] = {}
    for behaviour in behaviours:
        if behaviour not in means or "mean" not in means[behaviour]:
            raise ValueError(f"Released vector file lacks {behaviour!r}['mean']")
        layer = VENHOFF_LAYERS[behaviour]
        behaviour_mean = means[behaviour]["mean"].float()
        if overall.ndim != 2 or behaviour_mean.shape != overall.shape:
            raise ValueError(
                f"Unexpected released mean shape: overall={tuple(overall.shape)}, "
                f"{behaviour}={tuple(behaviour_mean.shape)}"
            )
        if layer >= overall.shape[0]:
            raise ValueError(
                f"{behaviour}: layer {layer} absent from mean tensor "
                f"with {overall.shape[0]} layers"
            )
        raw = behaviour_mean[layer] - overall[layer]
        raw_norm = float(raw.norm().item())
        overall_norm = float(overall[layer].norm().item())
        if not np.isfinite(raw_norm) or raw_norm < 1e-10:
            raise ValueError(f"{behaviour}: released mean-difference norm is invalid")
        vector = raw * (overall_norm / raw_norm)
        selected[behaviour] = {
            "layer": layer,
            "vector": vector.cpu().numpy().astype(np.float32),
            "target_norm": overall_norm,
        }
        selected[behaviour]["derived_vector_sha256"] = hashlib.sha256(
            selected[behaviour]["vector"].tobytes()
        ).hexdigest()

    contract = {
        "kind": "released_vector",
        "description": (
            "exact behaviour-minus-overall vectors derived from the supplied "
            "released mean-vector file and rescaled to ||mean_overall||"
        ),
        "path": str(mean_vectors_path.resolve()),
        "sha256": file_sha,
        "official_sha256_match": file_sha == RELEASED_MEAN_VECTORS_SHA256,
        "repository": RELEASED_REPOSITORY,
        "repository_commit_audited": RELEASED_REPOSITORY_COMMIT,
        "vector_file_commit": RELEASED_VECTOR_FILE_COMMIT,
        "derived_vectors": {
            b: {
                "layer": selected[b]["layer"],
                "sha256": selected[b]["derived_vector_sha256"],
                "norm": selected[b]["target_norm"],
            }
            for b in behaviours
        },
    }
    return selected, contract


def _load_existing_results(
    path: Path, vector_kind: str, run_contract_sha256: str
) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        results = json.load(f)
    kinds = {
        r.get("vector_source")
        for r in results
        if r.get("method") != "vanilla" and r.get("vector_source") is not None
    }
    if kinds and kinds != {vector_kind}:
        raise ValueError(
            f"Refusing to mix vector sources in {path}: existing={sorted(kinds)}, "
            f"requested={vector_kind}. Choose a fresh --out-dir."
        )
    contract_hashes = {
        r.get("run_contract_sha256") for r in results
        if r.get("run_contract_sha256") is not None
    }
    if contract_hashes and contract_hashes != {run_contract_sha256}:
        raise ValueError(
            f"Refusing to mix run contracts in {path}: "
            f"existing={sorted(contract_hashes)}, requested={run_contract_sha256}. "
            "Choose a fresh --out-dir."
        )
    logger.info("Resuming: %d generated records already saved", len(results))
    return results


def _task_record(
    task: dict,
    result: dict,
    *,
    max_new_tokens: int,
    run_contract_sha256: str,
    **fields,
) -> dict:
    chain = result["chain"]
    n_tokens = int(result["n_tokens"])
    return {
        "task_id": task["id"],
        "base_task_id": task["id"],
        "category": task.get("category", "unknown"),
        "instruction": task["prompt"],
        "prompt": result["prompt"],
        "chain": chain,
        "full_text": result["full_text"],
        "n_tokens": n_tokens,
        "closed_think": "</think>" in chain,
        "hit_token_cap": n_tokens >= int(max_new_tokens),
        "temperature": 0.0,
        "seed": 42,
        "run_contract_sha256": run_contract_sha256,
        **fields,
    }


_REUSED_BASELINE_CONTENT_FIELDS = (
    "task_id",
    "base_task_id",
    "category",
    "instruction",
    "prompt",
    "chain",
    "full_text",
    "n_tokens",
    "closed_think",
    "hit_token_cap",
    "temperature",
    "seed",
)


def _read_json_artifact(path: Path) -> tuple[object, str, int]:
    """Read and hash one immutable snapshot of a JSON artefact.

    The source runner checkpoints by atomically replacing its results file.  A
    single byte read therefore pins either the old or new complete file even if
    that runner is still active; hashing in a second pass would introduce an
    avoidable time-of-check/time-of-use race.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Required baseline-reuse artefact not found: {path}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        return json.loads(raw), digest, len(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON baseline-reuse artefact: {path}") from exc


def _canonical_formatted_prompts(
    model_contract: dict, tasks: list[dict]
) -> tuple[dict[str, str], dict]:
    """Format tasks with the pinned tokenizer and bind the exact template inputs."""
    from transformers import AutoTokenizer

    snapshot = Path(model_contract["local_snapshot"])
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    expected = {
        task["id"]: format_prompt(
            tokenizer,
            task["prompt"],
            model_id=MODEL_ID,
        )
        for task in tasks
    }
    chat_template = tokenizer.chat_template
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("Pinned tokenizer has no usable chat template")
    tokenizer_files = {}
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        path = snapshot / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Pinned tokenizer artefact required for baseline reuse is missing: {path}"
            )
        tokenizer_files[filename] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
        }
    adapter_path = Path(__file__).resolve().parent / "src/model_adapters.py"
    return expected, {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_files": tokenizer_files,
        "runtime_chat_template_sha256": hashlib.sha256(
            chat_template.encode("utf-8")
        ).hexdigest(),
        "formatter_source": {
            "path": str(adapter_path),
            "sha256": _sha256(adapter_path),
        },
        "formatted_prompts_sha256": _json_sha256(expected),
    }


def _prepare_baseline_reuse(
    source_dir: Path,
    *,
    model_contract: dict,
    dtype: str,
    use_4bit: bool,
    max_new_tokens: int,
    temperature: float,
    evaluation_task_ids: list[str],
    evaluation_ids_source_sha256: str,
    tasks_source_sha256: str,
    tasks_by_id: dict[str, dict],
    expected_formatted_prompts: dict[str, str],
    prompt_format_contract: dict,
) -> tuple[list[dict], dict]:
    """Validate and snapshot exactly 50 reusable shared vanilla records.

    Non-baseline rows may coexist in the source checkpoint (for example, when
    a steering arm had already begun).  They are counted and cryptographically
    covered by the source artefact hash, but are never copied.  Returned rows
    contain only a small, explicit baseline schema and have no destination run
    contract yet; that hash is installed only after the reuse manifest itself
    has been bound into the destination contract.
    """
    source_dir = source_dir.expanduser().resolve()
    if dtype != "bfloat16" or bool(use_4bit):
        raise ValueError(
            "Baseline reuse is restricted to the audited bfloat16, "
            "non-quantized protocol"
        )
    if int(max_new_tokens) != 1000 or float(temperature) != 0.0:
        raise ValueError(
            "Baseline reuse is restricted to max_new_tokens=1000 and greedy "
            "temperature=0 generation"
        )
    if len(evaluation_task_ids) != 50 or len(set(evaluation_task_ids)) != 50:
        raise ValueError(
            "Baseline reuse requires the complete ordered 50-task E1 hold-out"
        )
    if list(expected_formatted_prompts) != evaluation_task_ids:
        raise ValueError(
            "Canonical formatted-prompt map does not have the exact evaluation "
            "task IDs/order"
        )
    if prompt_format_contract.get("formatted_prompts_sha256") != _json_sha256(
        expected_formatted_prompts
    ):
        raise ValueError("Canonical formatted-prompt contract is internally inconsistent")

    provenance_path = source_dir / "provenance.json"
    results_path = source_dir / "steering_results.json"
    eval_ids_path = source_dir / "eval_task_ids.json"
    source_provenance, provenance_sha, provenance_bytes = _read_json_artifact(
        provenance_path
    )
    source_results, results_sha, results_bytes = _read_json_artifact(results_path)
    source_eval_ids, eval_ids_sha, eval_ids_bytes = _read_json_artifact(eval_ids_path)
    if not isinstance(source_provenance, dict):
        raise ValueError("Baseline source provenance.json must contain an object")
    if not isinstance(source_results, list):
        raise ValueError("Baseline source steering_results.json must contain a list")
    if not isinstance(source_eval_ids, dict):
        raise ValueError("Baseline source eval_task_ids.json must contain an object")

    source_contract = source_provenance.get("run_contract")
    source_contract_sha = source_provenance.get("run_contract_sha256")
    if not isinstance(source_contract, dict) or not isinstance(
        source_contract_sha, str
    ):
        raise ValueError("Baseline source provenance lacks a complete run contract")
    computed_contract_sha = _json_sha256(source_contract)
    if source_contract_sha != computed_contract_sha:
        raise ValueError(
            "Baseline source run-contract hash is internally inconsistent: "
            f"stated={source_contract_sha}, computed={computed_contract_sha}"
        )

    source_model = source_contract.get("model", {})
    for field in ("public_id", "revision", "model_weight_sha256"):
        if source_model.get(field) != model_contract.get(field):
            raise ValueError(
                f"Baseline source model {field} mismatch: "
                f"source={source_model.get(field)!r}, "
                f"requested={model_contract.get(field)!r}"
            )
    protocol_checks = {
        "dtype": (source_contract.get("dtype"), "bfloat16"),
        "use_4bit": (source_contract.get("use_4bit"), False),
        "max_new_tokens": (source_contract.get("max_new_tokens"), 1000),
        "temperature": (source_contract.get("temperature"), 0.0),
    }
    for field, (actual, expected) in protocol_checks.items():
        if actual != expected:
            raise ValueError(
                f"Baseline source {field} mismatch: expected {expected!r}, "
                f"got {actual!r}"
            )

    source_task_ids = source_contract.get("evaluation_task_ids")
    if source_task_ids != evaluation_task_ids:
        raise ValueError(
            "Baseline source evaluation task IDs/order do not exactly match "
            "the destination 50-task hold-out"
        )
    if source_eval_ids.get("task_ids") != evaluation_task_ids:
        raise ValueError(
            "Baseline source eval_task_ids.json does not match its requested "
            "50-task order"
        )
    if source_eval_ids.get("run_contract_sha256") != source_contract_sha:
        raise ValueError(
            "Baseline source eval_task_ids.json is not bound to the source "
            "run contract"
        )
    source_eval_ref = source_contract.get("evaluation_ids_source", {})
    if source_eval_ref.get("sha256") != evaluation_ids_source_sha256:
        raise ValueError(
            "Baseline source canonical evaluation-ID artefact hash does not "
            "match the destination"
        )
    source_tasks_ref = source_contract.get("tasks_source", {})
    if source_tasks_ref.get("sha256") != tasks_source_sha256:
        raise ValueError(
            "Baseline source tasks artefact hash does not match the destination"
        )

    def is_shared_vanilla(row: object) -> bool:
        return (
            isinstance(row, dict)
            and row.get("behaviour") == "shared"
            and row.get("method") == "vanilla"
            and row.get("alpha") == 0.0
        )

    source_baselines = [row for row in source_results if is_shared_vanilla(row)]
    foreign_rows = [row for row in source_results if not is_shared_vanilla(row)]
    if len(source_baselines) != 50:
        raise ValueError(
            "Baseline source must contain exactly 50 shared/vanilla/alpha=0 "
            f"records; found {len(source_baselines)}"
        )
    baseline_ids = [row.get("task_id") for row in source_baselines]
    if baseline_ids != evaluation_task_ids or len(set(baseline_ids)) != 50:
        raise ValueError(
            "Baseline source shared vanilla rows are duplicated, missing, or "
            "not in the exact evaluation-task order"
        )

    ignored_arm_counts: dict[str, int] = {}
    for row in foreign_rows:
        if isinstance(row, dict):
            arm = f"{row.get('behaviour')}::{row.get('method')}::alpha={row.get('alpha')}"
        else:
            arm = f"non_object::{type(row).__name__}"
        ignored_arm_counts[arm] = ignored_arm_counts.get(arm, 0) + 1

    source_record_hashes: list[str] = []
    reusable_rows: list[dict] = []
    row_reuse_common = {
        "source_dir": str(source_dir),
        "source_results_sha256": results_sha,
        "source_provenance_sha256": provenance_sha,
        "source_run_contract_sha256": source_contract_sha,
    }
    for task_id, row in zip(evaluation_task_ids, source_baselines):
        task = tasks_by_id.get(task_id)
        if task is None:
            raise ValueError(f"Canonical task {task_id!r} is unavailable")
        required = set(_REUSED_BASELINE_CONTENT_FIELDS) | {
            "behaviour",
            "method",
            "alpha",
            "layer",
            "mode",
            "vector_source",
            "vector_norm",
            "write_norm",
            "mean_abs_displacement",
            "run_contract_sha256",
        }
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(
                f"Baseline source row {task_id} is incomplete; missing {missing}"
            )
        if row.get("run_contract_sha256") != source_contract_sha:
            raise ValueError(
                f"Baseline source row {task_id} is not bound to the source contract"
            )
        if row.get("base_task_id") != task_id:
            raise ValueError(f"Baseline source row {task_id} has wrong base_task_id")
        if row.get("instruction") != task.get("prompt"):
            raise ValueError(
                f"Baseline source row {task_id} instruction differs from tasks source"
            )
        if row.get("category") != task.get("category", "unknown"):
            raise ValueError(
                f"Baseline source row {task_id} category differs from tasks source"
            )
        prompt = row.get("prompt")
        chain = row.get("chain")
        full_text = row.get("full_text")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"Baseline source row {task_id} has no formatted prompt")
        if not isinstance(chain, str) or not chain:
            raise ValueError(f"Baseline source row {task_id} has no generated chain")
        if full_text != prompt + chain:
            raise ValueError(
                f"Baseline source row {task_id} full_text is not prompt + chain"
            )
        if prompt != expected_formatted_prompts[task_id]:
            raise ValueError(
                f"Baseline source row {task_id} formatted prompt differs from the "
                "pinned tokenizer/chat-template output"
            )
        n_tokens = row.get("n_tokens")
        if (
            isinstance(n_tokens, bool)
            or not isinstance(n_tokens, int)
            or not (1 <= n_tokens <= 1000)
        ):
            raise ValueError(
                f"Baseline source row {task_id} has invalid n_tokens={n_tokens!r}"
            )
        if row.get("temperature") != 0.0 or row.get("seed") != 42:
            raise ValueError(
                f"Baseline source row {task_id} is not greedy temperature=0, seed=42"
            )
        if row.get("closed_think") != ("</think>" in chain):
            raise ValueError(
                f"Baseline source row {task_id} has inconsistent closed_think"
            )
        if row.get("hit_token_cap") != (n_tokens >= 1000):
            raise ValueError(
                f"Baseline source row {task_id} has inconsistent hit_token_cap"
            )
        expected_zero_fields = {
            "layer": None,
            "mode": "none",
            "vector_source": None,
            "vector_norm": 0.0,
            "write_norm": 0.0,
            "mean_abs_displacement": 0.0,
        }
        for field, expected in expected_zero_fields.items():
            if row.get(field) != expected:
                raise ValueError(
                    f"Baseline source row {task_id} is not a clean vanilla row: "
                    f"{field}={row.get(field)!r}"
                )

        source_record_sha = _json_sha256(row)
        source_record_hashes.append(source_record_sha)
        copied = {field: row[field] for field in _REUSED_BASELINE_CONTENT_FIELDS}
        copied.update(
            {
                "behaviour": "shared",
                "method": "vanilla",
                "alpha": 0.0,
                "layer": None,
                "mode": "none",
                "vector_source": None,
                "vector_norm": 0.0,
                "write_norm": 0.0,
                "mean_abs_displacement": 0.0,
                "baseline_reuse": {
                    **row_reuse_common,
                    "source_record_sha256": source_record_sha,
                },
            }
        )
        reusable_rows.append(copied)

    manifest = {
        "mode": "validated_shared_vanilla_copy",
        "source_dir": str(source_dir),
        "source_files": {
            "provenance": {
                "path": str(provenance_path),
                "sha256": provenance_sha,
                "bytes": provenance_bytes,
            },
            "steering_results": {
                "path": str(results_path),
                "sha256": results_sha,
                "bytes": results_bytes,
            },
            "eval_task_ids": {
                "path": str(eval_ids_path),
                "sha256": eval_ids_sha,
                "bytes": eval_ids_bytes,
            },
        },
        "source_run_contract_sha256": source_contract_sha,
        "source_record_count": len(source_results),
        "selected_shared_vanilla_count": len(reusable_rows),
        "selected_task_ids": evaluation_task_ids,
        "selected_source_records_sha256": _json_sha256(source_baselines),
        "selected_source_record_sha256s": source_record_hashes,
        "ignored_nonbaseline_count": len(foreign_rows),
        "ignored_nonbaseline_arms": ignored_arm_counts,
        "foreign_arms_used": False,
        "validated_protocol": {
            "model_revision": model_contract["revision"],
            "model_weight_sha256": model_contract["model_weight_sha256"],
            "dtype": "bfloat16",
            "use_4bit": False,
            "max_new_tokens": 1000,
            "decoding": "greedy",
            "temperature": 0.0,
            "evaluation_ids_source_sha256": evaluation_ids_source_sha256,
            "tasks_source_sha256": tasks_source_sha256,
            "prompt_format": prompt_format_contract,
        },
    }
    return reusable_rows, manifest


def _install_reused_baselines(
    existing_results: list[dict],
    reusable_rows: list[dict],
    run_contract_sha256: str,
) -> tuple[list[dict], bool]:
    """Install a complete baseline snapshot, or validate an atomic resume."""
    expected_rows = []
    for row in reusable_rows:
        materialized = dict(row)
        materialized["run_contract_sha256"] = run_contract_sha256
        expected_rows.append(materialized)

    existing_baselines = [
        row
        for row in existing_results
        if row.get("behaviour") == "shared"
        and row.get("method") == "vanilla"
        and row.get("alpha") == 0.0
    ]
    if existing_results:
        if existing_baselines != expected_rows:
            raise ValueError(
                "Destination contains missing, duplicate, or altered reused "
                "baselines; refusing to mix or silently repair the checkpoint"
            )
        return existing_results, False
    return expected_rows, True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generation-only h<-h-v bridge at Venhoff's Qwen-1.5B layers"
    )
    parser.add_argument(
        "--behaviours",
        nargs="+",
        default=list(DEFAULT_BEHAVIOURS),
        choices=list(VENHOFF_LAYERS),
        help=(
            "Behaviours to run (default: all four published behaviours at "
            "their exact Qwen-1.5B layers)."
        ),
    )
    parser.add_argument(
        "--max-eval-tasks",
        type=int,
        default=None,
        help="Use the first N of the same stratified hold-out (cheap smoke/subset).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1000,
        help="Generation cap (default: 1000, matching Venhoff's evaluation).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Constant-offset coefficient (default: 1, matching Venhoff).",
    )
    parser.add_argument(
        "--vectors-dir",
        type=Path,
        default=DEFAULT_VECTORS_DIR,
        help=(
            "Hybrid user vectors used by the bridge (default: exact current E1 "
            "directions for backtracking/example-testing plus reconstructed L18 "
            "directions for uncertainty-estimation/adding-knowledge, all at "
            "Venhoff's published layers)."
        ),
    )
    parser.add_argument(
        "--scale-convention",
        choices=["published-norm", "own-row-pooled-proxy"],
        default="published-norm",
        help=(
            "Norm for user directions: released-vector norm (default, matching "
            "the author's amount), or an explicitly non-exact own-data row-pooled "
            "proxy sensitivity. Ignored for exact --released-mean-vectors."
        ),
    )
    parser.add_argument(
        "--released-mean-vectors",
        type=Path,
        default=None,
        help=(
            "Optional path to the released Qwen-1.5B mean_vectors_*.pt file. "
            "When set, use exact released vectors instead of user directions."
        ),
    )
    parser.add_argument(
        "--allow-unverified-released-vectors",
        action="store_true",
        help="Allow a supplied .pt whose SHA-256 differs from the audited release.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Separate output directory; source-specific safe default if omitted.",
    )
    parser.add_argument(
        "--baseline-source-dir",
        type=Path,
        default=None,
        help=(
            "Optional prior bridge output whose complete 50-task shared vanilla "
            "baseline should be validated and copied into a fresh output. The "
            "source protocol and artefact hashes are bound into the new run."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Model dtype (default: bfloat16, matching Venhoff).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help=(
            "Local pinned model snapshot. Default resolves the audited "
            f"Hugging Face revision {MODEL_REVISION} from the standard cache."
        ),
    )
    parser.add_argument("--4bit", action="store_true", dest="use_4bit")
    args = parser.parse_args()

    # argparse choices validates each item, but duplicates would waste generation
    # and make resume provenance harder to read.
    behaviours = list(dict.fromkeys(args.behaviours))
    if args.max_eval_tasks is not None and args.max_eval_tasks < 1:
        parser.error("--max-eval-tasks must be >= 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be >= 1")
    if args.alpha <= 0:
        parser.error("--alpha must be > 0 for the suppression bridge")
    if args.use_4bit:
        parser.error("--4bit is not protocol-compatible with Venhoff's bf16 run")
    if args.dtype != "bfloat16":
        logger.warning(
            "dtype=%s is a protocol deviation; Venhoff used bfloat16", args.dtype
        )

    model_path = args.model_path or _default_pinned_snapshot()
    model_contract = _verify_model_snapshot(model_path)

    if args.released_mean_vectors is None:
        vector_data, vector_contract = load_user_directions(
            args.vectors_dir,
            behaviours,
            scale_convention=args.scale_convention,
        )
        default_suffix = (
            "hybrid_user_publishednorm"
            if args.scale_convention == "published-norm"
            else "hybrid_user_rowpooledproxy"
        )
    else:
        if args.scale_convention != "published-norm":
            parser.error(
                "--scale-convention applies only to user directions; omit it "
                "when using --released-mean-vectors"
            )
        vector_data, vector_contract = load_released_vectors(
            args.released_mean_vectors,
            behaviours,
            allow_unverified=args.allow_unverified_released_vectors,
        )
        default_suffix = "released"

    out_dir = args.out_dir or Path(
        f"results/eval/{MODEL_SHORT}__venhoff_constant_all4_{default_suffix}"
    )
    if (
        args.baseline_source_dir is not None
        and args.baseline_source_dir.expanduser().resolve()
        == out_dir.expanduser().resolve()
    ):
        parser.error(
            "--baseline-source-dir and --out-dir must be different; baseline "
            "reuse always writes a fresh, separately contracted artefact"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "steering_results.json"

    # Reuse the exact E1 hold-out IDs rather than recomputing a split.  These are
    # also the IDs excluded by the bridge-vector source builds.
    source_ids_path = Path("results/eval/R1-1.5B__E1/eval_task_ids.json")
    if not source_ids_path.is_file():
        raise FileNotFoundError(f"Canonical E1 eval IDs not found: {source_ids_path}")
    tasks_source_path = Path("data/tasks_final.json")
    with tasks_source_path.open() as f:
        all_tasks = json.load(f)
    by_id = {task["id"]: task for task in all_tasks}
    with source_ids_path.open() as f:
        source_split = json.load(f)
    missing_ids = [task_id for task_id in source_split["task_ids"] if task_id not in by_id]
    if missing_ids:
        raise ValueError(f"E1 eval IDs absent from tasks_final.json: {missing_ids}")
    test_tasks = [by_id[task_id] for task_id in source_split["task_ids"]]
    split_rule = source_split.get("rule", "canonical E1 eval_task_ids order")
    if args.max_eval_tasks is not None:
        test_tasks = test_tasks[: args.max_eval_tasks]
        split_rule += f" (subset: first {args.max_eval_tasks})"
    category_counts: dict[str, int] = {}
    for task in test_tasks:
        category = task.get("category", "unknown")
        category_counts[category] = category_counts.get(category, 0) + 1
    evaluation_ids_source_sha256 = _sha256(source_ids_path)
    tasks_source_sha256 = _sha256(tasks_source_path)
    run_contract = {
        "model": model_contract,
        "dtype": args.dtype,
        "use_4bit": bool(args.use_4bit),
        "behaviours": behaviours,
        "layers": {b: VENHOFF_LAYERS[b] for b in behaviours},
        "operator": "h <- h - alpha * v",
        "mode": "constant_subtract",
        "alpha": float(args.alpha),
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": 0.0,
        "annotation": "not_run_by_this_runner",
        "arms": ["shared vanilla baseline", "venhoff_constant_subtract"],
        "vector_source": vector_contract,
        "evaluation_split": split_rule,
        "evaluation_ids_source": {
            "path": str(source_ids_path),
            "sha256": evaluation_ids_source_sha256,
        },
        "tasks_source": {
            "path": str(tasks_source_path.resolve()),
            "sha256": tasks_source_sha256,
        },
        "evaluation_task_ids": [task["id"] for task in test_tasks],
    }
    reusable_baselines = None
    baseline_reuse_manifest = None
    if args.baseline_source_dir is not None:
        expected_formatted_prompts, prompt_format_contract = (
            _canonical_formatted_prompts(model_contract, test_tasks)
        )
        reusable_baselines, baseline_reuse_manifest = _prepare_baseline_reuse(
            args.baseline_source_dir,
            model_contract=model_contract,
            dtype=args.dtype,
            use_4bit=args.use_4bit,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            evaluation_task_ids=[task["id"] for task in test_tasks],
            evaluation_ids_source_sha256=evaluation_ids_source_sha256,
            tasks_source_sha256=tasks_source_sha256,
            tasks_by_id=by_id,
            expected_formatted_prompts=expected_formatted_prompts,
            prompt_format_contract=prompt_format_contract,
        )
        run_contract["baseline_reuse"] = baseline_reuse_manifest
    run_contract_sha256 = _json_sha256(run_contract)

    provenance_path = out_dir / "provenance.json"
    if provenance_path.exists():
        with provenance_path.open() as f:
            existing_provenance = json.load(f)
        existing_hash = existing_provenance.get("run_contract_sha256")
        if existing_hash != run_contract_sha256:
            raise ValueError(
                f"Refusing to overwrite {out_dir} with a different run contract: "
                f"existing={existing_hash}, requested={run_contract_sha256}. "
                "Choose a fresh --out-dir."
            )

    _atomic_json(
        {
            "task_ids": [task["id"] for task in test_tasks],
            "category_counts": category_counts,
            "rule": split_rule,
            "run_contract_sha256": run_contract_sha256,
        },
        out_dir / "eval_task_ids.json",
    )
    prov = provenance(args)
    prov["run_contract"] = run_contract
    prov["run_contract_sha256"] = run_contract_sha256
    _atomic_json(prov, provenance_path)
    if baseline_reuse_manifest is not None:
        _atomic_json(
            {
                "run_contract_sha256": run_contract_sha256,
                "baseline_reuse": baseline_reuse_manifest,
            },
            out_dir / "baseline_reuse.json",
        )

    results = _load_existing_results(
        results_path, vector_contract["kind"], run_contract_sha256
    )
    if reusable_baselines is not None:
        results, installed = _install_reused_baselines(
            results, reusable_baselines, run_contract_sha256
        )
        if installed:
            _atomic_json(results, results_path)
            logger.info(
                "Reused 50 validated shared vanilla baselines from %s; "
                "%d non-baseline source rows were ignored",
                baseline_reuse_manifest["source_dir"],
                baseline_reuse_manifest["ignored_nonbaseline_count"],
            )

    logger.info(
        "Bridge contract: %s; h<-h-alpha*v, alpha=%g; behaviours=%s",
        vector_contract["kind"],
        args.alpha,
        behaviours,
    )
    for behaviour in behaviours:
        actual_norm = float(np.linalg.norm(vector_data[behaviour]["vector"]))
        logger.info(
            "  %s: layer=%d, ||v||=%.6f",
            behaviour,
            vector_data[behaviour]["layer"],
            actual_norm,
        )

    logger.info(
        "Loading pinned model: %s @ %s (%s)",
        MODEL_ID,
        MODEL_REVISION,
        args.dtype,
    )
    model, tokenizer = load_model(
        model_contract["local_snapshot"],
        dtype=args.dtype,
        use_4bit=args.use_4bit,
    )

    done = {
        (r["behaviour"], r["method"], float(r["alpha"]), r["task_id"])
        for r in results
    }

    # Shared unsteered baseline.  It is the only reference arm and is generated
    # once per task, not once per behaviour.
    for task in tqdm(test_tasks, desc="vanilla (shared baseline)"):
        key = ("shared", "vanilla", 0.0, task["id"])
        if key in done:
            continue
        generated = generate_chain(
            model,
            tokenizer,
            task["prompt"],
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            seed=42,
            model_id=MODEL_ID,
        )
        results.append(
            _task_record(
                task,
                generated,
                max_new_tokens=args.max_new_tokens,
                run_contract_sha256=run_contract_sha256,
                behaviour="shared",
                method="vanilla",
                alpha=0.0,
                layer=None,
                mode="none",
                vector_source=None,
                vector_norm=0.0,
                write_norm=0.0,
                mean_abs_displacement=0.0,
            )
        )
        done.add(key)
        _atomic_json(results, results_path)

    # One fixed-offset suppression arm per requested behaviour.  There is no
    # manifold sweep and no random/control family in this runner.
    for behaviour in behaviours:
        layer = vector_data[behaviour]["layer"]
        vector = vector_data[behaviour]["vector"]
        vector_norm = float(np.linalg.norm(vector.astype(np.float64)))
        steered = SteeredModel(
            model,
            tokenizer,
            vector=vector,
            layer=layer,
            alpha=args.alpha,
            mode="constant_subtract",
            energy_scale=1.0,
        )
        for task in tqdm(test_tasks, desc=f"{behaviour} h<-h-v @ L{layer}"):
            key = (
                behaviour,
                "venhoff_constant_subtract",
                float(args.alpha),
                task["id"],
            )
            if key in done:
                continue
            generated = steered.generate(
                task["prompt"],
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                seed=42,
            )
            results.append(
                _task_record(
                    task,
                    generated,
                    max_new_tokens=args.max_new_tokens,
                    run_contract_sha256=run_contract_sha256,
                    behaviour=behaviour,
                    method="venhoff_constant_subtract",
                    alpha=float(args.alpha),
                    layer=layer,
                    mode="constant_subtract",
                    vector_source=vector_contract["kind"],
                    source_vector_sha256=vector_data[behaviour].get(
                        "source_sha256"
                    ),
                    derived_vector_sha256=vector_data[behaviour][
                        "derived_vector_sha256"
                    ],
                    vector_norm=vector_norm,
                    write_norm=abs(float(args.alpha)) * vector_norm,
                    realized_write_norm=generated.get(
                        "mean_abs_displacement"
                    ),
                    mean_abs_proj=generated.get("mean_abs_proj"),
                    mean_abs_displacement=generated.get("mean_abs_displacement"),
                )
            )
            done.add(key)
            _atomic_json(results, results_path)

    # No annotation path is imported or called.  These metrics contain only
    # generation length, repetition, and degenerate-output rate; behaviour means
    # remain null until a separate, explicit annotation step is run.
    generation_summary = aggregate_results(
        results, [], target_behaviours=behaviours
    )
    save_summary(generation_summary, out_dir / "generation_metrics.json")

    logger.info("Generation complete: %d records -> %s", len(results), results_path)
    logger.info(
        "No annotation API was called. Behaviour fractions require a separate "
        "annotation pass over steering_results.json."
    )
    print(f"\nResults saved -> {out_dir}")


if __name__ == "__main__":
    main()
