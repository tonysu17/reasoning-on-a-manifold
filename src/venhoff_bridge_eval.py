"""Guarded annotation and paired evaluation for the Venhoff steering bridge.

The generation runner :mod:`07f_venhoff_fixed_bridge` deliberately performs no
paid annotation.  This module is the corresponding local annotation/evaluation
stage.  It has three contracts:

* the paid stage is bound to an immutable, hash-checked 250-record generation
  artefact and to a durable USD 15 attempt journal;
* a missing, empty, transport-incomplete, or coverage-incomplete annotation is
  unresolved, never a measured zero;
* the primary score reproduces the *estimand* in Venhoff et al.'s released
  ``steering/evaluate_steering.py``: Qwen-token count assigned to the target
  label divided by the token count assigned to the four steered labels.  The
  six-label token denominator and the repository's sentence-fraction endpoint
  are reported as symmetric sensitivities.

The released code locates every repeated annotation with ``str.find`` from the
start of the response and silently skips unmatched text.  The primary endpoint
fixes both issues with occurrence-aware, coverage-gated annotation order.  A
separately named bug-compatible sensitivity preserves the released arithmetic
for direct comparison.  Alignment space is persisted per row.

No function in this module calls an API unless :func:`run_annotation` is
invoked with a manifest SHA-256 and the caller has already made the explicit
CLI spend confirmation enforced by ``07g_annotate_venhoff_bridge.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

from src.annotation import (
    ANNOTATION_MODEL,
    ANNOTATION_PROMPT_VERSION,
    TARGET_BEHAVIOURS,
    VALID_LABELS,
    annotate_chains,
    annotation_initial_request_count,
    max_prompt_chars,
)
from src.annotation_budget import (
    SCHEMA_VERSION as ATTEMPT_GUARD_SCHEMA,
    AnnotationAttemptGuard,
    AnnotationAttemptLimitError,
)
from src.annotation_coverage import (
    COVERAGE_RULE_VERSION,
    normalise,
    row_is_coverage_complete,
)
from src.delta_floor import paired_bootstrap_mean
from src.steering_analysis import holm_bonferroni


BRIDGE_METHOD = "venhoff_constant_subtract"
SHARED_BEHAVIOUR = "shared"
VANILLA_METHOD = "vanilla"
EXPECTED_TASKS = 50
EXPECTED_RECORDS = EXPECTED_TASKS * (1 + len(TARGET_BEHAVIOURS))
EXPECTED_MAX_NEW_TOKENS = 1000
EXPECTED_ALPHA = 1.0
EXPECTED_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
EXPECTED_MODEL_REVISION = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
EXPECTED_MODEL_WEIGHT_SHA256 = (
    "58858233513d76b8703e72eed6ce16807b523328188e13329257fb9594462945"
)
EXPECTED_PROMPT_PREFIX = "<｜begin▁of▁sentence｜><｜User｜>"
EXPECTED_PROMPT_SUFFIX = "<｜Assistant｜><think>\n"
EXPECTED_TOKENIZER_JSON_SHA256 = (
    "88145e3c3249adc2546ede277e9819d6e405e19072456e4b521cbc724bd60773"
)
EXPECTED_EVAL_IDS_SHA256 = (
    "c8f2d921e4f8506c68b488442ed51f05e6a14dbac638c66fc41569d7bbd37bcd"
)
EXPECTED_EVAL_TASK_ID_LIST_SHA256 = (
    "ab8da4610eac942f67b8b3880eb49eae551a0fb6eb0e5a8e8fcbe0fca9cc0e2b"
)
EXPECTED_TASK_IDS = (
    "CAUS_097", "CAUS_098", "CAUS_099", "CAUS_100", "CAUS_101",
    "CREA_095", "CREA_096", "CREA_097", "CREA_098", "CREA_099",
    "LATE_095", "LATE_096", "LATE_097", "LATE_098", "LATE_099",
    "MATH_102", "MATH_103", "MATH_104", "MATH_105", "MATH_106",
    "PATT_106", "PATT_107", "PATT_108", "PATT_109", "PATT_110",
    "PROB_111", "PROB_112", "PROB_113", "PROB_114", "PROB_115",
    "SCIE_101", "SCIE_102", "SCIE_103", "SCIE_104", "SCIE_105",
    "SPAT_112", "SPAT_113", "SPAT_114", "SPAT_115", "SPAT_116",
    "SYST_096", "SYST_097", "SYST_098", "SYST_099", "SYST_100",
    "VERB_101", "VERB_102", "VERB_103", "VERB_104", "VERB_105",
)
EXPECTED_TASKS_SHA256 = (
    "a4ba180d5722e016c95945867ca6fbcf7abb48640f4b8d0c9e6111c572051b1d"
)
EXPECTED_LAYERS = {
    "backtracking": 17,
    "uncertainty-estimation": 18,
    "example-testing": 15,
    "adding-knowledge": 18,
}
EXPECTED_VECTOR_NORMS = {
    "backtracking": 70.868286132812,
    "uncertainty-estimation": 79.720710754395,
    "example-testing": 56.323959350586,
    "adding-knowledge": 79.720710754395,
}
EXPECTED_VECTOR_METADATA_SHA256 = (
    "ea627bdcc8bc6f27115d3ee670a9527a9117a6109672ab3adb48c3dc40407005"
)
ABANDONED_SIXLABEL_VECTOR_METADATA_SHA256 = (
    "b5e1bcad30b7741cc71a799c7b00a66ebc649c0e349ac15827403aa80fb89196"
)
EXPECTED_HYBRID_VECTOR_DIRNAME = "R1-1.5B__venhoff_bridge_hybrid"
EXPECTED_VECTOR_DESCRIPTION = (
    "repository hybrid exact-layer unit directions: the current E1 directions "
    "are retained exactly for backtracking/example-testing, while "
    "uncertainty-estimation/adding-knowledge use the separately reconstructed "
    "mixed-vintage L18 directions"
)
EXPECTED_HYBRID_CLAIM_BOUNDARY = (
    "Backtracking and example-testing hold the current E1 directions fixed while "
    "changing operator/write dose. Uncertainty-estimation and adding-knowledge "
    "use reconstructed L18 directions and are a mixed-vintage sensitivity, not "
    "the same isolation."
)
EXPECTED_HYBRID_CONSTRUCTION = (
    "byte-for-byte composition of existing unit-vector artefacts; no directions "
    "recomputed"
)
EXPECTED_UNIT_VECTOR_SHA256 = {
    "backtracking": "10b12bdd2e90afafee4198e13156d30e585cde41f29993dca2ca91d7c221b2ca",
    "uncertainty-estimation": "1462474e842167ed00b3cf89ab537f5dc7066273dc9503d320bf186986b371c6",
    "example-testing": "e943cd417ff650df5eea05366fae33fe7bdf355de9213310770138168a43a3bb",
    "adding-knowledge": "b57a6301a392994008615fc55d4e39a899e4864e76520c25fba5cbbfdd800a5b",
}
EXPECTED_SCALED_VECTOR_SHA256 = {
    "backtracking": "cd6007612fde90c36e91c5daac7cf47375d916b8046394a4d0271b5d3b2dba18",
    "uncertainty-estimation": "3e65be515a7d2b62022299b9515ac44484178c4c39c556a35875160cc9229bb7",
    "example-testing": "6497216ed8c3ffe714ab067b5c95de3362ede295ff5b46f0a773a74954984766",
    "adding-knowledge": "5c4aaa1d5a1ccf612f9f5b245db6c751daaf29afdd18768b5e8b921fefc3d939",
}
EXPECTED_VECTOR_SOURCE_CONTRACT = {
    "backtracking": "current_E1_pooled",
    "uncertainty-estimation": "reconstructed_exact_layer",
    "example-testing": "current_E1_pooled",
    "adding-knowledge": "reconstructed_exact_layer",
}
EXPECTED_VECTOR_SOURCE_KIND = {
    "backtracking": "exact_current_E1_pooled_single_direction",
    "uncertainty-estimation": "reconstructed_exact_layer_six_label_direction",
    "example-testing": "exact_current_E1_pooled_single_direction",
    "adding-knowledge": "reconstructed_exact_layer_six_label_direction",
}
EXPECTED_SOURCE_METADATA_SHA256 = {
    "current_E1_pooled": (
        "0f756b9cbe1ec8f470625000784bdae1a07338987e01e2fcb1f7268e0eddd886"
    ),
    "reconstructed_exact_layer": ABANDONED_SIXLABEL_VECTOR_METADATA_SHA256,
}
EXPECTED_SOURCE_METADATA_PATH = {
    "current_E1_pooled": (
        "results/steering_vectors/R1-1.5B__E1_pooled/metadata.json"
    ),
    "reconstructed_exact_layer": (
        "results/steering_vectors/R1-1.5B__venhoff_bridge_sixlabel/metadata.json"
    ),
}
EXPECTED_VECTOR_INTERPRETATION = {
    "backtracking": (
        "Exact current thesis E1 direction; changing the operator and write dose "
        "can be isolated while holding this direction fixed."
    ),
    "uncertainty-estimation": (
        "Exact-layer six-label reconstruction from mixed clip-window vintages; "
        "this arm does not isolate only operator/write dose relative to the thesis "
        "E1 direction."
    ),
    "example-testing": (
        "Exact current thesis E1 direction; changing the operator and write dose "
        "can be isolated while holding this direction fixed."
    ),
    "adding-knowledge": (
        "Exact-layer six-label reconstruction from mixed clip-window vintages; "
        "this arm does not isolate only operator/write dose relative to the thesis "
        "E1 direction."
    ),
}
EXPECTED_PUBLISHED_SCALE_SOURCE = {
    "repository": "https://github.com/cvenhoff/steering-thinking-llms",
    "repository_commit_audited": "93259bc3410c99293351df41141cd16b4110422a",
    "vector_file_commit": "8c8f0a2d2ee41b1173877d488e0ae8565b4de2cc",
    "mean_vector_file_sha256": (
        "bbf7bcac3758df3236d485c41dcf3033a53fd6333a4abd555290e8ef71419412"
    ),
}

ANNOTATION_MAX_OUTPUT_TOKENS = 2800
ANNOTATION_MAX_RETRIES = 2
ANNOTATION_WINDOW_TOKENS = EXPECTED_MAX_NEW_TOKENS
SPEND_CEILING_USD = 15.0
MAX_COST_PER_ATTEMPT_USD = 0.05
RATE_CARD = {
    "input_usd_per_mtok": 3.0,
    "output_usd_per_mtok": 15.0,
    "tokens_per_char_upper": 0.4,
}

PLAN_SCHEMA = "venhoff-fixed-bridge-annotation-plan-1"
REPORT_SCHEMA = "venhoff-fixed-bridge-evaluation-1"
SCORED_SCHEMA = "venhoff-fixed-bridge-scored-records-1"

CORRECTED_FOUR_TARGET_METRIC = "venhoff_four_target_token_fraction"
RELEASED_BUG_COMPATIBLE_METRIC = (
    "venhoff_released_bug_compatible_four_target_token_fraction"
)
SIX_LABEL_TOKEN_METRIC = "six_label_token_fraction"
SIX_LABEL_SENTENCE_METRIC = "six_label_sentence_fraction"

_RELEASED_ANNOTATION_RE = re.compile(
    r'\["(\S+?)"\](.*?)\["end-section"\]', re.DOTALL
)

VENHOFF_ANNOTATION_REGION_POLICY = {
    "include_post_think": False,
    "exclude_final_answer_suffix": False,
    "exclude_degenerate_repetition": False,
}

DEFAULT_BOUND_PATHS = (
    "07g_annotate_venhoff_bridge.py",
    "src/venhoff_bridge_eval.py",
    "src/annotation.py",
    "src/annotation_budget.py",
    "src/annotation_coverage.py",
    "src/text_offsets.py",
)

# One repository-global ledger for this provisional experiment.  It is
# deliberately outside an eval directory, so copying generations to a fresh
# directory cannot reset the user's cumulative USD 15 ceiling.  Journal events
# also bind the manifest digest; a changed manifest after spending therefore
# fails closed instead of starting a second allowance.
GLOBAL_JOURNAL_RELATIVE = Path(
    "results/eval/_annotation_spend_ledgers/venhoff_fixed_bridge_usd15.jsonl"
)

DEDUP_KEYS = (
    "task_id",
    "behaviour",
    "method",
    "alpha",
    "run_contract_sha256",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_json(value: object, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def _load_json(path: Path) -> object:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _repo_relative(path: Path, root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path(root).resolve()))
    except ValueError as exc:
        raise ValueError(f"artefact is outside repository root: {resolved}") from exc


def _alpha_equal(left: object, right: float) -> bool:
    try:
        return abs(float(left) - float(right)) <= 1e-9
    except (TypeError, ValueError):
        return False


def _record_key(row: dict) -> tuple:
    return tuple(row.get(key) for key in DEDUP_KEYS)


def _canonical_hybrid_metadata(repository_root: Path) -> dict:
    """Load and independently verify the local final hybrid vector asset.

    The generation contract contains hashes and a copy of the build provenance,
    but those are still JSON claims.  Before paid annotation, bind those claims
    to the repository's audited hybrid metadata and to each vector/source file.
    """
    hybrid_dir = (
        Path(repository_root)
        / "results"
        / "steering_vectors"
        / EXPECTED_HYBRID_VECTOR_DIRNAME
    )
    metadata_path = hybrid_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"audited hybrid vector metadata is missing: {metadata_path}"
        )
    if sha256_file(metadata_path) != EXPECTED_VECTOR_METADATA_SHA256:
        raise ValueError("local final hybrid vector metadata hash has changed")
    metadata = _load_json(metadata_path)
    if not isinstance(metadata, dict):
        raise ValueError("final hybrid vector metadata must be a JSON object")

    build = metadata.get("_provenance")
    if not isinstance(build, dict):
        raise ValueError("final hybrid vector metadata lacks build provenance")
    if build.get("layers") != EXPECTED_LAYERS:
        raise ValueError("final hybrid vector metadata has the wrong exact layers")
    if build.get("claim_boundary") != EXPECTED_HYBRID_CLAIM_BOUNDARY:
        raise ValueError("final hybrid vector claim boundary has changed")
    if build.get("construction") != EXPECTED_HYBRID_CONSTRUCTION:
        raise ValueError("final hybrid vector construction contract has changed")

    for behaviour in TARGET_BEHAVIOURS:
        record = metadata.get(behaviour)
        source_contract = EXPECTED_VECTOR_SOURCE_CONTRACT[behaviour]
        source_metadata_path = EXPECTED_SOURCE_METADATA_PATH[source_contract]
        source_dirname = Path(source_metadata_path).parent.name
        source_file = (
            f"results/steering_vectors/{source_dirname}/{behaviour}_single.npy"
        )
        expected_fields = {
            "layer": EXPECTED_LAYERS[behaviour],
            "source_kind": EXPECTED_VECTOR_SOURCE_KIND[behaviour],
            "source_file": source_file,
            "source_sha256": EXPECTED_UNIT_VECTOR_SHA256[behaviour],
            "source_metadata": source_metadata_path,
            "source_metadata_sha256": EXPECTED_SOURCE_METADATA_SHA256[
                source_contract
            ],
            "vector_file": (
                "results/steering_vectors/"
                f"{EXPECTED_HYBRID_VECTOR_DIRNAME}/{behaviour}_single.npy"
            ),
            "vector_sha256": EXPECTED_UNIT_VECTOR_SHA256[behaviour],
            "byte_identical_to_source": True,
            "array_equal_to_source": True,
            "interpretation": EXPECTED_VECTOR_INTERPRETATION[behaviour],
        }
        if not isinstance(record, dict) or any(
            record.get(key) != value for key, value in expected_fields.items()
        ):
            raise ValueError(
                f"final hybrid source interpretation differs for {behaviour}"
            )

        vector_path = hybrid_dir / f"{behaviour}_single.npy"
        source_path = Path(repository_root) / source_file
        source_meta_path = Path(repository_root) / source_metadata_path
        for path, expected_sha, description in (
            (vector_path, EXPECTED_UNIT_VECTOR_SHA256[behaviour], "hybrid vector"),
            (source_path, EXPECTED_UNIT_VECTOR_SHA256[behaviour], "source vector"),
            (
                source_meta_path,
                EXPECTED_SOURCE_METADATA_SHA256[source_contract],
                "source metadata",
            ),
        ):
            if not path.is_file() or sha256_file(path) != expected_sha:
                raise ValueError(
                    f"{behaviour} {description} is absent or has changed hash"
                )
        vector = np.load(vector_path, allow_pickle=False)
        if (
            vector.shape != (1536,)
            or vector.dtype != np.float32
            or not np.isfinite(vector).all()
            or not math.isclose(
                float(np.linalg.norm(vector)), 1.0, rel_tol=1e-6, abs_tol=1e-6
            )
        ):
            raise ValueError(
                f"{behaviour} final hybrid vector is not finite unit float32[1536]"
            )
    return metadata


def _validate_hybrid_vector_contract(
    vector_source: object, repository_root: Path
) -> None:
    """Pin the run to the final hybrid directions and published write doses."""
    if (
        not isinstance(vector_source, dict)
        or vector_source.get("kind") != "user_direction_published_norm"
        or vector_source.get("description") != EXPECTED_VECTOR_DESCRIPTION
    ):
        raise ValueError(
            "bridge annotation is pinned to the final hybrid user directions "
            "at published norms"
        )
    vectors_dir = vector_source.get("vectors_dir")
    if (
        not isinstance(vectors_dir, str)
        or Path(vectors_dir).name != EXPECTED_HYBRID_VECTOR_DIRNAME
    ):
        raise ValueError("bridge run does not identify the final hybrid vector asset")

    scale = vector_source.get("scale")
    if (
        not isinstance(scale, dict)
        or scale.get("convention") != "published-norm"
        or scale.get("values") != EXPECTED_VECTOR_NORMS
        or scale.get("source") != EXPECTED_PUBLISHED_SCALE_SOURCE
    ):
        raise ValueError(
            "bridge vector magnitudes/source do not match the four released norms"
        )

    canonical = _canonical_hybrid_metadata(repository_root)
    metadata = vector_source.get("metadata")
    if (
        not isinstance(metadata, dict)
        or metadata.get("sha256") != EXPECTED_VECTOR_METADATA_SHA256
    ):
        raise ValueError(
            "bridge vector metadata hash is not the audited final hybrid asset"
        )
    metadata_path = metadata.get("path")
    if (
        not isinstance(metadata_path, str)
        or Path(metadata_path).name != "metadata.json"
        or Path(metadata_path).parent.name != EXPECTED_HYBRID_VECTOR_DIRNAME
    ):
        raise ValueError("bridge run metadata path is not the final hybrid asset")

    build = metadata.get("build_provenance")
    if not isinstance(build, dict):
        raise ValueError("bridge run lacks final hybrid build provenance")
    if build.get("claim_boundary") != EXPECTED_HYBRID_CLAIM_BOUNDARY:
        raise ValueError("bridge run has a changed hybrid claim boundary")
    if build.get("construction") != EXPECTED_HYBRID_CONSTRUCTION:
        raise ValueError("bridge run has a changed hybrid construction contract")
    if build.get("layers") != EXPECTED_LAYERS:
        raise ValueError("bridge run hybrid provenance has the wrong layers")
    source_contracts = build.get("source_contracts")
    if not isinstance(source_contracts, dict) or set(source_contracts) != set(
        EXPECTED_SOURCE_METADATA_SHA256
    ):
        raise ValueError("bridge run lacks the two audited hybrid source contracts")
    observed_sources: dict[str, str] = {}
    for source_name, expected_metadata_sha in EXPECTED_SOURCE_METADATA_SHA256.items():
        source_record = source_contracts.get(source_name)
        if (
            not isinstance(source_record, dict)
            or source_record.get("metadata")
            != EXPECTED_SOURCE_METADATA_PATH[source_name]
            or source_record.get("metadata_sha256") != expected_metadata_sha
            or not isinstance(source_record.get("behaviours"), list)
        ):
            raise ValueError(f"bridge hybrid source contract changed: {source_name}")
        for behaviour in source_record["behaviours"]:
            if behaviour in observed_sources:
                raise ValueError(
                    f"bridge hybrid source interpretation duplicates {behaviour}"
                )
            observed_sources[behaviour] = source_name
    if observed_sources != EXPECTED_VECTOR_SOURCE_CONTRACT:
        raise ValueError("bridge per-behaviour source interpretation has changed")
    if (
        source_contracts["current_E1_pooled"].get("source_git_commit") is not None
        or source_contracts["reconstructed_exact_layer"].get(
            "source_contract_status"
        )
        != "mixed clip-window vintages"
    ):
        raise ValueError("bridge hybrid source provenance status has changed")
    if build != canonical.get("_provenance"):
        raise ValueError("bridge hybrid build provenance differs from audited metadata")

    unit_vectors = vector_source.get("unit_vectors")
    if not isinstance(unit_vectors, dict) or set(unit_vectors) != set(TARGET_BEHAVIOURS):
        raise ValueError("bridge run lacks the four audited hybrid unit-vector records")
    for behaviour in TARGET_BEHAVIOURS:
        vector_record = unit_vectors.get(behaviour)
        vector_path = vector_record.get("path") if isinstance(vector_record, dict) else None
        if (
            not isinstance(vector_record, dict)
            or not isinstance(vector_path, str)
            or Path(vector_path).name != f"{behaviour}_single.npy"
            or Path(vector_path).parent.name != EXPECTED_HYBRID_VECTOR_DIRNAME
            or vector_record.get("sha256") != EXPECTED_UNIT_VECTOR_SHA256[behaviour]
            or vector_record.get("derived_scaled_vector_sha256")
            != EXPECTED_SCALED_VECTOR_SHA256[behaviour]
        ):
            raise ValueError(
                f"bridge direction hash differs for {behaviour} in the final "
                "hybrid asset"
            )


def venhoff_reasoning_slice(chain: str) -> tuple[str, int, int]:
    """Return Venhoff's stripped reasoning text from a generated suffix.

    The pinned chat prompt already ends in ``<think>\n`` and ``chain`` is only
    the newly generated suffix.  Therefore Venhoff's slice of the *full decoded
    response* starts at suffix offset zero and ends at the suffix's first
    ``</think>`` (or completion end when unclosed).  A generated/nested
    ``<think>`` is ordinary reasoning text and must not reset the start.
    """
    if not isinstance(chain, str):
        raise ValueError("chain must be a string")
    start = 0
    close = chain.find("</think>")
    end = close if close >= 0 else len(chain)
    raw = chain[start:end]
    left_trim = len(raw) - len(raw.lstrip())
    stripped = raw.strip()
    start += left_trim
    end = start + len(stripped)
    return chain[start:end], start, end


def validate_generation_contract(eval_dir: Path) -> tuple[list[dict], dict, dict]:
    """Load and strictly validate the complete all-four bridge generation.

    Annotation is refused until there are exactly 50 shared baselines and 50
    matched records for each of the four behaviours.  Pilot/subset generations
    therefore cannot accidentally consume the full-run annotation budget.
    """
    eval_dir = Path(eval_dir)
    results_path = eval_dir / "steering_results.json"
    provenance_path = eval_dir / "provenance.json"
    ids_path = eval_dir / "eval_task_ids.json"
    for path in (results_path, provenance_path, ids_path):
        if not path.is_file():
            raise FileNotFoundError(f"required bridge artefact is missing: {path}")

    rows = _load_json(results_path)
    provenance = _load_json(provenance_path)
    split = _load_json(ids_path)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("steering_results.json must be a list of objects")
    if not isinstance(provenance, dict) or not isinstance(split, dict):
        raise ValueError("bridge provenance/eval split must be JSON objects")

    task_ids = split.get("task_ids")
    if (
        not isinstance(task_ids, list)
        or len(task_ids) != EXPECTED_TASKS
        or len(set(task_ids)) != EXPECTED_TASKS
        or not all(isinstance(task_id, str) and task_id for task_id in task_ids)
    ):
        raise ValueError(
            f"bridge annotation requires exactly {EXPECTED_TASKS} unique eval task IDs"
        )
    if len(rows) != EXPECTED_RECORDS:
        raise ValueError(
            f"bridge annotation requires exactly {EXPECTED_RECORDS} records "
            f"(50 shared + four x 50); found {len(rows)}"
        )

    # Resolve the corpus from the repository, not from the provenance's
    # user-editable path.  The file hash, ID order, prompt text and category are
    # all part of the paid-run contract.
    repository_root = Path(__file__).resolve().parents[1]
    canonical_tasks_path = repository_root / "data" / "tasks_final.json"
    canonical_ids_path = (
        repository_root / "results" / "eval" / "R1-1.5B__E1"
        / "eval_task_ids.json"
    )
    if sha256_file(canonical_tasks_path) != EXPECTED_TASKS_SHA256:
        raise ValueError("local frozen task corpus hash has changed")
    if sha256_file(canonical_ids_path) != EXPECTED_EVAL_IDS_SHA256:
        raise ValueError("local canonical E1 split hash has changed")
    canonical_tasks = _load_json(canonical_tasks_path)
    canonical_split = _load_json(canonical_ids_path)
    if not isinstance(canonical_tasks, list) or not isinstance(canonical_split, dict):
        raise ValueError("canonical task/split assets have invalid JSON structure")
    canonical_by_id = {
        task.get("id"): task for task in canonical_tasks if isinstance(task, dict)
    }
    if canonical_split.get("task_ids") != list(EXPECTED_TASK_IDS):
        raise ValueError("canonical E1 source no longer matches the frozen ID list")

    run_contract = provenance.get("run_contract")
    run_sha = provenance.get("run_contract_sha256")
    if not isinstance(run_contract, dict) or not isinstance(run_sha, str):
        raise ValueError("bridge provenance lacks its run contract/hash")
    if _json_sha256(run_contract) != run_sha:
        raise ValueError("bridge run_contract_sha256 does not match run_contract")
    if (
        split.get("rule") != canonical_split.get("rule")
        or split.get("category_counts") != canonical_split.get("category_counts")
        or split.get("run_contract_sha256") != run_sha
    ):
        raise ValueError("bridge eval_task_ids metadata differs from canonical E1/run")
    if task_ids != list(EXPECTED_TASK_IDS):
        raise ValueError("eval task IDs are not the frozen canonical E1 hold-out")
    if _json_sha256(task_ids) != EXPECTED_EVAL_TASK_ID_LIST_SHA256:
        raise RuntimeError("internal canonical E1 task-list digest is inconsistent")
    if run_contract.get("behaviours") != list(TARGET_BEHAVIOURS):
        raise ValueError("bridge run contract is not the all-four behaviour run")
    if run_contract.get("mode") != "constant_subtract":
        raise ValueError("bridge run contract is not constant subtraction")
    if run_contract.get("operator") != "h <- h - alpha * v":
        raise ValueError("bridge run contract has the wrong intervention operator")
    if not _alpha_equal(run_contract.get("alpha"), EXPECTED_ALPHA):
        raise ValueError("bridge annotation requires alpha=1")
    if int(run_contract.get("max_new_tokens", -1)) != EXPECTED_MAX_NEW_TOKENS:
        raise ValueError("bridge annotation requires the published 1000-token cap")
    if run_contract.get("evaluation_task_ids") != task_ids:
        raise ValueError("provenance and eval_task_ids.json disagree on task order")
    if run_contract.get("evaluation_split") != canonical_split.get("rule"):
        raise ValueError("bridge run has the wrong canonical E1 split rule")
    model = run_contract.get("model")
    if not isinstance(model, dict) or {
        "public_id": model.get("public_id"),
        "revision": model.get("revision"),
        "model_weight_sha256": model.get("model_weight_sha256"),
    } != {
        "public_id": EXPECTED_MODEL_ID,
        "revision": EXPECTED_MODEL_REVISION,
        "model_weight_sha256": EXPECTED_MODEL_WEIGHT_SHA256,
    }:
        raise ValueError("bridge run is not the pinned audited R1-Distill-Qwen-1.5B")
    if run_contract.get("dtype") != "bfloat16" or run_contract.get("use_4bit") is not False:
        raise ValueError("bridge annotation requires bfloat16 and no quantisation")
    if run_contract.get("temperature") != 0.0:
        raise ValueError("bridge annotation requires greedy temperature=0 generation")
    if run_contract.get("layers") != EXPECTED_LAYERS:
        raise ValueError("bridge run does not use all four exact Venhoff layers")
    if run_contract.get("arms") != [
        "shared vanilla baseline", "venhoff_constant_subtract"
    ]:
        raise ValueError("bridge run has an unexpected arm family")
    ids_source = run_contract.get("evaluation_ids_source")
    tasks_source = run_contract.get("tasks_source")
    if not isinstance(ids_source, dict) or ids_source.get("sha256") != EXPECTED_EVAL_IDS_SHA256:
        raise ValueError("bridge run does not pin the canonical E1 split artefact")
    if not isinstance(tasks_source, dict) or tasks_source.get("sha256") != EXPECTED_TASKS_SHA256:
        raise ValueError("bridge run does not pin the frozen task corpus")
    vector_source = run_contract.get("vector_source")
    _validate_hybrid_vector_contract(vector_source, repository_root)

    keys = [_record_key(row) for row in rows]
    if len(set(keys)) != len(keys):
        duplicates = [key for key, n in Counter(keys).items() if n > 1]
        raise ValueError(f"duplicate bridge generation keys: {duplicates[:3]}")
    if any(row.get("run_contract_sha256") != run_sha for row in rows):
        raise ValueError("one or more generation records have a foreign run contract")
    if any(row.get("task_id") not in set(task_ids) for row in rows):
        raise ValueError("generation contains a task outside eval_task_ids.json")
    for row in rows:
        if row.get("temperature") != 0.0 or row.get("seed") != 42:
            raise ValueError("generation row is not the frozen greedy seed-42 record")
        task_id = row.get("task_id")
        canonical_task = canonical_by_id.get(task_id)
        if not isinstance(canonical_task, dict):
            raise ValueError(f"generation task is absent from frozen corpus: {task_id}")
        if row.get("base_task_id") != task_id:
            raise ValueError("bridge pairing requires base_task_id == task_id")
        if row.get("instruction") != canonical_task.get("prompt"):
            raise ValueError(f"generation instruction differs from corpus: {task_id}")
        if row.get("category") != canonical_task.get("category"):
            raise ValueError(f"generation category differs from corpus: {task_id}")
        expected_prompt = (
            EXPECTED_PROMPT_PREFIX + canonical_task["prompt"] + EXPECTED_PROMPT_SUFFIX
        )
        if row.get("prompt") != expected_prompt:
            raise ValueError(f"generation prompt differs from pinned chat template: {task_id}")
        if not isinstance(row.get("chain"), str) or not isinstance(row.get("instruction"), str):
            raise ValueError("generation row lacks its completion or instruction text")
        reasoning, reasoning_start, reasoning_end = venhoff_reasoning_slice(row["chain"])
        if not reasoning or reasoning_end < reasoning_start:
            raise ValueError(f"generation has empty Venhoff reasoning region: {task_id}")
        if not isinstance(row.get("prompt"), str) or row.get("full_text") != row["prompt"] + row["chain"]:
            raise ValueError("generation row does not preserve exact prompt+completion text")
        if row.get("method") == BRIDGE_METHOD:
            behaviour = row.get("behaviour")
            if (
                row.get("mode") != "constant_subtract"
                or row.get("layer") != EXPECTED_LAYERS.get(behaviour)
                or row.get("vector_source") != "user_direction_published_norm"
                or row.get("source_vector_sha256")
                != EXPECTED_UNIT_VECTOR_SHA256.get(behaviour)
                or row.get("derived_vector_sha256")
                != EXPECTED_SCALED_VECTOR_SHA256.get(behaviour)
                or not math.isclose(
                    float(row.get("vector_norm", math.nan)),
                    EXPECTED_VECTOR_NORMS.get(behaviour, math.nan),
                    rel_tol=1e-5,
                    abs_tol=1e-5,
                )
                or not math.isclose(
                    float(row.get("write_norm", math.nan)),
                    EXPECTED_VECTOR_NORMS.get(behaviour, math.nan),
                    rel_tol=1e-5,
                    abs_tol=1e-5,
                )
            ):
                raise ValueError(f"generation row violates fixed bridge contract: {behaviour}")
        elif row.get("method") == VANILLA_METHOD:
            if (
                row.get("behaviour") != SHARED_BEHAVIOUR
                or row.get("mode") != "none"
                or row.get("layer") is not None
                or row.get("vector_source") is not None
                or not _alpha_equal(row.get("alpha"), 0.0)
                or not _alpha_equal(row.get("vector_norm"), 0.0)
                or not _alpha_equal(row.get("write_norm"), 0.0)
            ):
                raise ValueError("shared baseline row violates the frozen vanilla contract")
        else:
            raise ValueError(f"generation row has an unexpected method: {row.get('method')}")

    by_cell: dict[tuple[str, str, float], list[dict]] = {}
    for row in rows:
        cell = (
            str(row.get("behaviour")),
            str(row.get("method")),
            float(row.get("alpha", math.nan)),
        )
        by_cell.setdefault(cell, []).append(row)
    expected_cells = {(SHARED_BEHAVIOUR, VANILLA_METHOD, 0.0)} | {
        (behaviour, BRIDGE_METHOD, EXPECTED_ALPHA)
        for behaviour in TARGET_BEHAVIOURS
    }
    if set(by_cell) != expected_cells:
        raise ValueError(
            "bridge generation cells differ from shared vanilla + four fixed arms: "
            f"found={sorted(by_cell)}"
        )
    for cell, cell_rows in by_cell.items():
        cell_ids = [row["task_id"] for row in cell_rows]
        if len(cell_rows) != EXPECTED_TASKS or set(cell_ids) != set(task_ids):
            raise ValueError(f"bridge cell {cell} is not complete over all 50 tasks")

    # Canonical plan order: shared baseline once, then the four behaviour arms;
    # within each cell use the frozen eval-task order.
    index = {
        (row["behaviour"], row["method"], float(row["alpha"]), row["task_id"]): row
        for row in rows
    }
    ordered = [
        index[(SHARED_BEHAVIOUR, VANILLA_METHOD, 0.0, task_id)]
        for task_id in task_ids
    ]
    for behaviour in TARGET_BEHAVIOURS:
        ordered.extend(
            index[(behaviour, BRIDGE_METHOD, EXPECTED_ALPHA, task_id)]
            for task_id in task_ids
        )
    return ordered, provenance, split


def _annotation_region_text(row: dict) -> str:
    chain = row.get("chain")
    if not isinstance(chain, str):
        raise ValueError(f"generation row {row.get('task_id')!r} has no string chain")
    reasoning, _, _ = venhoff_reasoning_slice(chain)
    return reasoning


def _request_plan(rows: Sequence[dict]) -> tuple[list[dict], int]:
    plan_rows: list[dict] = []
    total_requests = 0
    for row in rows:
        region = _annotation_region_text(row)
        requests = annotation_initial_request_count(region)
        total_requests += requests
        plan_rows.append(
            {
                "task_id": row["task_id"],
                "behaviour": row["behaviour"],
                "method": row["method"],
                "alpha": row["alpha"],
                "run_contract_sha256": row["run_contract_sha256"],
                "chain_sha256": hashlib.sha256(row["chain"].encode()).hexdigest(),
                "annotation_region_sha256": hashlib.sha256(region.encode()).hexdigest(),
                "annotation_region_chars": len(region),
                "planned_initial_requests": requests,
            }
        )
    return plan_rows, total_requests


def _estimated_cost_range(plan_rows: Sequence[dict], planned_requests: int) -> dict:
    """Transparent planning estimate; the journal, not this estimate, is the cap.

    A labelled echo is commonly about 1.0--1.6 times the source-region token
    count once delimiters are included.  Input uses the ordinary 4-char/token
    planning approximation.  The upper number is still only an estimate; the
    durable guard reserves the stricter $0.05 theoretical maximum before every
    call and reconciles against proxy-reported actual cost afterwards.
    """
    body_tokens = sum(max(0, int(row["annotation_region_chars"]) // 4) for row in plan_rows)
    # Approximate one 650-token instruction envelope per request.  The precise
    # prompt-size *safety* bound is handled independently by max_prompt_chars().
    input_tokens = body_tokens + planned_requests * 650
    low_output = body_tokens * 1.0
    high_output = body_tokens * 1.6
    low = input_tokens * 3.0e-6 + low_output * 15.0e-6
    high = input_tokens * 3.0e-6 + high_output * 15.0e-6
    return {
        "method": "source echo 1.0x--1.6x plus ~650 input tokens/request",
        "estimated_low_usd": round(float(low), 4),
        "estimated_high_usd": round(float(high), 4),
        "hard_ceiling_usd": SPEND_CEILING_USD,
        "note": (
            "planning range only; the attempt journal enforces the hard ceiling "
            "from reported cost and conservative unresolved-attempt commitments"
        ),
    }


def prepare_annotation_manifest(
    eval_dir: Path,
    *,
    root: Path,
    bound_paths: Sequence[str] = DEFAULT_BOUND_PATHS,
) -> dict:
    """Write the immutable annotation plan and $15 guard manifest, API-free.

    Existing manifests are never replaced.  An identical re-run simply returns
    the existing digest; a changed plan/code contract requires a fresh output
    directory so a prior attempt journal cannot be reset by manifest rotation.
    """
    root = Path(root).resolve()
    eval_dir = Path(eval_dir).resolve()
    rows, provenance, split = validate_generation_contract(eval_dir)
    plan_rows, planned_requests = _request_plan(rows)
    if planned_requests < 1:
        raise ValueError("the bridge annotation plan contains no API requests")

    plan = {
        "schema": PLAN_SCHEMA,
        "record_count": len(rows),
        "task_count": len(split["task_ids"]),
        "behaviours": list(TARGET_BEHAVIOURS),
        "cells": {
            "shared_vanilla": EXPECTED_TASKS,
            **{behaviour: EXPECTED_TASKS for behaviour in TARGET_BEHAVIOURS},
        },
        "annotation": {
            "model": ANNOTATION_MODEL,
            "prompt": "src.annotation current Venhoff-schema prompt",
            "prompt_version": ANNOTATION_PROMPT_VERSION,
            "coverage_rule_version": COVERAGE_RULE_VERSION,
            **VENHOFF_ANNOTATION_REGION_POLICY,
            "region_definition": (
                "strip whitespace around text after <think> and before the first "
                "</think>, or completion end when unclosed; no other cuts"
            ),
            "max_output_tokens": ANNOTATION_MAX_OUTPUT_TOKENS,
            "max_retries_per_chunk": ANNOTATION_MAX_RETRIES,
            "planned_initial_requests": planned_requests,
        },
        "run_contract_sha256": provenance["run_contract_sha256"],
        "rows": plan_rows,
    }
    plan_path = eval_dir / "annotation_plan.json"
    manifest_path = eval_dir / "annotation_guard_manifest.json"
    journal_path = root / GLOBAL_JOURNAL_RELATIVE

    # A pre-existing global journal is expected on resume (including from a
    # copied eval directory).  Its manifest binding is checked fail-closed by
    # AnnotationAttemptGuard before any reservation.
    if plan_path.exists():
        existing_plan = _load_json(plan_path)
        if existing_plan != plan:
            raise ValueError(
                "existing annotation_plan.json differs; use a fresh eval directory"
            )
    else:
        atomic_json(plan, plan_path)

    source_paths = (
        eval_dir / "steering_results.json",
        eval_dir / "provenance.json",
        eval_dir / "eval_task_ids.json",
        plan_path,
    )
    source_artifacts = [
        {"path": _repo_relative(path, root), "sha256": sha256_file(path)}
        for path in source_paths
    ]
    bound_artifacts = []
    for rel in bound_paths:
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError(f"bound annotation code is missing: {path}")
        bound_artifacts.append({"path": rel, "sha256": sha256_file(path)})

    # One retry per deterministic chunk scope.  The static maximum count can
    # exceed $15; the journal admits the next request only while actual spend
    # plus a conservative $0.05 reservation remains within $15.
    retry_capacity = planned_requests
    manifest = {
        "schema": ATTEMPT_GUARD_SCHEMA,
        "status": "authorised",
        "model": ANNOTATION_MODEL,
        "annotation_window_tokens": ANNOTATION_WINDOW_TOKENS,
        "guards": {
            "planned_initial_requests": planned_requests,
            "global_retry_capacity": retry_capacity,
            "max_total_attempts": planned_requests + retry_capacity,
            "max_attempts_per_scope": ANNOTATION_MAX_RETRIES,
            "approved_spend_ceiling_usd": SPEND_CEILING_USD,
            "max_cost_per_attempt_usd": MAX_COST_PER_ATTEMPT_USD,
            "remaining_quota_floor_usd": 0.0,
            "max_output_tokens": ANNOTATION_MAX_OUTPUT_TOKENS,
            "max_prompt_chars": max_prompt_chars(),
            "prior_committed_spend_usd": 0.0,
        },
        "rate_card": RATE_CARD,
        "coverage_rule_version": COVERAGE_RULE_VERSION,
        "source_artifacts": source_artifacts,
        "bound_artifacts": bound_artifacts,
        "cost_estimate": _estimated_cost_range(plan_rows, planned_requests),
        "scientific_contract": {
            "records": EXPECTED_RECORDS,
            "shared_baseline_records": EXPECTED_TASKS,
            "steered_records": EXPECTED_TASKS * len(TARGET_BEHAVIOURS),
            "primary_estimand": (
                "target-labelled Qwen tokens / tokens labelled as any of the "
                "four steered behaviours, before </think>"
            ),
            "annotation_region_policy": VENHOFF_ANNOTATION_REGION_POLICY,
            "sensitivities": [
                "released first-find/silent-skip alignment arithmetic",
                "target-labelled Qwen tokens / all six labelled Qwen tokens",
                "target-labelled spans / all six annotated spans",
            ],
            "missingness": "unresolved; never scored as zero",
            "baseline_annotation_convention": (
                "50 shared baseline generations annotated once; unlike the "
                "released loop, no fourfold stochastic baseline re-annotation"
            ),
        },
    }

    if manifest_path.exists():
        existing = _load_json(manifest_path)
        if existing != manifest:
            raise ValueError(
                "existing annotation guard differs; refusing to replace a spend manifest"
            )
    else:
        atomic_json(manifest, manifest_path)

    return {
        "plan_path": str(plan_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "journal_path": str(journal_path),
        "record_count": len(rows),
        "planned_initial_requests": planned_requests,
        "cost_estimate": manifest["cost_estimate"],
    }


def _assert_bridge_guard_policy(guard: AnnotationAttemptGuard) -> None:
    """Pin all spend-relevant manifest fields again in executing code."""
    policy = guard.policy
    # The CLI's literal ``--confirm-spend-ceiling-usd 15`` is not enough on
    # its own: a hand-edited manifest with a fresh digest must still be unable
    # to widen any monetary/rate-card boundary.  Pin the bridge policy in code
    # as well as in the signed-by-hash document.
    if abs(policy.approved_spend_ceiling_usd - SPEND_CEILING_USD) > 1e-12:
        raise AnnotationAttemptLimitError(
            "bridge guard spend ceiling is not exactly USD 15; no call made"
        )
    if abs(policy.max_cost_per_attempt_usd - MAX_COST_PER_ATTEMPT_USD) > 1e-12:
        raise AnnotationAttemptLimitError(
            "bridge guard per-attempt ceiling differs from USD 0.05; no call made"
        )
    actual_rate_card = {
        "input_usd_per_mtok": policy.rate_card.input_usd_per_mtok,
        "output_usd_per_mtok": policy.rate_card.output_usd_per_mtok,
        "tokens_per_char_upper": policy.rate_card.tokens_per_char_upper,
    }
    if actual_rate_card != RATE_CARD:
        raise AnnotationAttemptLimitError(
            "bridge guard rate card differs from the pinned conservative card; no call made"
        )
    if (
        policy.max_attempts_per_scope != ANNOTATION_MAX_RETRIES
        or policy.global_retry_capacity != policy.planned_initial_requests
        or policy.max_total_attempts != 2 * policy.planned_initial_requests
    ):
        raise AnnotationAttemptLimitError(
            "bridge retry policy differs from one retry per planned chunk; no call made"
        )


def _load_guard(eval_dir: Path, root: Path, manifest_sha256: str) -> AnnotationAttemptGuard:
    eval_dir = Path(eval_dir).resolve()
    root = Path(root).resolve()
    manifest_path = eval_dir / "annotation_guard_manifest.json"
    expected_bound_paths = tuple(DEFAULT_BOUND_PATHS)
    guard = AnnotationAttemptGuard.from_manifest(
        manifest_path,
        manifest_sha256,
        root / GLOBAL_JOURNAL_RELATIVE,
        root=root,
        expected_model=ANNOTATION_MODEL,
        expected_annotation_window_tokens=ANNOTATION_WINDOW_TOKENS,
        expected_bound_paths=expected_bound_paths,
        expected_max_output_tokens=ANNOTATION_MAX_OUTPUT_TOKENS,
        expected_max_prompt_chars=max_prompt_chars(),
        expected_coverage_rule_version=COVERAGE_RULE_VERSION,
    )
    _assert_bridge_guard_policy(guard)
    return guard


def run_annotation(
    eval_dir: Path,
    *,
    root: Path,
    manifest_sha256: str,
) -> dict:
    """Run/resume the paid annotation stage under the immutable $15 guard."""
    eval_dir = Path(eval_dir).resolve()
    root = Path(root).resolve()
    rows, _, _ = validate_generation_contract(eval_dir)
    guard = _load_guard(eval_dir, root, manifest_sha256)

    expected_sources = {
        _repo_relative(eval_dir / "steering_results.json", root),
        _repo_relative(eval_dir / "provenance.json", root),
        _repo_relative(eval_dir / "eval_task_ids.json", root),
        _repo_relative(eval_dir / "annotation_plan.json", root),
    }
    if set(guard.policy.source_paths) != expected_sources:
        raise AnnotationAttemptLimitError(
            "annotation source-set mismatch; refusing all API calls"
        )
    observed_requests = sum(
        annotation_initial_request_count(_annotation_region_text(row)) for row in rows
    )
    guard.assert_planned_initial_requests(observed_requests)

    # The released evaluator annotates exactly the stripped reasoning substring,
    # not the final response and not the current thesis's degeneration-trimmed
    # region.  Preserve the original generated completion alongside the exact
    # paid request text so later token scoring remains auditable.
    annotation_rows = []
    for row in rows:
        reasoning, start, end = venhoff_reasoning_slice(row["chain"])
        annotation_rows.append({
            **row,
            "chain_full": row["chain"],
            "chain": reasoning,
            "venhoff_reasoning_start": start,
            "venhoff_reasoning_end": end,
            "reasoning_open_marker_present": "<think>" in row["prompt"],
            "generated_nested_open_marker_present": "<think>" in row["chain"],
            "reasoning_close_marker_present": "</think>" in row["chain"],
        })

    annotations_path = eval_dir / "annotated_venhoff_bridge.json"
    try:
        annotated = annotate_chains(
            annotation_rows,
            save_path=annotations_path,
            checkpoint_every=1,
            dedup_keys=DEDUP_KEYS,
            model=ANNOTATION_MODEL,
            max_tokens=ANNOTATION_MAX_OUTPUT_TOKENS,
            shrink_on_retry=True,
            max_retries=ANNOTATION_MAX_RETRIES,
            attempt_guard=guard,
            coverage_validation=True,
            include_post_think=False,
            exclude_final_answer_suffix=False,
            exclude_degenerate_repetition=False,
            strict_resume_scope=True,
        )
    except AnnotationAttemptLimitError:
        # ``annotate_chains`` atomically saves completed work before re-raising
        # a budget refusal.  Materialise the partial status as well, then keep
        # the refusal terminal so no caller can mistake attrition for success.
        annotated = (
            _load_json(annotations_path) if annotations_path.is_file() else []
        )
        status = annotation_status(rows, annotated)
        status["budget"] = guard.summary()
        status["manifest_sha256"] = manifest_sha256
        status["stage_complete"] = False
        atomic_json(status, eval_dir / "annotation_status.json")
        raise
    status = annotation_status(rows, annotated)
    status["budget"] = guard.summary()
    status["manifest_sha256"] = manifest_sha256
    status["stage_complete"] = status["unresolved_records"] == 0
    atomic_json(status, eval_dir / "annotation_status.json")
    return status


def _annotation_index(annotated: Sequence[dict]) -> dict[tuple, dict]:
    index: dict[tuple, dict] = {}
    for row in annotated:
        if not isinstance(row, dict):
            raise ValueError("annotation checkpoint contains a non-object row")
        key = _record_key(row)
        if key in index:
            raise ValueError(f"duplicate annotation key: {key}")
        index[key] = row
    return index


def _has_exact_venhoff_region_policy(row: dict) -> bool:
    return row.get("annotation_region_policy") == VENHOFF_ANNOTATION_REGION_POLICY


def annotation_status(generation: Sequence[dict], annotated: Sequence[dict]) -> dict:
    index = _annotation_index(annotated)
    reasons: Counter = Counter()
    complete = 0
    for source in generation:
        row = index.get(_record_key(source))
        if row is None:
            reasons["missing_annotation_record"] += 1
        elif not bool(row.get("annotation_complete")):
            reasons["annotation_incomplete"] += 1
        elif not _has_exact_venhoff_region_policy(row):
            reasons["annotation_region_policy_mismatch"] += 1
        elif not row_is_coverage_complete(row):
            reasons["coverage_incomplete"] += 1
        elif not row.get("annotations"):
            reasons["empty_annotations"] += 1
        else:
            complete += 1
    return {
        "planned_records": len(generation),
        "checkpoint_records": len(annotated),
        "complete_scored_candidates": complete,
        "unresolved_records": len(generation) - complete,
        "unresolved_by_reason": dict(sorted(reasons.items())),
        "missing_is_zero": False,
        "coverage_rule_version": COVERAGE_RULE_VERSION,
    }


def load_qwen_tokenizer(provenance: dict, tokenizer_path: Path | None = None):
    """Load only the pinned bridge tokenizer (never the model weights)."""
    from transformers import AutoTokenizer

    model_contract = provenance.get("run_contract", {}).get("model", {})
    source = Path(tokenizer_path or model_contract.get("local_snapshot", ""))
    if not source.is_dir():
        raise FileNotFoundError(
            "pinned Qwen tokenizer snapshot is unavailable; pass --tokenizer-path"
        )
    tokenizer_json = source / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise FileNotFoundError(f"tokenizer.json is missing: {tokenizer_json}")
    tokenizer_sha256 = sha256_file(tokenizer_json)
    if tokenizer_sha256 != EXPECTED_TOKENIZER_JSON_SHA256:
        raise ValueError(
            "tokenizer.json is not the audited Qwen-1.5B tokenizer; "
            "token-count evaluation refused"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        source, local_files_only=True, use_fast=True
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Venhoff token-position scoring requires a fast tokenizer")
    return tokenizer, {
        "path": str(source.resolve()),
        "tokenizer_json_sha256": tokenizer_sha256,
        "model_revision": EXPECTED_MODEL_REVISION,
    }


def _ordered_span_ranges(source: str, spans: Sequence[dict]) -> list[tuple[int, int]]:
    cursor = 0
    ranges: list[tuple[int, int]] = []
    for span in spans:
        text = span["text"]
        start = source.find(text, cursor)
        if start < 0:
            raise ValueError(
                f"annotation span cannot be aligned after character {cursor}: "
                f"{text[:80]!r}"
            )
        end = start + len(text)
        ranges.append((start, end))
        cursor = end
    return ranges


def _token_offsets(tokenizer, text: str) -> list[tuple[int, int]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]
    # Batch-shaped tokenizers are not expected here, but fail explicitly rather
    # than silently interpreting the first batch dimension as token offsets.
    if offsets and isinstance(offsets[0], (list, tuple)) and len(offsets[0]) != 2:
        raise ValueError("tokenizer returned batched/invalid offset mappings")
    cleaned = [(int(start), int(end)) for start, end in offsets if int(end) > int(start)]
    if text and not cleaned:
        raise ValueError("tokenizer returned no offsets for a non-empty reasoning region")
    return cleaned


def released_annotation_spans(annotated_response: str) -> list[dict]:
    """Parse annotations with Venhoff's released regex, without normalisation.

    This is intentionally *not* the thesis annotation parser.  It exists only
    for replaying the public JSON and documenting the effect of released-code
    quirks.  In particular, unquoted or numbered labels are not recognised.
    """
    if not isinstance(annotated_response, str):
        raise ValueError("released annotated response must be a string")
    spans = []
    for match in _RELEASED_ANNOTATION_RE.finditer(annotated_response):
        text = match.group(2).strip()
        if text:
            spans.append({"label": match.group(1).strip(), "text": text})
    return spans


def released_bug_compatible_token_label_counts(
    response_text: str,
    spans: Sequence[dict],
    tokenizer,
) -> dict:
    """Reproduce released ``get_label_positions/get_label_counts`` arithmetic.

    Every span is located with ``response.find(text)`` from character zero;
    repeated text therefore aliases its first occurrence.  A span with no exact
    match, no mapped boundary character, or an invalid interval is silently
    skipped.  These behaviours are bugs from a coverage perspective, so this is
    an explicitly named sensitivity rather than the primary corrected endpoint.

    Given spans produced by :func:`released_annotation_spans`, this function is
    byte-for-byte compatible with the 600 released Qwen-1.5B evaluation records.
    For new bridge annotations it receives the current strict six-label schema's
    parsed spans because the historical raw annotator response is not retained.
    """
    if not isinstance(response_text, str):
        raise ValueError("released scoring response must be a string")
    if not isinstance(spans, Sequence):
        raise ValueError("released scoring spans must be a sequence")

    if hasattr(tokenizer, "encode_plus"):
        encoded = tokenizer.encode_plus(
            response_text, return_offsets_mapping=True
        )
    else:
        # Test fixtures and tokenizer-compatible adapters may expose only the
        # callable fast-tokenizer interface.  ``add_special_tokens=True`` is the
        # Hugging Face default used by the released ``encode_plus`` call.
        encoded = tokenizer(
            response_text,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise ValueError("tokenizer did not return offset_mapping")
    char_to_token: dict[int, int] = {}
    for token_index, pair in enumerate(offsets):
        if len(pair) != 2:
            raise ValueError("tokenizer returned invalid offset mappings")
        start, end = int(pair[0]), int(pair[1])
        for char_position in range(start, end):
            char_to_token[char_position] = token_index

    counts = {label: 0 for label in VALID_LABELS}
    skipped = Counter()
    skipped_indices: list[int] = []
    mapped_spans = 0
    for index, span in enumerate(spans):
        if not isinstance(span, dict):
            raise ValueError("released scoring received a non-object span")
        label = str(span.get("label", "")).strip()
        text = str(span.get("text", "")).strip()
        if not text:
            skipped["empty_text"] += 1
            skipped_indices.append(index)
            continue
        text_position = response_text.find(text)
        if text_position < 0:
            skipped["text_not_found"] += 1
            skipped_indices.append(index)
            continue
        token_start = char_to_token.get(text_position)
        token_end = char_to_token.get(text_position + len(text) - 1)
        if token_start is None or token_end is None:
            skipped["boundary_unmapped"] += 1
            skipped_indices.append(index)
            continue
        token_end += 1
        if token_start >= token_end:
            skipped["invalid_interval"] += 1
            skipped_indices.append(index)
            continue
        mapped_spans += 1
        if label in counts:
            counts[label] += token_end - token_start

    four_total = sum(counts[label] for label in TARGET_BEHAVIOURS)
    six_total = sum(counts.values())
    return {
        "released_bug_compatible_label_token_counts": counts,
        "released_bug_compatible_four_target_token_total": four_total,
        "released_bug_compatible_six_label_token_total": six_total,
        "released_bug_compatible_mapped_spans": mapped_spans,
        "released_bug_compatible_skipped_spans": int(sum(skipped.values())),
        "released_bug_compatible_skipped_by_reason": dict(sorted(skipped.items())),
        "released_bug_compatible_skipped_span_indices": skipped_indices,
        "released_bug_compatible_four_target_denominator_zero": four_total == 0,
        RELEASED_BUG_COMPATIBLE_METRIC: {
            label: (counts[label] / four_total if four_total > 0 else 0.0)
            for label in TARGET_BEHAVIOURS
        },
        "venhoff_released_bug_compatible_six_label_token_fraction": {
            label: (counts[label] / six_total if six_total > 0 else 0.0)
            for label in TARGET_BEHAVIOURS
        },
    }


def audit_released_qwen15_results(results: dict, tokenizer) -> dict:
    """Replay all public Qwen-1.5B fractions and quantify exact agreement."""
    if not isinstance(results, dict):
        raise ValueError("released evaluation results must be an object")
    differences: list[float] = []
    records = 0
    skipped_spans = 0
    for target in TARGET_BEHAVIOURS:
        examples = results.get(target)
        if not isinstance(examples, list):
            raise ValueError(f"released results lack list for {target}")
        for example in examples:
            if not isinstance(example, dict):
                raise ValueError("released example must be an object")
            for arm in ("original", "positive", "negative"):
                record = example.get(arm)
                if not isinstance(record, dict):
                    raise ValueError(f"released example lacks {arm} record")
                spans = released_annotation_spans(record.get("annotated_response", ""))
                replay = released_bug_compatible_token_label_counts(
                    record.get("response"), spans, tokenizer
                )
                expected = record.get("label_fractions")
                if not isinstance(expected, dict):
                    raise ValueError("released record lacks label_fractions")
                for label in TARGET_BEHAVIOURS:
                    differences.append(abs(
                        float(replay[RELEASED_BUG_COMPATIBLE_METRIC][label])
                        - float(expected[label])
                    ))
                records += 1
                skipped_spans += replay["released_bug_compatible_skipped_spans"]
    return {
        "records": records,
        "label_fraction_comparisons": len(differences),
        "max_absolute_difference": max(differences, default=0.0),
        "mean_absolute_difference": float(np.mean(differences)) if differences else 0.0,
        "exact_comparisons": sum(value == 0.0 for value in differences),
        "released_silently_skipped_spans": skipped_spans,
    }


def token_label_counts(
    chain: str,
    spans: Sequence[dict],
    tokenizer,
    *,
    full_text: str | None = None,
) -> dict:
    """Map ordered spans to Qwen token intervals and count tokens per label.

    Raw-source alignment is attempted first, matching Venhoff's evaluator.  If
    the current annotator has made only a coverage-normalised typography or
    spacing change, both source and spans are normalised and scored there.  A
    row that cannot be aligned in either space is unresolved; no partial count
    is returned.
    """
    if not spans:
        raise ValueError("cannot token-score an empty annotation")
    if any(
        not isinstance(span, dict)
        or span.get("label") not in VALID_LABELS
        or not isinstance(span.get("text"), str)
        or not span["text"].strip()
        for span in spans
    ):
        raise ValueError("annotation contains an invalid label or empty span")

    raw_source, reasoning_start, reasoning_end = venhoff_reasoning_slice(chain)
    raw_texts = [span["text"] for span in spans]
    try:
        region_ranges = _ordered_span_ranges(raw_source, [
            {"text": text} for text in raw_texts
        ])
        source = raw_source
        alignment_space = "raw_pre_think_annotation_region"
        tokenization_context = "completion_only"
        ranges = region_ranges
        region_start = 0
        region_end = len(raw_source)
        # Venhoff's released evaluator tokenizes the complete decoded
        # prompt+response, then maps reasoning spans into it.  Use that context
        # when 07f persisted it (or the caller reconstructed it).  rfind binds
        # to the completion if identical instruction text appears in the prompt.
        if isinstance(full_text, str) and full_text:
            chain_start = full_text.rfind(chain)
            if chain_start >= 0:
                source = full_text
                ranges = [
                    (chain_start + reasoning_start + start,
                     chain_start + reasoning_start + end)
                    for start, end in region_ranges
                ]
                region_start = chain_start + reasoning_start
                region_end = chain_start + reasoning_end
                tokenization_context = "full_prompt_plus_completion"
    except ValueError:
        source = normalise(raw_source)
        normalised_spans = [normalise(text) for text in raw_texts]
        ranges = _ordered_span_ranges(source, [
            {"text": text} for text in normalised_spans
        ])
        alignment_space = "normalised_pre_think_annotation_region"
        tokenization_context = "normalised_completion_only"
        region_start = 0
        region_end = len(source)

    offsets = _token_offsets(tokenizer, source)
    counts = {label: 0 for label in VALID_LABELS}
    for span, (start, end) in zip(spans, ranges):
        # This mirrors the released interval convention: token_start is the
        # token containing the first span character and token_end is one past
        # the token containing the last character.  Boundary-spanning tokens
        # can consequently be assigned to both adjacent spans, as in the
        # released code's summed (end-start) intervals.
        n_tokens = sum(
            1 for token_start, token_end in offsets
            if token_end > start and token_start < end
        )
        if n_tokens <= 0:
            raise ValueError(f"span mapped to no Qwen token: {span['text'][:80]!r}")
        counts[span["label"]] += n_tokens

    four_total = sum(counts[label] for label in TARGET_BEHAVIOURS)
    six_total = sum(counts.values())
    if six_total <= 0:
        raise ValueError("all-six labelled token denominator is zero")
    span_counts = Counter(span["label"] for span in spans)
    return {
        "alignment_space": alignment_space,
        "tokenization_context": tokenization_context,
        "reasoning_open_marker_present": bool(
            isinstance(full_text, str) and "<think>" in full_text
        ),
        "generated_nested_open_marker_present": "<think>" in chain,
        "reasoning_close_marker_present": "</think>" in chain,
        "annotation_region_chars": region_end - region_start,
        "annotation_region_qwen_tokens": sum(
            1 for token_start, token_end in offsets
            if token_end > region_start and token_start < region_end
        ),
        "tokenization_context_chars": len(source),
        "tokenization_context_qwen_tokens": len(offsets),
        "label_token_counts": counts,
        "four_target_token_total": four_total,
        "six_label_token_total": six_total,
        # The released evaluator explicitly returns zero for every label when
        # no target-labelled token exists.  This is a resolved 0/0 convention,
        # not an annotation failure; the denominator-zero flag remains visible.
        "four_target_denominator_zero": four_total == 0,
        "venhoff_four_target_token_fraction": {
            label: (counts[label] / four_total if four_total > 0 else 0.0)
            for label in TARGET_BEHAVIOURS
        },
        "six_label_token_fraction": {
            label: counts[label] / six_total for label in TARGET_BEHAVIOURS
        },
        "six_label_sentence_fraction": {
            label: span_counts[label] / len(spans) for label in TARGET_BEHAVIOURS
        },
    }


def score_records(
    generation: Sequence[dict],
    annotated: Sequence[dict],
    tokenizer,
) -> list[dict]:
    """Score every planned generation record, retaining unresolved rows."""
    index = _annotation_index(annotated)
    scored: list[dict] = []
    for source in generation:
        identity = {
            "task_id": source["task_id"],
            "base_task_id": source.get("base_task_id", source["task_id"]),
            "behaviour": source["behaviour"],
            "method": source["method"],
            "alpha": source["alpha"],
            "run_contract_sha256": source["run_contract_sha256"],
        }
        annotation = index.get(_record_key(source))
        if annotation is None:
            scored.append({**identity, "score_status": "unresolved",
                           "unresolved_reason": "missing_annotation_record"})
            continue
        if not bool(annotation.get("annotation_complete")):
            scored.append({**identity, "score_status": "unresolved",
                           "unresolved_reason": "annotation_incomplete"})
            continue
        if not _has_exact_venhoff_region_policy(annotation):
            scored.append({**identity, "score_status": "unresolved",
                           "unresolved_reason": "annotation_region_policy_mismatch"})
            continue
        if not row_is_coverage_complete(annotation):
            scored.append({**identity, "score_status": "unresolved",
                           "unresolved_reason": "coverage_incomplete"})
            continue
        spans = annotation.get("annotations")
        if not spans:
            scored.append({**identity, "score_status": "unresolved",
                           "unresolved_reason": "empty_annotations"})
            continue
        full_text = source.get("full_text")
        tokenization_source = "persisted_full_text"
        if not isinstance(full_text, str) or not full_text:
            prompt = source.get("prompt")
            tokenization_source = "persisted_prompt_plus_chain"
            if not isinstance(prompt, str) or not prompt:
                # 07f versions before the annotation bridge kept the decoded
                # completion but not the formatted prompt.  Reconstruct it from
                # the pinned tokenizer when possible; if a tokenizer fixture or
                # transferred artefact cannot, the completion-only score remains
                # well-defined and the context field discloses the fallback.
                try:
                    from src.chain_gen import format_prompt
                    prompt = format_prompt(tokenizer, source["instruction"])
                    tokenization_source = "reconstructed_prompt_plus_chain"
                except Exception:
                    prompt = None
                    tokenization_source = "completion_only_fallback"
            full_text = (prompt + source["chain"] if isinstance(prompt, str) else None)
        released_response = full_text if isinstance(full_text, str) else source["chain"]
        try:
            released_metrics = released_bug_compatible_token_label_counts(
                released_response, spans, tokenizer
            )
        except Exception as exc:
            # A tokenizer-level failure invalidates both the corrected and the
            # released-code sensitivity; it is never converted to a zero.
            scored.append({
                **identity,
                "score_status": "unresolved",
                "unresolved_reason": "released_tokenization_failed",
                "error_class": type(exc).__name__,
                "error": str(exc)[:300],
            })
            continue
        try:
            metrics = token_label_counts(
                source["chain"], spans, tokenizer, full_text=full_text
            )
        except Exception as exc:
            # Retain the explicitly labelled released-bug-compatible
            # sensitivity even when the corrected occurrence-aware endpoint is
            # withheld.  Metric-specific pairing below will use it only for its
            # own sensitivity family.
            scored.append({
                **identity,
                "score_status": "unresolved",
                "unresolved_reason": "corrected_token_alignment_failed",
                "error_class": type(exc).__name__,
                "error": str(exc)[:300],
                "n_annotations": len(spans),
                "tokenization_source": tokenization_source,
                **released_metrics,
            })
            continue
        scored.append({
            **identity,
            "score_status": "resolved",
            "n_annotations": len(spans),
            "tokenization_source": tokenization_source,
            **released_metrics,
            **metrics,
        })
    return scored


def _metric_value(row: dict, metric: str, behaviour: str) -> float:
    mapping = row.get(metric)
    if not isinstance(mapping, dict) or behaviour not in mapping:
        raise ValueError(f"resolved row lacks {metric}[{behaviour}]")
    return float(mapping[behaviour])


def paired_sign_flip_p(
    values: Sequence[float],
    *,
    n_randomizations: int = 100_000,
    seed: int = 0,
) -> dict:
    """Two-sided paired sign-flip test of a zero mean difference.

    The null assumes the per-task paired differences are exchangeable under a
    sign reversal (equivalently, symmetric about zero).  For <=20 nonzero
    differences every sign assignment is enumerated; larger samples use a
    deterministic Monte Carlo randomization with the +1 correction.  Zeros are
    retained in the reported task count but removed from the sign dimension.
    """
    vals = np.asarray(values, dtype=float)
    if vals.ndim != 1 or not np.all(np.isfinite(vals)):
        raise ValueError("paired sign-flip values must be a finite vector")
    nonzero = vals[np.abs(vals) > 1e-15]
    observed = abs(float(vals.mean())) if vals.size else 0.0
    if nonzero.size == 0:
        return {
            "p_value": 1.0,
            "method": "exact paired sign-flip",
            "assumption": "paired differences are sign-exchangeable under H0",
            "n_tasks": int(vals.size),
            "n_nonzero": 0,
            "n_randomizations": 1,
        }
    # Zeros still dilute the mean in both observed and randomized statistics.
    denominator = vals.size
    tolerance = 1e-15
    if nonzero.size <= 20:
        total = 1 << int(nonzero.size)
        extreme = 0
        for mask in range(total):
            signs = np.fromiter(
                (1.0 if (mask >> bit) & 1 else -1.0
                 for bit in range(nonzero.size)),
                dtype=float,
                count=nonzero.size,
            )
            statistic = abs(float(np.sum(signs * nonzero) / denominator))
            extreme += int(statistic >= observed - tolerance)
        p_value = extreme / total
        method = "exact paired sign-flip"
        draws = total
    else:
        if n_randomizations < 1000:
            raise ValueError("Monte Carlo sign-flip needs >=1000 randomizations")
        rng = np.random.default_rng(seed)
        extreme = 0
        remaining = int(n_randomizations)
        batch_size = 10_000
        while remaining:
            batch = min(batch_size, remaining)
            signs = rng.integers(0, 2, size=(batch, nonzero.size), dtype=np.int8)
            signs = signs.astype(np.float64) * 2.0 - 1.0
            statistics = np.abs((signs @ nonzero) / denominator)
            extreme += int(np.sum(statistics >= observed - tolerance))
            remaining -= batch
        p_value = (extreme + 1) / (int(n_randomizations) + 1)
        method = "Monte Carlo paired sign-flip (+1 correction)"
        draws = int(n_randomizations)
    return {
        "p_value": float(p_value),
        "method": method,
        "assumption": "paired differences are sign-exchangeable under H0",
        "n_tasks": int(vals.size),
        "n_nonzero": int(nonzero.size),
        "n_randomizations": int(draws),
    }


def _paired_metric(
    scored: Sequence[dict],
    behaviour: str,
    metric: str,
    *,
    n_resamples: int,
    seed: int,
) -> tuple[dict, float | None]:
    baseline = {
        row["task_id"]: row for row in scored
        if row["behaviour"] == SHARED_BEHAVIOUR
        and row["method"] == VANILLA_METHOD
    }
    arm = {
        row["task_id"]: row for row in scored
        if row["behaviour"] == behaviour and row["method"] == BRIDGE_METHOD
    }
    expected = sorted(set(baseline) | set(arm))
    task_values = []
    unresolved = []
    for task_id in expected:
        base = baseline.get(task_id)
        steered = arm.get(task_id)
        if base is None or steered is None:
            unresolved.append({
                "task_id": task_id,
                "reason": "missing_scored_cell",
                "baseline_present": base is not None,
                "steered_present": steered is not None,
            })
            continue
        base_mapping = base.get(metric)
        steered_mapping = steered.get(metric)
        if (
            not isinstance(base_mapping, dict)
            or behaviour not in base_mapping
            or not isinstance(steered_mapping, dict)
            or behaviour not in steered_mapping
        ):
            unresolved.append({
                "task_id": task_id,
                "reason": "metric_unresolved",
                "baseline_status": base.get("unresolved_reason", base.get("score_status")),
                "steered_status": steered.get("unresolved_reason", steered.get("score_status")),
            })
            continue
        try:
            base_value = _metric_value(base, metric, behaviour)
            arm_value = _metric_value(steered, metric, behaviour)
        except (TypeError, ValueError) as exc:
            unresolved.append({
                "task_id": task_id,
                "reason": "invalid_metric_value",
                "error": str(exc)[:200],
            })
            continue
        if not math.isfinite(base_value) or not math.isfinite(arm_value):
            unresolved.append({
                "task_id": task_id,
                "reason": "nonfinite_metric_value",
            })
            continue
        task_values.append({
            "task_id": task_id,
            "baseline": base_value,
            "steered": arm_value,
            "suppression_difference": base_value - arm_value,
        })

    values = [row["suppression_difference"] for row in task_values]
    if len(values) > EXPECTED_TASKS:
        raise ValueError(f"{behaviour}/{metric} has more than 50 task pairs")
    n_missing = EXPECTED_TASKS - len(values)
    difference_sum = float(np.sum(values)) if values else 0.0
    worst_case_bounds = {
        "assumed_per_missing_pair_range": [-1.0, 1.0],
        "lower": (difference_sum - n_missing) / EXPECTED_TASKS,
        "upper": (difference_sum + n_missing) / EXPECTED_TASKS,
        "exclude_zero": bool(
            (difference_sum - n_missing) / EXPECTED_TASKS > 0
            or (difference_sum + n_missing) / EXPECTED_TASKS < 0
        ),
    }
    complete = len(values) == EXPECTED_TASKS
    cell = {
        "estimand": "mean paired (shared baseline - negative-steered) fraction",
        "positive_direction": "target behaviour suppressed by steering",
        "n_expected": EXPECTED_TASKS,
        "n_resolved_pairs": len(values),
        "n_unresolved_pairs": n_missing,
        "baseline_mean": (
            float(np.mean([row["baseline"] for row in task_values]))
            if task_values else None
        ),
        "steered_mean": (
            float(np.mean([row["steered"] for row in task_values]))
            if task_values else None
        ),
        "difference": float(np.mean(values)) if values else None,
        "relative_reduction": None,
        "bootstrap": None,
        "null_test": None,
        "raw_p": None,
        "holm_input_p": 1.0,
        "holm_p": None,
        "complete_case_ci_excludes_zero": None,
        "confirmatory_ci_excludes_zero": None,
        "sign_flip_rejects_uncorrected_0_05": None,
        "sign_flip_rejects_holm_0_05": None,
        "confirmatory_eligible": complete,
        "inferential_status": (
            "complete_50_of_50"
            if complete
            else "provisional_complete_case; significance withheld for missingness"
        ),
        "missing_pair_worst_case_bounds": worst_case_bounds,
        "unresolved": unresolved,
        "task_values": task_values,
    }
    if cell["baseline_mean"] not in (None, 0.0) and cell["difference"] is not None:
        cell["relative_reduction"] = cell["difference"] / cell["baseline_mean"]
    raw_p = None
    if len(values) >= 2:
        bootstrap, bootstrap_distribution = paired_bootstrap_mean(
            values,
            n_resamples=n_resamples,
            seed=seed,
            return_distribution=True,
        )
        null_test = paired_sign_flip_p(
            values,
            n_randomizations=max(10_000, n_resamples),
            seed=seed + 100_000,
        )
        raw_p = float(null_test["p_value"])
        value_array = np.asarray(values, dtype=float)
        jackknife = np.asarray([
            np.delete(value_array, index).mean()
            for index in range(value_array.size)
        ])
        percentile_fallback = bool(
            np.allclose(bootstrap_distribution, bootstrap_distribution[0])
            or np.allclose(jackknife, jackknife[0])
        )
        cell["bootstrap"] = {
            "method": (
                "percentile (BCa degenerate fallback)"
                if percentile_fallback else "BCa"
            ),
            "requested_method": "BCa",
            "fallback_rule": (
                "percentile when bootstrap or leave-one-task jackknife "
                "distribution is degenerate"
            ),
            "estimate": bootstrap.estimate,
            "ci_low": bootstrap.ci_low,
            "ci_high": bootstrap.ci_high,
            "ci": 0.95,
            "n_tasks": bootstrap.n_tasks,
            "n_resamples": bootstrap.n_resamples,
            "resample_unit": "task",
        }
        cell["null_test"] = null_test
        cell["raw_p"] = raw_p
        ci_excludes = bool(
            bootstrap.ci_low > 0 or bootstrap.ci_high < 0
        )
        cell["complete_case_ci_excludes_zero"] = ci_excludes
        if complete:
            cell["holm_input_p"] = raw_p
            cell["confirmatory_ci_excludes_zero"] = ci_excludes
            cell["sign_flip_rejects_uncorrected_0_05"] = raw_p < 0.05
    # Incomplete cells stay in the four-hypothesis Holm family at p=1.  Their
    # complete-case estimate/CI remains descriptive and explicitly provisional.
    return cell, float(cell["holm_input_p"])


def analyse_scored_records(
    scored: Sequence[dict],
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict:
    """Compute task-paired estimates/BCa CIs and Holm within each estimand."""
    if n_resamples < 100:
        raise ValueError("n_resamples must be >= 100")
    metrics = (
        CORRECTED_FOUR_TARGET_METRIC,
        RELEASED_BUG_COMPATIBLE_METRIC,
        SIX_LABEL_TOKEN_METRIC,
        SIX_LABEL_SENTENCE_METRIC,
    )
    report_metrics: dict = {}
    for metric_index, metric in enumerate(metrics):
        cells = {}
        raw_p = {}
        for behaviour_index, behaviour in enumerate(TARGET_BEHAVIOURS):
            cell, p_value = _paired_metric(
                scored,
                behaviour,
                metric,
                n_resamples=n_resamples,
                seed=seed + metric_index * 1000 + behaviour_index,
            )
            cells[behaviour] = cell
            # Every family always contains all four behaviours.  An incomplete
            # cell contributes p=1 rather than silently shrinking multiplicity.
            raw_p[behaviour] = 1.0 if p_value is None else p_value
        adjusted = holm_bonferroni(raw_p)
        for behaviour, cell in cells.items():
            cell["holm_p"] = float(adjusted[behaviour])
            if cell["confirmatory_eligible"]:
                cell["sign_flip_rejects_holm_0_05"] = cell["holm_p"] < 0.05
        report_metrics[metric] = {
            "multiplicity_family": "four behaviours; Holm within this estimand",
            "holm": adjusted,
            "cells": cells,
        }

    unresolved_counts = Counter(
        row.get("unresolved_reason", "unknown")
        for row in scored if row.get("score_status") != "resolved"
    )
    return {
        "schema": REPORT_SCHEMA,
        "record_counts": {
            "planned": len(scored),
            "resolved": sum(row.get("score_status") == "resolved" for row in scored),
            "unresolved": sum(row.get("score_status") != "resolved" for row in scored),
            "unresolved_by_reason": dict(sorted(unresolved_counts.items())),
            "resolved_four_target_denominator_zero": sum(
                row.get("score_status") == "resolved"
                and bool(row.get("four_target_denominator_zero"))
                for row in scored
            ),
        },
        "primary_metric": CORRECTED_FOUR_TARGET_METRIC,
        "primary_metric_provenance": {
            "released_repository": "https://github.com/cvenhoff/steering-thinking-llms",
            "audited_commit": "93259bc3410c99293351df41141cd16b4110422a",
            "released_file": "steering/evaluate_steering.py",
            "definition": (
                "target-label Qwen token interval count divided by the token "
                "interval count assigned to the four steering labels"
            ),
            "reasoning_region": (
                "whitespace-stripped text after <think> and before the first "
                "</think>, or completion end when unclosed; no degeneration "
                "or final-answer suffix cut"
            ),
            "tokenization_context": (
                "full prompt+completion when persisted/reconstructable; completion-only "
                "fallback is disclosed per record"
            ),
            "repeat_alignment_deviation": (
                "occurrence-aware ordered alignment fixes released str.find first-match aliasing"
            ),
            "unmatched_span_deviation": (
                "released code silently omits unmapped spans; this bridge withholds the "
                "entire row as unresolved, so missing labelled text cannot shrink a denominator"
            ),
        },
        "released_bug_compatible_sensitivity": {
            "metric": RELEASED_BUG_COMPATIBLE_METRIC,
            "definition": (
                "released response.find-from-zero alignment, exact boundary-token "
                "intervals, and silent per-span omission"
            ),
            "parser_scope": (
                "new bridge rows use current schema-parsed spans; raw annotator "
                "syntax is not retained, so released regex misparsing is not replayed"
            ),
            "official_asset_replay": (
                "with released strict-regex spans, all 2400 fractions across the "
                "600 public Qwen-1.5B records reproduce exactly"
            ),
        },
        "baseline_annotation_convention": {
            "bridge": (
                "50 shared baseline generations annotated once and reused in all "
                "four paired behaviour contrasts"
            ),
            "released_venhoff_code": (
                "the same 50 original generations were re-annotated independently "
                "inside each of four behaviour loops (200 baseline annotations)"
            ),
            "public_qwen15_consequence": (
                "reusing the backtracking-loop baseline annotations in the public "
                "asset changes original on-target means, relative to each loop's "
                "own annotation, by -0.02327 uncertainty, +0.00532 example, and "
                "+0.03447 knowledge; this 250-record bridge does not reproduce "
                "that annotation stochasticity"
            ),
        },
        "protocol_deviations_from_released_run": {
            "annotator": (
                f"{ANNOTATION_MODEL} with the thesis current schema/coverage gate; "
                "the audited released annotation helper defaults to GPT-4.1, "
                "while the stored public run does not independently log its "
                "executed annotator model"
            ),
            "baseline_annotation": "shared once rather than independently four times",
            "tasks_prompts": "the thesis frozen E1 tasks, not Venhoff's evaluation messages",
            "directions": (
                "thesis held-out six-label directions at released Qwen-1.5B norms, "
                "not Venhoff's original direction tensors"
            ),
        },
        "null_inference": (
            "BCa task bootstrap confidence intervals plus paired sign-flip p-values; "
            "the p-value assumes sign-exchangeable paired differences. CI exclusion "
            "and Holm-adjusted sign-flip rejection are reported separately"
        ),
        "missingness": (
            "missing/empty/incomplete/coverage-failed rows are unresolved, never zero; "
            "incomplete cells retain descriptive complete-case estimates but are "
            "ineligible for significance and enter Holm at p=1"
        ),
        "metrics": report_metrics,
    }


def run_analysis(
    eval_dir: Path,
    *,
    tokenizer_path: Path | None = None,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict:
    """Score the saved checkpoint and atomically write the comparison report."""
    eval_dir = Path(eval_dir).resolve()
    generation, provenance, _ = validate_generation_contract(eval_dir)
    annotations_path = eval_dir / "annotated_venhoff_bridge.json"
    if not annotations_path.is_file():
        raise FileNotFoundError(f"annotation checkpoint is missing: {annotations_path}")
    annotated = _load_json(annotations_path)
    if not isinstance(annotated, list):
        raise ValueError("annotation checkpoint must be a list")
    wrong_region = [
        _record_key(row) for row in annotated
        if isinstance(row, dict)
        and bool(row.get("annotation_complete"))
        and not _has_exact_venhoff_region_policy(row)
    ]
    if wrong_region:
        raise ValueError(
            "annotation checkpoint was not scored over the exact Venhoff "
            "reasoning region; refusing analysis (first keys: "
            f"{wrong_region[:3]})"
        )
    tokenizer, tokenizer_provenance = load_qwen_tokenizer(provenance, tokenizer_path)
    scored = score_records(generation, annotated, tokenizer)
    scored_doc = {
        "schema": SCORED_SCHEMA,
        "tokenizer": tokenizer_provenance,
        "coverage_rule_version": COVERAGE_RULE_VERSION,
        "records": scored,
    }
    scored_path = eval_dir / "venhoff_scored_records.json"
    atomic_json(scored_doc, scored_path)
    report = analyse_scored_records(scored, n_resamples=n_resamples, seed=seed)
    report["tokenizer"] = tokenizer_provenance
    report["sources"] = {
        "generation": {
            "path": str(eval_dir / "steering_results.json"),
            "sha256": sha256_file(eval_dir / "steering_results.json"),
        },
        "annotations": {
            "path": str(annotations_path),
            "sha256": sha256_file(annotations_path),
        },
        "scored_records": {
            "path": str(scored_path),
            "sha256": sha256_file(scored_path),
        },
    }
    for name in ("annotation_plan.json", "annotation_guard_manifest.json",
                 "annotation_status.json"):
        path = eval_dir / name
        if path.is_file():
            report["sources"][name.removesuffix(".json")] = {
                "path": str(path), "sha256": sha256_file(path)
            }
    root = Path(__file__).resolve().parents[1]
    report["analysis_code"] = {
        rel: sha256_file(root / rel)
        for rel in (
            "src/venhoff_bridge_eval.py",
            "src/annotation_coverage.py",
            "src/delta_floor.py",
            "src/steering_analysis.py",
        )
    }
    report_path = eval_dir / "venhoff_bridge_report.json"
    atomic_json(report, report_path)
    return report


def format_report(report: dict) -> str:
    """Compact terminal table for the primary and sensitivity conventions."""
    lines = []
    for metric_name, metric in report["metrics"].items():
        lines.append(f"\n{metric_name}")
        lines.append(
            f"{'behaviour':26s} {'N':>3s} {'base':>8s} {'steered':>8s} "
            f"{'diff':>8s} {'95% CI':>19s} {'Holm p':>8s} {'reject':>8s}"
        )
        for behaviour in TARGET_BEHAVIOURS:
            cell = metric["cells"][behaviour]
            boot = cell.get("bootstrap")
            ci = (
                f"[{boot['ci_low']:+.4f},{boot['ci_high']:+.4f}]"
                if boot else "unresolved"
            )
            def number(value):
                return f"{value:.4f}" if value is not None else "—"
            reject = cell["sign_flip_rejects_holm_0_05"]
            reject_text = "withheld" if reject is None else ("yes" if reject else "no")
            lines.append(
                f"{behaviour:26s} {cell['n_resolved_pairs']:3d} "
                f"{number(cell['baseline_mean']):>8s} "
                f"{number(cell['steered_mean']):>8s} "
                f"{number(cell['difference']):>8s} {ci:>19s} "
                f"{number(cell['holm_p']):>8s} "
                f"{reject_text:>8s}"
            )
    return "\n".join(lines)


__all__ = [
    "ANNOTATION_MAX_OUTPUT_TOKENS",
    "ANNOTATION_WINDOW_TOKENS",
    "BRIDGE_METHOD",
    "DEDUP_KEYS",
    "DEFAULT_BOUND_PATHS",
    "EXPECTED_RECORDS",
    "EXPECTED_TASKS",
    "SPEND_CEILING_USD",
    "analyse_scored_records",
    "annotation_status",
    "format_report",
    "prepare_annotation_manifest",
    "run_analysis",
    "run_annotation",
    "score_records",
    "token_label_counts",
    "validate_generation_contract",
]
