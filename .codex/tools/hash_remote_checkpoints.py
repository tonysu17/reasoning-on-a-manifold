#!/usr/bin/env python3
"""Hash metadata-only files for the 15 remote provenance rows.

Only tokenizer.json, tokenizer_config.json, and config.json are downloaded. No model weights,
adapters, generation, or inference are involved. By default the script writes the permitted
`resolved_checkpoints` fields into configs/analysis/checkpoint_provenance.yaml and renders
.codex/out/PROVENANCE_COMPLETION.md. Use --no-update-yaml for a read-only network audit.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError, RemoteEntryNotFoundError


ROOT = Path(__file__).resolve().parents[2]
YAML_PATH = ROOT / "configs/analysis/checkpoint_provenance.yaml"
REPORT_PATH = ROOT / ".codex/out/PROVENANCE_COMPLETION.md"
FILES = ("tokenizer.json", "tokenizer_config.json", "config.json")
DATE = "2026-08-02"


@dataclass(frozen=True)
class Target:
    repo_id: str
    revision: str = "main"
    alias: str | None = None


TARGETS: dict[str, tuple[Target, ...]] = {
    "Nickyang/FastCuRL-1.5B-Preview|V2|V3": (
        Target("Nickyang/FastCuRL-1.5B-Preview"),
        Target("Nickyang/FastCuRL-1.5B-V2"),
        Target("Nickyang/FastCuRL-1.5B-V3"),
    ),
    "nvidia/Nemotron-Research-Reasoning-Qwen-1.5B (v1,v2 revisions)": (
        Target("nvidia/Nemotron-Research-Reasoning-Qwen-1.5B", "v1"),
        Target("nvidia/Nemotron-Research-Reasoning-Qwen-1.5B", "v2"),
    ),
    "knoveleng/Open-RS1|RS2|RS3": (
        Target("knoveleng/Open-RS1"),
        Target("knoveleng/Open-RS2"),
        Target("knoveleng/Open-RS3"),
    ),
    "RUC-AIBOX/STILL-3-1.5B-preview": (Target("RUC-AIBOX/STILL-3-1.5B-preview"),),
    "hbx/JustRL-DeepSeek-1.5B": (Target("hbx/JustRL-DeepSeek-1.5B"),),
    "agentica-org/DeepCoder-1.5B-Preview": (Target("agentica-org/DeepCoder-1.5B-Preview"),),
    "nvidia/DLER-R1-1.5B-Research": (Target("nvidia/DLER-R1-1.5B-Research"),),
    "Zyphra/ZR1-1.5B": (Target("Zyphra/ZR1-1.5B"),),
    "l3lab/L1-Qwen-1.5B, L1-Qwen-1.5B-Max": (
        Target("l3lab/L1-Qwen-1.5B-Exact", alias="row's unqualified L1 checkpoint"),
        Target("l3lab/L1-Qwen-1.5B-Max"),
    ),
    "theshyustc/CoRT-Prompt-Hint-1.5B-RL, CoRT-Hint-Engineering-1.5B-RL": (
        Target("theshyustc/CoRT-Prompt-Hint-1.5B-RL"),
        Target("theshyustc/CoRT-Hint-Engineering-1.5B-RL"),
    ),
    "huihui-ai/DeepSeek-R1-Distill-Qwen-1.5B-abliterated": (
        Target("huihui-ai/DeepSeek-R1-Distill-Qwen-1.5B-abliterated"),
    ),
    "stepenZEN/DeepSeek-R1-Distill-Qwen-1.5B-Abliterated-dpo": (
        Target("stepenZEN/DeepSeek-R1-Distill-Qwen-1.5B-Abliterated-dpo"),
    ),
    "UCSC-VLAA/STAR1-R1-Distill-7B|8B|14B|32B": (
        Target("UCSC-VLAA/STAR1-R1-Distill-7B"),
        Target("UCSC-VLAA/STAR1-R1-Distill-8B"),
        Target("UCSC-VLAA/STAR1-R1-Distill-14B"),
        Target("UCSC-VLAA/STAR1-R1-Distill-32B"),
    ),
    "openai/gpt-oss-20b (registry: safety_gpt_oss)": (Target("openai/gpt-oss-20b"),),
    "openai/gpt-oss-safeguard-20b": (Target("openai/gpt-oss-safeguard-20b"),),
}


def sha256_16(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _nested(config: dict, key: str) -> Any:
    if key in config:
        return config[key]
    text = config.get("text_config")
    if isinstance(text, dict) and key in text:
        return text[key]
    return None


def arch_string(config: dict) -> str:
    hidden = _nested(config, "hidden_size")
    layers = _nested(config, "num_hidden_layers")
    context = _nested(config, "max_position_embeddings")
    bos = _nested(config, "bos_token_id")
    model_type = config.get("model_type") or _nested(config, "model_type")
    architectures = config.get("architectures") or []
    architecture = architectures[0] if architectures else None
    pieces = [
        f"{hidden}h" if hidden is not None else "?h",
        f"{layers}L" if layers is not None else "?L",
        f"{context}ctx" if context is not None else "?ctx",
        f"bos{bos}" if bos is not None else "bos?",
    ]
    suffix = "/".join(str(value) for value in (model_type, architecture) if value)
    return "/".join(pieces) + (f" [{suffix}]" if suffix else "")


def download_target(api: HfApi, target: Target, temp_root: Path) -> tuple[dict, dict[str, dict]]:
    info = api.model_info(target.repo_id, revision=target.revision, files_metadata=False)
    commit = str(info.sha)
    target_dir = temp_root / re.sub(r"[^A-Za-z0-9_.-]+", "__", target.repo_id) / commit[:12]
    target_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict] = {}
    parsed: dict[str, dict] = {}
    for filename in FILES:
        try:
            local = Path(hf_hub_download(
                repo_id=target.repo_id,
                filename=filename,
                revision=commit,
                local_dir=target_dir,
            ))
            files[filename] = {"sha256_16": sha256_16(local), "bytes": local.stat().st_size}
            if filename.endswith(".json"):
                parsed[filename] = json.loads(local.read_text())
        except (RemoteEntryNotFoundError, HfHubHTTPError) as exc:
            files[filename] = {"sha256_16": None, "bytes": None,
                               "missing": type(exc).__name__}
    config = parsed.get("config.json", {})
    result = {
        "id": target.repo_id,
        "revision": target.revision,
        "commit": commit,
        "tokenizer_json_sha256_16": files["tokenizer.json"]["sha256_16"],
        "tokenizer_config_json_sha256_16": files["tokenizer_config.json"]["sha256_16"],
        "config_json_sha256_16": files["config.json"]["sha256_16"],
        "arch": arch_string(config),
        "verified": f"HF metadata hash {DATE}",
    }
    if target.alias:
        result["resolution_note"] = target.alias
    missing = [name for name, row in files.items() if row["sha256_16"] is None]
    if missing:
        result["missing_files"] = missing
    return result, parsed


def _render_resolved(rows: list[dict]) -> list[str]:
    lines = ["    resolved_checkpoints:"]
    for row in rows:
        lines.append(f"      - id: {json.dumps(row['id'])}")
        for key in ("revision", "commit", "tokenizer_json_sha256_16",
                    "tokenizer_config_json_sha256_16", "config_json_sha256_16", "arch",
                    "verified", "resolution_note", "missing_files"):
            if key not in row:
                continue
            value = row[key]
            if isinstance(value, list):
                lines.append(f"        {key}:")
                lines.extend(f"          - {json.dumps(item)}" for item in value)
            elif value is None:
                lines.append(f"        {key}: null")
            else:
                lines.append(f"        {key}: {json.dumps(value)}")
    return lines


def update_yaml_preserving_existing_text(path: Path, results: dict[str, list[dict]]) -> None:
    """Append/replace only generated fields inside each remote row; preserve all other text."""
    lines = path.read_text().splitlines()
    row_starts = [i for i, line in enumerate(lines) if line.startswith("  - id: ")]
    if len(row_starts) != len(TARGETS):
        raise RuntimeError(f"expected {len(TARGETS)} remote rows, found {len(row_starts)}")
    output: list[str] = []
    for pos, start in enumerate(row_starts):
        if pos == 0:
            output.extend(lines[:start])
        end = row_starts[pos + 1] if pos + 1 < len(row_starts) else len(lines)
        block = lines[start:end]
        match = re.fullmatch(r'  - id: "(.*)"', block[0])
        if not match:
            raise RuntimeError(f"cannot parse remote row: {block[0]}")
        row_id = match.group(1)
        if row_id not in results:
            raise RuntimeError(f"no metadata results for row {row_id!r}")
        generated_at = next((i for i, line in enumerate(block)
                             if line.startswith("    resolved_checkpoints:")), None)
        if generated_at is not None:
            block = block[:generated_at]
        output.extend(block)
        output.extend(_render_resolved(results[row_id]))
    path.write_text("\n".join(output) + "\n")


def flatten_json(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    out: dict[str, Any] = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(child, dict):
            out.update(flatten_json(child, path))
        else:
            out[path] = child
    return out


def selected_config_diff(reference: dict, candidate: dict) -> list[dict]:
    """Architecture/token-id diff; weight-file metadata and auto-map noise are excluded."""
    keys = {
        "model_type", "architectures", "hidden_size", "num_hidden_layers",
        "max_position_embeddings", "bos_token_id", "eos_token_id", "vocab_size",
        "tie_word_embeddings", "rope_theta", "sliding_window",
        "text_config.hidden_size", "text_config.num_hidden_layers",
        "text_config.max_position_embeddings", "text_config.bos_token_id",
        "text_config.eos_token_id", "text_config.vocab_size",
    }
    ref = flatten_json(reference)
    cand = flatten_json(candidate)
    return [{"field": key, "r1": ref.get(key), "candidate": cand.get(key)}
            for key in sorted(keys) if ref.get(key) != cand.get(key)]


def chat_template_report(reference: dict, candidate: dict) -> dict:
    ref = reference.get("chat_template")
    cand = candidate.get("chat_template")
    if ref == cand:
        return {"identical": True,
                "sha256_16": hashlib.sha256((ref or "").encode()).hexdigest()[:16]}
    diff = list(difflib.unified_diff(
        (ref or "<missing>").splitlines(), (cand or "<missing>").splitlines(),
        fromfile="R1", tofile="candidate", lineterm=""))
    # Templates are often a single multi-kilobyte Jinja line. Keep the report auditable
    # without embedding both complete templates; the file hashes remain the exact identity.
    preview = [line if len(line) <= 600 else line[:600] + "... [truncated]" for line in diff[:24]]
    return {
        "identical": False,
        "r1_sha256_16": hashlib.sha256((ref or "").encode()).hexdigest()[:16],
        "candidate_sha256_16": hashlib.sha256((cand or "").encode()).hexdigest()[:16],
        "diff_preview": preview,
    }


def render_report(results: dict[str, list[dict]], parsed_by_id: dict[str, dict[str, dict]],
                  r1_parsed: dict[str, dict]) -> str:
    all_rows = [row for rows in results.values() for row in rows]
    lines = [
        "# Provenance completion — remote checkpoint metadata",
        "",
        f"**Date:** {DATE}  ",
        "**Scope:** tokenizer/config metadata only. No weights or adapters were downloaded.",
        "",
        f"Resolved **{len(results)} provenance rows** to **{len(all_rows)} immutable "
        "checkpoint revisions**. The YAML now records a commit plus sha256_16 for each of "
        "`tokenizer.json`, `tokenizer_config.json`, and `config.json` when present, and an "
        "architecture summary derived from `config.json`.",
        "",
        "## Metadata table",
        "",
        "| checkpoint @ revision | commit | tokenizer sha16 | tokenizer-config sha16 | "
        "config sha16 | architecture |",
        "|---|---|---|---|---|---|",
    ]
    for row in all_rows:
        lines.append(
            f"| `{row['id']}@{row['revision']}` | `{row['commit'][:12]}` | "
            f"`{row['tokenizer_json_sha256_16'] or 'MISSING'}` | "
            f"`{row['tokenizer_config_json_sha256_16'] or 'MISSING'}` | "
            f"`{row['config_json_sha256_16'] or 'MISSING'}` | {row['arch']} |")

    lines += [
        "",
        "## CoRT deep-check",
        "",
        "Reference R1 tokenizer hashes from the core provenance row: "
        "`tokenizer.json=88145e3c3249adc2`; STAR1/DeepScaleR alternate family hash "
        "`e20ddafc659ba902`.",
        "",
    ]
    r1_tok_cfg = r1_parsed.get("tokenizer_config.json", {})
    r1_cfg = r1_parsed.get("config.json", {})
    for repo_id in ("theshyustc/CoRT-Prompt-Hint-1.5B-RL",
                    "theshyustc/CoRT-Hint-Engineering-1.5B-RL"):
        row = next(item for item in all_rows if item["id"] == repo_id)
        parsed = parsed_by_id[repo_id]
        template = chat_template_report(r1_tok_cfg, parsed.get("tokenizer_config.json", {}))
        config_diff = selected_config_diff(r1_cfg, parsed.get("config.json", {}))
        tok_hash = row["tokenizer_json_sha256_16"]
        family = ("R1-exact" if tok_hash == "88145e3c3249adc2" else
                  "STAR1/DeepScaleR-exact" if tok_hash == "e20ddafc659ba902" else
                  "neither known family hash")
        lines += [
            f"### `{repo_id}`",
            "",
            f"- tokenizer.json `{tok_hash}`: **{family}**.",
            f"- chat template identical to R1 metadata: **{template['identical']}**.",
        ]
        if not template["identical"]:
            lines += [
                f"  R1 template sha16 `{template['r1_sha256_16']}`; candidate "
                f"`{template['candidate_sha256_16']}`. Diff preview:",
                "",
                "```diff",
                *template["diff_preview"],
                "```",
            ]
        if config_diff:
            lines += ["- selected config differences from R1:", ""]
            for diff in config_diff:
                lines.append(
                    f"  - `{diff['field']}`: R1 `{diff['r1']}`; candidate "
                    f"`{diff['candidate']}`")
        else:
            lines.append("- selected architecture/token-id config fields: identical to R1.")
        lines.append("")

    lines += [
        "## Weight-delta sanity-check specification — unrun",
        "",
        "This section is a specification only; the metadata pass downloaded no weights.",
        "",
        "1. Resolve R1 and each CoRT checkpoint to the immutable commits recorded above; "
        "verify identical tensor-key sets, shapes, and dtypes before subtraction.",
        "2. Stream one safetensors shard at a time on CPU. For every floating tensor compute "
        "`||theta_CoRT-theta_R1||_F`, `||theta_R1||_F`, relative norm, maximum absolute delta, "
        "and nonzero fraction; hash the ordered tensor-key/shape manifest.",
        "3. Aggregate squared norms globally and by layer/module (attention, MLP, embeddings, "
        "normalisation, LM head). Verify finite deltas and reject a purported descendant if "
        "keys/shapes differ outside a predeclared tied-weight exception.",
        "4. Compare the two CoRT delta vectors by streaming dot product/cosine, and compare "
        "their norms with an identity reload (zero) and a same-lineage post-training reference. "
        "Do not infer lineage from a small norm alone.",
        "5. Record source commits, per-file sha256, code commit/dirty state, accumulation dtype, "
        "and complete/partial status. Do not promote CoRT into direct-coordinate causal transport "
        "until this check and the byte-identical-input-id gate pass.",
        "",
        "## Caveats",
        "",
        "- A metadata hash establishes the exact small files inspected, not declared parentage "
        "or weight lineage.",
        "- The combined YAML labels are planning aliases. The L1 row's unqualified checkpoint "
        "was resolved to the public `l3lab/L1-Qwen-1.5B-Exact`; this resolution remains explicit "
        "rather than silently changing the original row label.",
        "- Missing metadata files remain `null`/`MISSING`; they are not synthesized from a "
        "parent checkpoint.",
        "- The pre-existing file is named `.yaml` but is not currently accepted by PyYAML: "
        "some unquoted scalar values before `remote_verified` contain a colon. This pass was "
        "only authorised to append provenance fields, so it preserves that text and discloses "
        "the syntax debt rather than silently rewriting existing rows.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", type=Path, default=YAML_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--no-update-yaml", action="store_true")
    args = parser.parse_args()

    source_text = args.yaml.read_text()
    if "remote_verified:" not in source_text:
        raise SystemExit("checkpoint provenance file has no remote_verified section")
    remote_text = source_text.split("remote_verified:", 1)[1]
    remote_ids = re.findall(r'^  - id: "(.*)"$', remote_text, flags=re.MULTILINE)
    if remote_ids != list(TARGETS):
        raise SystemExit("remote_verified rows do not match the script's sealed resolution map")

    api = HfApi()
    results: dict[str, list[dict]] = {}
    parsed_by_id: dict[str, dict[str, dict]] = {}
    with tempfile.TemporaryDirectory(prefix="phase0-provenance-") as temp:
        temp_root = Path(temp)
        for row_id, targets in TARGETS.items():
            resolved = []
            for target in targets:
                row, parsed = download_target(api, target, temp_root)
                resolved.append(row)
                parsed_by_id[target.repo_id] = parsed
            results[row_id] = resolved
        _, r1_parsed = download_target(
            api, Target("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"), temp_root)

    if not args.no_update_yaml:
        update_yaml_preserving_existing_text(args.yaml, results)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(results, parsed_by_id, r1_parsed))
    print(json.dumps({
        "remote_rows": len(results),
        "resolved_revisions": sum(map(len, results.values())),
        "yaml_updated": not args.no_update_yaml,
        "report": str(args.report),
    }, indent=2))


if __name__ == "__main__":
    main()
