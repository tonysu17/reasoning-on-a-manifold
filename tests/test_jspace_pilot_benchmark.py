from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import jspace_pilot_benchmark as benchmark


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "jspace_pilot_benchmark.py"
MANIFEST = ROOT / "results/prereg/JSPACE_R1_BENCHMARK_MANIFEST_2026-08-10.json"


def _valid_layer_summary() -> dict[str, dict]:
    return {
        str(layer): {"shape": [1536, 1536], "dtype": "torch.float32", "finite": True}
        for layer in range(27)
    }


def _valid_prompt_summary() -> dict:
    return {
        "layers": _valid_layer_summary(),
        "peak_gpu_allocated_bytes": 5 * 1024**3,
        "peak_gpu_reserved_bytes": 6 * 1024**3,
        "peak_cpu_rss_bytes": 4 * 1024**3,
    }


def test_runner_refuses_without_explicit_authorisation() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--jlens-checkout", str(ROOT)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "requires --authorised" in completed.stderr


def test_gate_accepts_complete_synthetic_report() -> None:
    report = {
        "status": "complete",
        "prompts": [_valid_prompt_summary() for _ in range(5)],
        "hardware": {"total_memory_bytes": 24 * 1024**3},
        "peak_gpu_reserved_bytes": 20 * 1024**3,
        "coordinate_identity": {
            "same_module_object": True,
            "allclose_rtol_1e-5_atol_1e-6": True,
        },
        "repeat_first": {"relative_frobenius": {"max": 0.0}},
    }
    assert benchmark.evaluate_gate(report)["pass"] is True


def test_gate_rejects_missing_headroom_and_repeat_instability() -> None:
    report = {
        "status": "complete",
        "prompts": [_valid_prompt_summary() for _ in range(5)],
        "hardware": {"total_memory_bytes": 24 * 1024**3},
        "peak_gpu_reserved_bytes": 23 * 1024**3,
        "coordinate_identity": {
            "same_module_object": True,
            "allclose_rtol_1e-5_atol_1e-6": True,
        },
        "repeat_first": {"relative_frobenius": {"max": 2e-6}},
    }
    gate = benchmark.evaluate_gate(report)
    assert gate["pass"] is False
    assert gate["checks"]["gpu_headroom_at_least_2gib"] is False
    assert gate["checks"]["repeat_relative_frobenius_le_1e-6"] is False


def test_gate_rejects_missing_per_prompt_resource_metrics() -> None:
    report = {
        "status": "complete",
        "prompts": [{"layers": _valid_layer_summary()} for _ in range(5)],
        "hardware": {"total_memory_bytes": 24 * 1024**3},
        "peak_gpu_reserved_bytes": 20 * 1024**3,
        "coordinate_identity": {
            "same_module_object": True,
            "allclose_rtol_1e-5_atol_1e-6": True,
        },
        "repeat_first": {"relative_frobenius": {"max": 0.0}},
    }
    gate = benchmark.evaluate_gate(report)
    assert gate["pass"] is False
    assert gate["checks"]["per_prompt_resource_metrics_recorded"] is False


def test_worker_command_carries_explicit_authorisation(tmp_path: Path) -> None:
    args = benchmark.parse_args(
        [
            "--authorised",
            "--manifest",
            str(MANIFEST),
            "--jlens-checkout",
            str(tmp_path),
            "--local-files-only",
        ]
    )
    command = benchmark._worker_command(
        args,
        dim_batch=8,
        output=tmp_path / "worker.json",
        prompt_id="MATH_000",
        repeat_first=False,
    )
    assert "--authorised" in command
    assert "--local-files-only" in command
    assert command[command.index("--dim-batch") + 1] == "8"
