from __future__ import annotations

import json
from pathlib import Path

import jspace_pilot_preflight as preflight


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/prereg/JSPACE_R1_BENCHMARK_MANIFEST_2026-08-10.json"


def test_sealed_repository_inputs_pass_offline_preflight() -> None:
    report = preflight.validate(MANIFEST, root=ROOT)
    assert report["ok"], report["errors"]
    assert report["execution_authorised"] is False
    assert report["n_prompts"] == 5


def test_tampered_prompt_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(MANIFEST.read_text())
    payload["prompts"][0]["prompt"] += " tampered"
    candidate = tmp_path / "manifest.json"
    candidate.write_text(json.dumps(payload))

    report = preflight.validate(candidate, root=ROOT)
    assert not report["ok"]
    assert any("prompt text differs" in error for error in report["errors"])


def test_execution_authorisation_cannot_be_silently_enabled(tmp_path: Path) -> None:
    payload = json.loads(MANIFEST.read_text())
    payload["execution_authorised"] = True
    candidate = tmp_path / "manifest.json"
    candidate.write_text(json.dumps(payload))

    report = preflight.validate(candidate, root=ROOT)
    assert not report["ok"]
    assert "manifest must keep execution_authorised=false" in report["errors"]

