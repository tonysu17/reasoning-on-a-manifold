from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import ph2_verified_pull_terminate as watcher


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_terminal_status_contract():
    assert watcher.is_terminal("DONE")
    assert watcher.is_terminal("FAILED:j3-gates")
    assert not watcher.is_terminal("RUNNING:j3-gates")
    assert not watcher.is_terminal("")


def test_verify_tree_refuses_missing_and_mismatched(monkeypatch, tmp_path):
    local = tmp_path / "local"
    local.mkdir()
    (local / "a").write_bytes(b"a")
    monkeypatch.setattr(watcher, "ROOT", tmp_path)
    monkeypatch.setattr(
        watcher,
        "remote_manifest",
        lambda _, excluded_parts=(): {
            "a": hashlib.sha256(b"a").hexdigest(),
            "b": "0" * 64,
        },
    )
    with pytest.raises(watcher.FinishError, match="missing"):
        watcher.verify_tree("remote", "local")

    (local / "b").write_bytes(b"wrong")
    with pytest.raises(watcher.FinishError, match="mismatched"):
        watcher.verify_tree("remote", "local")


def test_verify_tree_accepts_remote_subset_with_matching_hashes(monkeypatch, tmp_path):
    local = tmp_path / "local"
    local.mkdir()
    (local / "a").write_bytes(b"a")
    (local / "extra").write_bytes(b"extra-local-audit")
    digest = hashlib.sha256(b"a").hexdigest()
    monkeypatch.setattr(watcher, "ROOT", tmp_path)
    monkeypatch.setattr(
        watcher, "remote_manifest", lambda _, excluded_parts=(): {"a": digest}
    )
    report = watcher.verify_tree("remote", "local")
    assert report["n_files"] == 1


def test_filtered_manifest_ignores_merged_model_copies(monkeypatch, tmp_path):
    local = tmp_path / "adapter"
    (local / "merged").mkdir(parents=True)
    (local / "adapter.bin").write_bytes(b"adapter")
    (local / "merged" / "model.bin").write_bytes(b"large reconstructable copy")
    digest = hashlib.sha256(b"adapter").hexdigest()
    monkeypatch.setattr(watcher, "ROOT", tmp_path)
    monkeypatch.setattr(
        watcher,
        "remote_manifest",
        lambda _, excluded_parts=(): {"adapter.bin": digest},
    )
    report = watcher.verify_tree("remote", "adapter", excluded_parts=("merged",))
    assert report["n_files"] == 1
    assert report["excluded_path_parts"] == ["merged"]


def test_remote_manifest_uses_literal_heredoc_transport(monkeypatch):
    seen = {}

    def fake_ssh(command, *, timeout):
        seen["command"] = command
        seen["timeout"] = timeout
        return json.dumps({"artifact.json": "f" * 64})

    monkeypatch.setattr(watcher, "ssh", fake_ssh)
    result = watcher.remote_manifest("results/ph2", excluded_parts=("merged",))
    assert result == {"artifact.json": "f" * 64}
    assert seen["command"].startswith("python3 - <<'PY'\n")
    assert seen["command"].endswith("\nPY")
    assert "excluded=set(['merged'])" in seen["command"]
    assert seen["timeout"] == 1800


def build_semantic_fixture(root: Path, status: str = "DONE") -> None:
    runtime = root / "results/ph2/runtime"
    runtime.mkdir(parents=True)
    (runtime / "PH2_STATUS").write_text(status)
    (runtime / "PH2_DONE.marker").write_text("done")
    p5 = root / "results/p5_wildguard/smoke"
    (p5 / "rows").mkdir(parents=True)
    (p5 / "rows_diag").mkdir()
    (p5 / "SMOKE_DONE.marker").write_text("done")
    for index in range(8):
        (p5 / "rows" / f"{index}.json").write_text("{}")
        (p5 / "rows_diag" / f"{index}.json").write_text("{}")
    if status == "DONE":
        for relative in watcher.REQUIRED_DONE:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("done")
        write_json(
            root / "results/ph2/gates/target_gates.json",
            {role: {"complete": True} for role in ("base", "star1", "deepscaler")},
        )
        write_json(
            root / "results/ph2/refit/refit_report.json",
            {role: {"complete": True} for role in ("star1", "deepscaler")},
        )


def test_semantic_done_gate_passes_only_with_complete_roles(monkeypatch, tmp_path):
    build_semantic_fixture(tmp_path)
    monkeypatch.setattr(watcher, "ROOT", tmp_path)
    watcher.verify_semantics("DONE")
    report = tmp_path / "results/ph2/refit/refit_report.json"
    write_json(report, {"star1": {"complete": True}})
    with pytest.raises(watcher.FinishError, match="role set"):
        watcher.verify_semantics("DONE")


def test_failed_terminal_can_preserve_debug_artifacts_without_false_done_claim(
    monkeypatch, tmp_path
):
    build_semantic_fixture(tmp_path, status="FAILED:j3-gates")
    monkeypatch.setattr(watcher, "ROOT", tmp_path)
    watcher.verify_semantics("FAILED:j3-gates")


def test_running_snapshot_never_pulls_or_terminates(monkeypatch, tmp_path):
    monkeypatch.setattr(watcher, "TERMINATION_RECORD", tmp_path / "none.json")
    monkeypatch.setattr(
        watcher,
        "remote_snapshot",
        lambda: {
            "pod_id": watcher.EXPECTED_POD_ID,
            "status": "RUNNING:j3-gates",
            "completion_marker": False,
            "driver_present": True,
            "p5_status": "P5_DONE",
        },
    )
    monkeypatch.setattr(watcher, "audit", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        watcher, "pull_and_verify", lambda *_: pytest.fail("must not pull running pod")
    )
    monkeypatch.setattr(
        watcher, "terminate_exact_pod", lambda: pytest.fail("must not terminate running pod")
    )
    result = watcher.check_once()
    assert result["status"] == "running"
    assert result["phase2_status"] == "RUNNING:j3-gates"


def test_terminal_without_marker_or_with_driver_refuses(monkeypatch, tmp_path):
    monkeypatch.setattr(watcher, "TERMINATION_RECORD", tmp_path / "none.json")
    monkeypatch.setattr(watcher, "audit", lambda *args, **kwargs: None)
    snapshot = {
        "pod_id": watcher.EXPECTED_POD_ID,
        "status": "DONE",
        "completion_marker": False,
        "driver_present": False,
        "p5_status": "P5_DONE",
    }
    monkeypatch.setattr(watcher, "remote_snapshot", lambda: dict(snapshot))
    with pytest.raises(watcher.FinishError, match="without PH2_DONE"):
        watcher.check_once()
    snapshot["completion_marker"] = True
    snapshot["driver_present"] = True
    with pytest.raises(watcher.FinishError, match="driver still exists"):
        watcher.check_once()


def test_terminal_pull_failure_never_calls_termination(monkeypatch, tmp_path):
    monkeypatch.setattr(watcher, "TERMINATION_RECORD", tmp_path / "none.json")
    monkeypatch.setattr(watcher, "audit", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        watcher,
        "remote_snapshot",
        lambda: {
            "pod_id": watcher.EXPECTED_POD_ID,
            "status": "DONE",
            "completion_marker": True,
            "driver_present": False,
            "p5_status": "P5_DONE",
        },
    )
    monkeypatch.setattr(
        watcher,
        "pull_and_verify",
        lambda *_: (_ for _ in ()).throw(watcher.FinishError("hash mismatch")),
    )
    monkeypatch.setattr(
        watcher, "terminate_exact_pod", lambda: pytest.fail("must not terminate")
    )
    with pytest.raises(watcher.FinishError, match="hash mismatch"):
        watcher.check_once()


def test_terminal_verified_path_terminates_once_and_records(monkeypatch, tmp_path):
    record = tmp_path / "terminated.json"
    monkeypatch.setattr(watcher, "TERMINATION_RECORD", record)
    monkeypatch.setattr(watcher, "audit", lambda *args, **kwargs: None)
    snapshot = {
        "pod_id": watcher.EXPECTED_POD_ID,
        "status": "DONE",
        "completion_marker": True,
        "driver_present": False,
        "p5_status": "P5_DONE",
    }
    monkeypatch.setattr(watcher, "remote_snapshot", lambda: dict(snapshot))
    monkeypatch.setattr(
        watcher,
        "pull_and_verify",
        lambda *_: [{"remote": "results/ph2", "manifest_sha256": "a" * 64}],
    )
    calls = []
    monkeypatch.setattr(
        watcher,
        "terminate_exact_pod",
        lambda: calls.append(watcher.EXPECTED_POD_ID) or {"data": {"podTerminate": None}},
    )

    first = watcher.check_once()
    assert first["status"] == "termination_accepted"
    assert calls == [watcher.EXPECTED_POD_ID]
    saved = json.loads(record.read_text())
    assert saved["pod_id"] == watcher.EXPECTED_POD_ID
    assert saved["termination_api_accepted"] is True

    second = watcher.check_once()
    assert second["status"] == "already_terminated"
    assert calls == [watcher.EXPECTED_POD_ID]
