import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


WATCHER_PATH = (
    Path(__file__).resolve().parents[1]
    / ".codex/out/jspace_phase1_finish_watch.py"
)
SPEC = importlib.util.spec_from_file_location("jspace_phase1_finish_watch", WATCHER_PATH)
watcher = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(watcher)


class TerminalIntegrityTests(unittest.TestCase):
    def test_gate_must_be_real_boolean_and_equal_everywhere(self):
        validation = {"details": {"scientific_gate": {"pass": False}}}
        report = {"scientific_gate": {"pass": False}}
        done = {"scientific_gate_pass": False}
        watcher.require_gate_consistency(validation, report, done)
        with self.assertRaises(watcher.HoldPod):
            watcher.require_gate_consistency(
                validation, report, {"scientific_gate_pass": True}
            )
        with self.assertRaises(watcher.HoldPod):
            watcher.require_gate_consistency(
                validation, report, {"scientific_gate_pass": 0}
            )

    def test_promoted_bundle_is_rehashed_before_termination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.bin"
            data.write_bytes(b"sealed")
            manifest = {
                "files": {
                    "data.bin": {
                        "sha256": hashlib.sha256(b"sealed").hexdigest(),
                        "size_bytes": 6,
                    }
                }
            }
            artifact = root / "ARTIFACT_MANIFEST.json"
            artifact.write_text(json.dumps(manifest))
            done = root / "DONE.json"
            done.write_text(json.dumps({"state": "DONE"}))
            receipt = {
                "artifact_manifest_sha256": watcher.sha256_file(artifact),
                "done_sha256": watcher.sha256_file(done),
            }
            watcher.verify_local_terminal_bundle(root, receipt)
            data.write_bytes(b"changed")
            with self.assertRaises(watcher.HoldPod):
                watcher.verify_local_terminal_bundle(root, receipt)


if __name__ == "__main__":
    unittest.main()
