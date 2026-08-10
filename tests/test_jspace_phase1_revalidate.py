import hashlib
import tempfile
import unittest
from pathlib import Path

import jspace_phase1_revalidate as recovery


class ImmutableSourceTests(unittest.TestCase):
    def test_exact_inventory_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.txt").write_text("a")
            expected = {
                "a.txt": hashlib.sha256(b"a").hexdigest(),
            }
            recovery.verify_source_artifacts(root, expected)
            (root / "validation.json").write_text("v")
            recovery.verify_source_artifacts(
                root, expected, allowed_additions={"validation.json"}
            )

    def test_extra_or_changed_source_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.txt").write_text("a")
            expected = {"a.txt": hashlib.sha256(b"a").hexdigest()}
            (root / "extra.txt").write_text("extra")
            with self.assertRaisesRegex(ValueError, "inventory mismatch"):
                recovery.verify_source_artifacts(root, expected)
            (root / "extra.txt").unlink()
            (root / "a.txt").write_text("changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                recovery.verify_source_artifacts(root, expected)

    def test_validation_check_inventory_must_be_exact_and_true(self):
        complete = {name: True for name in recovery.REQUIRED_VALIDATION_CHECKS}
        recovery.require_exact_validation_checks(complete)
        missing = dict(complete)
        missing.pop(next(iter(missing)))
        with self.assertRaisesRegex(RuntimeError, "inventory"):
            recovery.require_exact_validation_checks(missing)
        failed = dict(complete)
        failed[next(iter(failed))] = False
        with self.assertRaisesRegex(RuntimeError, "inventory"):
            recovery.require_exact_validation_checks(failed)
        extra = {**complete, "unregistered": True}
        with self.assertRaisesRegex(RuntimeError, "inventory"):
            recovery.require_exact_validation_checks(extra)


if __name__ == "__main__":
    unittest.main()
