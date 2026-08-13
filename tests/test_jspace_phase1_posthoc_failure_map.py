import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / ".codex/out/jspace_phase1_posthoc_failure_map.py"
)
SPEC = importlib.util.spec_from_file_location("jspace_phase1_posthoc", SCRIPT_PATH)
posthoc = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(posthoc)


class PassAt25Tests(unittest.TestCase):
    def make_values(self):
        values = np.full((2, posthoc.N_LAYERS, posthoc.K), 99, dtype=np.int64)
        values[0, 0, 0] = 10
        values[0, 1, 0] = 11
        values[1, 0, 0] = 20
        return values

    def test_item_level_fraction_and_layer_selection(self):
        values = self.make_values()
        labels = [[10, 11], [20]]
        self.assertEqual(posthoc.pass_at_25(values, labels, layers=[0]), 0.75)
        self.assertEqual(posthoc.pass_at_25(values, labels, layers=[1]), 0.25)

    def test_cumulative_union_is_not_layer_average(self):
        values = self.make_values()
        labels = [[10, 11], [20]]
        self.assertEqual(posthoc.pass_at_25(values, labels, layers=[0, 1]), 1.0)
        self.assertNotEqual(
            posthoc.pass_at_25(values, labels, layers=[0, 1]),
            np.mean(
                [
                    posthoc.pass_at_25(values, labels, layers=[0]),
                    posthoc.pass_at_25(values, labels, layers=[1]),
                ]
            ),
        )

    def test_items_with_any_hit_uses_items_not_labels(self):
        values = self.make_values()
        labels = [[10, 11], [20]]
        self.assertEqual(
            posthoc.items_with_any_hit(values, labels, layers=[0]), 2
        )
        self.assertEqual(
            posthoc.items_with_any_hit(values, labels, layers=[1]), 1
        )


class RankAndPeakTests(unittest.TestCase):
    def test_rank_is_one_based_and_censored_at_26(self):
        row = np.arange(100, 125, dtype=np.int64)
        self.assertEqual(posthoc.censored_rank(row, 100), 1)
        self.assertEqual(posthoc.censored_rank(row, 124), 25)
        self.assertEqual(posthoc.censored_rank(row, 999), 26)

    def test_peak_tie_uses_lowest_layer(self):
        self.assertEqual(posthoc.lowest_peak_layer([0.1, 0.4, 0.4, 0.2]), 1)


class IntegrityTests(unittest.TestCase):
    def test_require_hash_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.bin"
            path.write_bytes(b"frozen source")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                posthoc.require_hash(path, "0" * 64)

    def test_npz_inventory_rejects_missing_and_extra_keys(self):
        expected = posthoc.expected_npz_keys()
        posthoc.require_exact_npz_inventory(expected)
        with self.assertRaisesRegex(ValueError, "NPZ key inventory mismatch"):
            posthoc.require_exact_npz_inventory(expected - {next(iter(expected))})
        with self.assertRaisesRegex(ValueError, "NPZ key inventory mismatch"):
            posthoc.require_exact_npz_inventory(expected | {"unexpected"})

    def test_manifest_records_generated_hashes_and_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            names = (
                "failure_map.json",
                "layerwise_pass25.csv",
                "target_rank_censored.csv",
                "jspace_phase1_layerwise_pass25.png",
                "REPORT.md",
            )
            for name in names:
                (output_dir / name).write_bytes(name.encode())
            posthoc.write_manifest(output_dir)
            manifest = json.loads(
                (output_dir / "DIAGNOSTIC_MANIFEST.json").read_text()
            )
            self.assertEqual(set(manifest["generated"]), set(names))
            for name in names:
                self.assertEqual(
                    manifest["generated"][name]["sha256"],
                    posthoc.sha256_file(output_dir / name),
                )
            self.assertFalse(manifest["scientific_gate_pass"])
            self.assertFalse(manifest["phase2_licensed"])


if __name__ == "__main__":
    unittest.main()
