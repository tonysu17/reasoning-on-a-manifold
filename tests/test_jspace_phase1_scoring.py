import unittest

import numpy as np
import torch

import jspace_phase1_scoring as scoring
import jspace_phase1_run as runner


class DeterministicTopKTests(unittest.TestCase):
    def test_strict_boundary_orders_equal_logits_by_token_id(self):
        logits = torch.zeros(1, 30)
        logits[0, 7] = 3
        logits[0, 3] = 3
        logits[0, 20] = 2
        ids, tied = scoring.deterministic_topk_ids(logits, k=3)
        self.assertEqual(tied, 0)
        self.assertEqual(ids.tolist(), [[3, 7, 20]])

    def test_boundary_tie_selects_lower_token_ids(self):
        logits = torch.zeros(1, 30)
        logits[0, 29] = 2
        logits[0, 8] = 1
        logits[0, 4] = 1
        logits[0, 2] = 1
        ids, tied = scoring.deterministic_topk_ids(logits, k=3)
        self.assertEqual(tied, 1)
        self.assertEqual(ids.tolist(), [[29, 2, 4]])

    def test_all_equal_selects_lowest_ids(self):
        ids, tied = scoring.deterministic_topk_ids(torch.zeros(1, 30), k=25)
        self.assertEqual(tied, 1)
        self.assertEqual(ids.tolist(), [list(range(25))])

    def test_nonfinite_fails(self):
        logits = torch.zeros(1, 30)
        logits[0, 2] = float("nan")
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            scoring.deterministic_topk_ids(logits, k=25)


class StabilityTests(unittest.TestCase):
    def test_jaccard_known_overlap(self):
        left = np.arange(25, dtype=np.int32)[None, :]
        right = np.arange(15, 40, dtype=np.int32)[None, :]
        observed = scoring.jaccard_topk(left, right)
        self.assertAlmostEqual(float(observed[0]), 10 / 40)

    def test_registered_median_order(self):
        base = np.arange(25, dtype=np.int32)
        disjoint = np.arange(25, 50, dtype=np.int32)
        a = np.tile(base, (2, 27, 3, 1))
        b = a.copy()
        b[0, :, 0] = disjoint
        b[1, :, :2] = disjoint
        _, row_layer, by_layer = scoring.row_layer_and_layer_medians(a, b)
        np.testing.assert_allclose(row_layer[0], 1.0)
        np.testing.assert_allclose(row_layer[1], 0.0)
        np.testing.assert_allclose(by_layer, 0.5)

    def test_band_scan_ties_shortest_then_lowest_start(self):
        values = np.zeros(27)
        winner = scoring.band_scan(values)
        self.assertEqual((winner["start"], winner["end"]), (14, 17))
        self.assertEqual(winner["length"], 4)
        self.assertEqual(winner["score"], 0.0)
        values[14:20] = 0.4
        winner = scoring.band_scan(values)
        self.assertEqual((winner["start"], winner["end"]), (14, 17))
        self.assertEqual(winner["score"], 0.4)

    def test_band_scan_has_174_candidates_and_registered_oracle(self):
        self.assertEqual(len(scoring.all_candidate_bands(27)), 174)
        values = np.full(27, 0.05)
        values[14:21] = [0.14, 0.20, 0.30, 0.40, 0.13, 0.50, 0.60]
        winner = scoring.band_scan(values)
        self.assertEqual((winner["start"], winner["end"]), (14, 17))
        self.assertAlmostEqual(winner["score"], 0.14)

    def test_strict_p95_and_finite_sample_p(self):
        null = np.arange(1000) / 1000
        self.assertEqual(scoring.higher_quantile(null, 0.95), 0.95)
        self.assertAlmostEqual(scoring.empirical_upper_p(null, 0.951), 50 / 1001)
        self.assertAlmostEqual(scoring.empirical_upper_p(null, 0.950), 51 / 1001)

    def test_permutation_null_is_reproducible(self):
        a = np.empty((2, 27, 3, 25), dtype=np.int64)
        b = np.empty_like(a)
        for row in range(2):
            for layer in range(27):
                for pos in range(3):
                    offset = (row * 7 + layer * 3 + pos) % 50
                    a[row, layer, pos] = (np.arange(25) + offset) % 100
                    b[row, layer, pos] = a[row, layer, pos]
        first = scoring.stability_permutation_null(
            a, b, vocab_size=100, seed=11, n_perm=3, device="cpu"
        )
        second = scoring.stability_permutation_null(
            a, b, vocab_size=100, seed=11, n_perm=3, device="cpu"
        )
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_permutation_jaccard_uses_float64_like_observed(self):
        permutation = np.random.Generator(np.random.PCG64(11)).permutation(50)
        inverse = np.argsort(permutation)
        a = np.tile(np.arange(25, dtype=np.int64), (1, 27, 1, 1))
        mapped_targets = np.asarray([0, *range(25, 49)], dtype=np.int64)
        b_set = inverse[mapped_targets]
        b = np.tile(b_set, (1, 27, 1, 1))
        scan, layers = scoring.stability_permutation_null(
            a, b, vocab_size=50, seed=11, n_perm=1, device="cpu"
        )
        expected = np.float64(1) / np.float64(49)
        self.assertEqual(scan[0], expected)
        np.testing.assert_array_equal(layers[0], np.full(27, expected))


class ExternalTests(unittest.TestCase):
    def setUp(self):
        self.values = np.empty((3, 27, 25), dtype=np.int32)
        for item in range(3):
            self.values[item] = np.arange(item * 25, item * 25 + 25)
        self.labels = [[0], [25], [50]]

    def test_pass_at_25(self):
        self.assertEqual(
            scoring.pass_at_25(self.values, self.labels, layers=range(27)), 1.0
        )
        changed = self.values.copy()
        changed[1] += 100
        self.assertAlmostEqual(
            scoring.pass_at_25(changed, self.labels, layers=[17]), 2 / 3
        )

    def test_whole_label_permutation_reproducible(self):
        first = scoring.external_permutation_null(
            self.values,
            self.labels,
            rng=np.random.Generator(np.random.PCG64(12)),
            n_perm=10,
        )
        second = scoring.external_permutation_null(
            self.values,
            self.labels,
            rng=np.random.Generator(np.random.PCG64(12)),
            n_perm=10,
        )
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_external_report_requires_all_checks_in_same_eval(self):
        top = {
            "fit_a": self.values,
            "fit_b": self.values,
            "merged": self.values,
            "logit": self.values,
        }
        null = np.zeros(1000)
        report = scoring.external_report(top, self.labels, null, null)
        self.assertTrue(report["qualifying_success"])
        self.assertTrue(all(report["qualifying_checks"].values()))

    def test_item_fraction_preserves_whole_label_list(self):
        values = np.zeros((3, 27, 25), dtype=np.int32)
        values[0] = np.arange(1, 26)
        values[1] = np.arange(3, 28)
        values[2] = np.arange(5, 30)
        labels = [[1, 2], [3], [4, 5]]
        self.assertAlmostEqual(scoring.pass_at_25(values, labels, layers=[17]), 5 / 6)


class NumericalTests(unittest.TestCase):
    def test_zero_denominator_contract(self):
        zero = torch.zeros(2, 2)
        self.assertEqual(runner.relative_frobenius(zero, zero), 0.0)
        self.assertEqual(runner.relative_frobenius(zero, torch.ones(2, 2)), float("inf"))

    def test_bitwise_identity_distinguishes_signed_zero(self):
        positive = torch.tensor([0.0], dtype=torch.float32)
        negative = torch.tensor([-0.0], dtype=torch.float32)
        self.assertTrue(torch.equal(positive, negative))
        self.assertFalse(runner.bitwise_float32_equal(positive, negative))


if __name__ == "__main__":
    unittest.main()
