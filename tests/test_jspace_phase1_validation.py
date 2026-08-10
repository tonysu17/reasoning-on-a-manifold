import unittest

import jspace_phase1_validate as validator


class EligibleItemMetadataTests(unittest.TestCase):
    def setUp(self):
        self.eligibility = {
            "evaluations": {
                "association": {
                    "items": [
                        {"name": "a-1", "item_eligible": True},
                        {"name": "a-skip", "item_eligible": False},
                        {"name": "a-2", "item_eligible": True},
                    ]
                },
                "typo": {
                    "items": [{"name": "t-1", "item_eligible": True}]
                },
                "multihop": {
                    "items": [
                        {"name": "m-2", "item_eligible": True},
                        {"name": "m-1", "item_eligible": True},
                    ]
                },
            }
        }

    def test_names_are_slug_local_ordered_and_filter_ineligible(self):
        self.assertEqual(
            validator.eligible_item_names_for(self.eligibility, "association"),
            ["a-1", "a-2"],
        )
        self.assertEqual(
            validator.eligible_item_names_for(self.eligibility, "typo"), ["t-1"]
        )
        self.assertEqual(
            validator.eligible_item_names_for(self.eligibility, "multihop"),
            ["m-2", "m-1"],
        )

    def test_order_or_name_corruption_is_not_equal(self):
        expected = validator.eligible_item_names_for(
            self.eligibility, "association"
        )
        self.assertNotEqual(list(reversed(expected)), expected)
        self.assertNotEqual(["a-1", "corrupt"], expected)


if __name__ == "__main__":
    unittest.main()
