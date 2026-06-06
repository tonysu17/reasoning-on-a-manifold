"""Integration test crossing src/cbs/matching and src/cbs/trajectory (AUDIT.md §5).

Concern raised in the audit: matching builds success/failure sentence IDs while
trajectory builds its own sentence IDs and the activations_lookup keys — if the
two used different indexing conventions (filtered-list vs full-array position),
matched-pair lookups would silently miss-join and pairs would be dropped.

Reading the code, BOTH use f"{chain_id}:{full_array_span_index}". This test pins
that consistency so a future edit to either module can't reintroduce the drift.
"""

import numpy as np

from src.cbs.matching import _tier_spans
from src.cbs.trajectory import build_trajectory


def test_matching_and_trajectory_use_same_sentence_id_convention(tmp_path):
    chain = {"task_id": "c0", "annotations": [
        {"label": "deduction"},                        # full-array i=0 (not a target)
        {"label": "adding-knowledge", "cbs_tier": 3},  # i=1
        {"label": "deduction"},                        # i=2
        {"label": "adding-knowledge", "cbs_tier": 1},  # i=3
    ]}
    # Two adding-knowledge spans -> two activation rows, in build_row_index order.
    acts = {"adding-knowledge": np.arange(16, dtype=np.float32).reshape(2, 8)}

    traj = build_trajectory(chain, tmp_path, layer=27,
                            activations=acts, target_behaviours=("adding-knowledge",))

    # matching's success_sentence_id is f"{_chain_id}:{_sentence_idx}" with the
    # FULL-array index stored by _tier_spans.
    t3 = _tier_spans(chain, tier=3)
    matching_id = f"{t3[0]['_chain_id']}:{t3[0]['_sentence_idx']}"
    assert matching_id == "c0:1"

    # The same id must be resolvable against the trajectory's sentence IDs
    # (which is what the runner uses to build activations_lookup).
    assert traj.sentence_ids == ["c0:1", "c0:3"]
    assert matching_id in traj.sentence_ids
