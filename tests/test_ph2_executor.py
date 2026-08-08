"""Pre-spend checks for the Phase-2 executor (integrated plan Priority 3)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import ph2_manifest as pm  # noqa: E402
import ph2_executor as px  # noqa: E402


# ── decision table (prereg §8) ───────────────────────────────────────────────

BASE = dict(sensitivity_adequate=True, damage_ok=True, gate_pass=False, raw_retained=False,
            corrected_gate_pass=False, corrected_retained=False, refit_pass=False,
            refit_aligned=False, refit_misaligned_beyond_null=False, refit_recovered=False,
            repr_signal_above_null=False)


def _o(**kw):
    return {**BASE, **kw}


def test_retained():
    assert px.decide_outcome(_o(gate_pass=True, refit_aligned=True,
                                raw_retained=True)) == "retained"


def test_rescaled():
    assert px.decide_outcome(_o(corrected_gate_pass=True, refit_aligned=True,
                                corrected_retained=True)) == "rescaled"


def test_rotated():
    assert px.decide_outcome(_o(refit_pass=True, refit_misaligned_beyond_null=True,
                                refit_recovered=True)) == "rotated"


def test_decoupled_precedes_disabled():
    # representational signal remains, nothing causal recovers -> decoupled, not disabled
    assert px.decide_outcome(_o(repr_signal_above_null=True)) == "decoupled"


def test_disabled_requires_no_repr_signal():
    assert px.decide_outcome(_o()) == "disabled"


def test_downgrade_beats_everything():
    assert px.decide_outcome(_o(sensitivity_adequate=False, gate_pass=True,
                                refit_aligned=True, raw_retained=True)) == "downgraded"


def test_damage_stop():
    assert px.decide_outcome(_o(damage_ok=False, gate_pass=True, refit_aligned=True,
                                raw_retained=True)) == "damage_stop"


def test_rotated_not_claimed_without_recovery():
    out = px.decide_outcome(_o(refit_pass=True, refit_misaligned_beyond_null=True,
                               repr_signal_above_null=True))
    assert out == "decoupled"  # no refit_recovered -> falls through to decoupled


# ── missing-annotation policy ────────────────────────────────────────────────

def test_missing_annotation_is_unresolved_never_zero():
    rows = [{"task_id": "A", "annotations": []},
            {"task_id": "B", "annotations": None},
            {"task_id": "C", "annotations": [{"labels": ["backtracking"]},
                                             {"labels": []}]}]
    m = px.merge_annotations(rows)
    assert m["A"]["status"] == "unresolved" and "label_counts" not in m["A"]
    assert m["B"]["status"] == "unresolved"
    assert m["C"]["status"] == "ok" and m["C"]["label_counts"]["backtracking"] == 1
    assert m["C"]["n_sentences"] == 2


# ── manifest: ids, stratification, disjointness, immutability ────────────────

FAKE = [{"id": "x", "prompt": "p", "difficulty": "moderate", "category": c}
        for c in ["mathematical_logic"] * 10 + ["causal_reasoning"] * 10]


def test_assign_ids_start_and_format():
    t = pm.assign_ids(FAKE)
    math_ids = [x["id"] for x in t if x["category"] == "mathematical_logic"]
    assert math_ids[0] == "MATH_102" and math_ids[-1] == "MATH_111"
    caus_ids = [x["id"] for x in t if x["category"] == "causal_reasoning"]
    assert caus_ids[0] == "CAUS_102"


def test_disjointness_catches_collision():
    t = pm.assign_ids(FAKE)
    try:
        pm.verify_disjoint(t, {"MATH_105"})
        assert False, "collision not caught"
    except ValueError:
        pass
    pm.verify_disjoint(t, {"MATH_099", "CAUS_101"})  # real exclusion zone: no clash


def test_dynamic_start_clears_every_real_exclusion():
    exc = pm.build_exclusions()
    start = pm.dynamic_id_start(exc)
    # the dynamic start must sit strictly above every excluded suffix (corpus reaches _116
    # in some categories — the original static _102 assumption was wrong and caught here)
    assert start > max(int(i.rsplit("_", 1)[1]) for i in exc)
    fake = [{"id": "x", "prompt": "p", "difficulty": "d", "category": c}
            for c in ["spatial_reasoning"] * 10]
    pm.verify_disjoint(pm.assign_ids(fake, id_start=start), exc)


def test_hash_is_order_invariant():
    t = pm.assign_ids(FAKE)
    assert pm.manifest_hash(t) == pm.manifest_hash(list(reversed(t)))


def test_write_manifest_refuses_overwrite(tmp_path):
    t = pm.assign_ids(FAKE)
    p = tmp_path / "m.json"
    pm.write_manifest(t, set(), {"module": "fake"}, path=p)
    try:
        pm.write_manifest(t, set(), {"module": "fake"}, path=p)
        assert False, "overwrite not refused"
    except SystemExit:
        pass


def test_write_manifest_enforces_stratification(tmp_path):
    bad = pm.assign_ids(FAKE[:15])  # 10 + 5 -> unequal strata
    try:
        pm.write_manifest(bad, set(), {"module": "fake"}, path=tmp_path / "m.json")
        assert False, "stratification not enforced"
    except ValueError:
        pass


# ── provenance payload ───────────────────────────────────────────────────────

def test_provenance_has_all_contract_keys():
    p = px.build_provenance({"authorised": False})
    for k in px.PROVENANCE_KEYS:
        assert k in p
    assert p["amended"] == ["A1"] and p["authorised"] is False
