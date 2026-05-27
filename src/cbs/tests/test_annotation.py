"""Unit tests for src/cbs/annotation.py (M1). Filled in at M1 implementation."""
import pytest


def test_annotation_module_importable() -> None:
    from src.cbs import annotation  # noqa: F401


def test_cbsresult_validates_fields() -> None:
    from src.cbs.schemas import CBSResult
    with pytest.raises(ValueError):
        CBSResult(tier=4, knowledge_domain="algebra", cross_domain=False,
                  rationale="x", confidence="high")
    with pytest.raises(ValueError):
        CBSResult(tier=1, knowledge_domain="bogus", cross_domain=False,
                  rationale="x", confidence="high")
    with pytest.raises(ValueError):
        CBSResult(tier=1, knowledge_domain="algebra", cross_domain=False,
                  rationale="x", confidence="medium-high")
    ok = CBSResult(tier=2, knowledge_domain="algebra", cross_domain=False,
                   rationale="x", confidence="medium")
    assert ok.tier == 2


@pytest.mark.skip(reason="Filled in at M1 (synthesis §M1.4).")
def test_annotate_sentence_mocked_sonnet() -> None:
    pass


@pytest.mark.skip(reason="Filled in at M1 (synthesis §M1.4).")
def test_dual_seed_kappa_smoke() -> None:
    pass
