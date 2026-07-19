"""pt11b: harmful/benign refusal-pool builder — schema, balance, exclusions, pt11 compat."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pool(tmp_path_factory):
    out = tmp_path_factory.mktemp("pool") / "pool.json"
    pt11b = _load("pt11b", "pt11b_build_refusal_pool.py")
    pt11b.main(["--star1", str(ROOT / "data/safety_star1_sft.json"),
                "--tasks", str(ROOT / "data/tasks_final.json"),
                "--n-harmful", "40", "--n-benign", "40", "--seed", "7",
                "--out", str(out)])
    return out


def test_schema_balance_exclusions(pool):
    recs = json.load(open(pool))
    meta = json.load(open(str(pool) + ".meta.json"))
    assert isinstance(recs, list) and len(recs) == 80
    assert sum(r["harmful"] for r in recs) == 40
    assert all(set(r) >= {"id", "prompt", "harmful", "source", "category"} for r in recs)
    assert all(r["category"] != "Self-Harm" for r in recs)
    assert "Self-Harm" in meta["excluded_categories"]
    norm = {" ".join(r["prompt"].split()).lower() for r in recs}
    assert len(norm) == len(recs), "duplicate prompts"


def test_deterministic(pool, tmp_path):
    out2 = tmp_path / "pool2.json"
    pt11b = _load("pt11b_again", "pt11b_build_refusal_pool.py")
    pt11b.main(["--star1", str(ROOT / "data/safety_star1_sft.json"),
                "--tasks", str(ROOT / "data/tasks_final.json"),
                "--n-harmful", "40", "--n-benign", "40", "--seed", "7",
                "--out", str(out2)])
    assert json.load(open(pool)) == json.load(open(out2))


def test_pt11_loader_compat(pool):
    pt11 = _load("pt11_mod", "pt11_train_grpo.py")
    ds = pt11.build_dataset(str(pool), "refusal-format")
    assert len(ds) == 80
    assert "harmful" in ds.column_names
    assert sum(ds["harmful"]) == 40
