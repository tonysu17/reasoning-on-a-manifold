"""Regression tests for src/config.py — the single source of truth that
replaced the divergent per-runner MODELS / STEERING_LAYERS dicts.

These guard against (a) config.yaml silently not being loaded again, and
(b) the steering-layer registry drifting away from the YAML (which is how
05b ended up missing the baseline model).
"""

import yaml

from src.config import (
    load_config, STEERING_LAYERS, MODELS, SEED, TARGET_BEHAVIOURS, CONFIG_PATH,
    PEAK_LAYERS, provenance,
)


def test_config_file_exists_and_loads():
    assert CONFIG_PATH.exists()
    cfg = load_config()
    assert "models" in cfg and "project" in cfg


def test_all_configured_models_present_in_registry():
    """Every model in the YAML must appear in the derived registry — no model
    may be silently dropped (the 05b bug)."""
    cfg = load_config()
    yaml_short_names = {m["short_name"] for m in cfg["models"].values()
                        if isinstance(m, dict) and "short_name" in m}
    assert yaml_short_names == set(STEERING_LAYERS) == set(MODELS)


def test_steering_layers_match_yaml():
    cfg = load_config()
    for spec in cfg["models"].values():
        sn = spec["short_name"]
        assert STEERING_LAYERS[sn] == spec["steering_layer"]


def test_baseline_model_present():
    """The QwenMath-1.5B baseline must be in the registry with a steering layer
    (its omission from 05b was a latent wrong-layer bug)."""
    assert "QwenMath-1.5B" in STEERING_LAYERS
    assert STEERING_LAYERS["QwenMath-1.5B"] == 27


def test_seed_is_from_config():
    assert SEED == load_config()["project"]["seed"]


def test_target_behaviours_are_the_four():
    assert set(TARGET_BEHAVIOURS) == {
        "backtracking", "uncertainty-estimation", "example-testing", "adding-knowledge",
    }


def test_peak_layers_from_config():
    cfg = load_config()
    assert PEAK_LAYERS == {k: int(v) for k, v in cfg["analysis"]["peak_layers"].items()}
    assert PEAK_LAYERS["adding-knowledge"] == 17


def test_provenance_has_git_and_seed():
    p = provenance()
    assert p["seed"] == SEED
    assert "git_commit" in p and isinstance(p["git_dirty"], bool)


def test_provenance_captures_args_and_input_hashes(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("hello")
    p = provenance(args={"model": "R1-1.5B", "k": 5}, inputs=[str(f)])
    assert p["args"]["model"] == "R1-1.5B" and p["args"]["k"] == 5
    assert isinstance(p["input_sha256"][str(f)], str) and len(p["input_sha256"][str(f)]) == 16


def test_backup_existing_copies_and_preserves_original(tmp_path):
    from src.config import backup_existing
    f = tmp_path / "out.json"
    f.write_text("v1")
    bak = backup_existing(f)
    assert bak is not None and bak.read_text() == "v1"
    assert f.exists()                      # copy, not move (resume still reads it)
    f.write_text("v2")
    assert bak.read_text() == "v1"         # backup is an independent snapshot


def test_backup_existing_missing_returns_none(tmp_path):
    from src.config import backup_existing
    assert backup_existing(tmp_path / "absent.json") is None


def test_model_cli_resolver():
    from src.config import MODELS_BY_CLI, model_tuple, model_dict
    cfg = load_config()
    expected = {m["cli_alias"] for m in cfg["models"].values() if "cli_alias" in m}
    assert set(MODELS_BY_CLI) == expected
    # 04/07 tuple shape
    mid, short, dtype = model_tuple("1.5b")
    assert short == "R1-1.5B" and dtype == "float16"
    # 02 dict shape
    d = model_dict("qwen-math-1.5b")
    assert d["short"] == "QwenMath-1.5B" and set(d) == {"id", "short", "dtype"}


def test_runners_import_shared_steering_layers():
    """The migrated runners must expose the SAME object as src.config, proving
    the duplication is gone."""
    import importlib
    from src.config import STEERING_LAYERS as canonical
    for mod_name in ("05_pca_analysis", "05b_geometric_diagnostics", "06_build_steering"):
        mod = importlib.import_module(mod_name)
        assert mod.STEERING_LAYERS == canonical
        assert "QwenMath-1.5B" in mod.STEERING_LAYERS
