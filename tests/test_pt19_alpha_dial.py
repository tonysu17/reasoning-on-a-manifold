import pytest
import torch
from safetensors.torch import load_file, save_file

from pt19_alpha_dial import (
    dry_run_probe,
    interpolate_state_dict,
    interpolate_tensor,
    load_manifest,
    merge_one_alpha,
    parse_alphas,
)


def tiny_state_dicts(seed: int = 20260802):
    generator = torch.Generator().manual_seed(seed)
    base = {
        "model.embed.weight": torch.randn(7, 5, generator=generator),
        "model.layers.0.mlp.weight": torch.randn(5, 5, generator=generator),
        "model.norm.weight": torch.randn(5, generator=generator),
    }
    target = {key: value + torch.randn(value.shape, generator=generator)
              for key, value in base.items()}
    return base, target


def test_alpha_zero_reproduces_base_exactly():
    base, target = tiny_state_dicts()
    merged = interpolate_state_dict(base, target, 0.0)
    assert merged.keys() == base.keys()
    assert all(torch.equal(merged[key], base[key]) for key in base)


def test_alpha_one_reproduces_target_exactly():
    base, target = tiny_state_dicts()
    merged = interpolate_state_dict(base, target, 1.0)
    assert merged.keys() == target.keys()
    assert all(torch.equal(merged[key], target[key]) for key in target)


def test_linearity_on_probe_tensor():
    base, target = tiny_state_dicts()
    key = "model.layers.0.mlp.weight"
    low = interpolate_tensor(base[key], target[key], 0.25)
    high = interpolate_tensor(base[key], target[key], 0.75)
    expected = 0.5 * (target[key] - base[key])
    assert torch.allclose(high - low, expected, rtol=1e-6, atol=1e-6)


def test_key_mismatch_is_rejected():
    base, target = tiny_state_dicts()
    target.pop("model.norm.weight")
    with pytest.raises(ValueError, match="tensor-key mismatch"):
        interpolate_state_dict(base, target, 0.5)


def test_alpha_grid_rejects_extrapolation_and_duplicates():
    assert parse_alphas([0.25, 0.5, 0.75, 1.0]) == (0.25, 0.5, 0.75, 1.0)
    with pytest.raises(ValueError, match="within"):
        parse_alphas([-0.25])
    with pytest.raises(ValueError, match="duplicate"):
        parse_alphas([0.5, 0.5])


def test_dry_run_validates_exactly_three_tiny_safetensors(tmp_path):
    base, target = tiny_state_dicts()
    base_dir, target_dir = tmp_path / "base", tmp_path / "target"
    base_dir.mkdir()
    target_dir.mkdir()
    save_file(base, base_dir / "model.safetensors")
    save_file(target, target_dir / "model.safetensors")

    report = dry_run_probe(
        load_manifest(base_dir), load_manifest(target_dir), (0.25, 0.5, 0.75, 1.0))
    assert report["status"].startswith("dry-run only")
    assert len(report["probe_tensors"]) == 3


def test_shard_streaming_merge_writes_expected_tiny_checkpoint(tmp_path):
    base, target = tiny_state_dicts()
    base_dir, target_dir = tmp_path / "base", tmp_path / "target"
    base_dir.mkdir()
    target_dir.mkdir()
    save_file(base, base_dir / "model.safetensors")
    save_file(target, target_dir / "model.safetensors")

    output = merge_one_alpha(
        load_manifest(base_dir), load_manifest(target_dir), 0.5, tmp_path / "out")
    merged = load_file(output / "model.safetensors")
    expected = interpolate_state_dict(base, target, 0.5)
    assert all(torch.equal(merged[key], expected[key]) for key in expected)
