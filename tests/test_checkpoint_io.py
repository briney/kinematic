"""Tests for checkpoint loading dispatch helpers."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("boltz")

from safetensors.torch import save_file

from kinematic.model.checkpoint_io import (
    find_latest_step_checkpoint,
    find_model_weights_file,
    has_unresolved_step_placeholder,
    load_checkpoint_file,
    load_model_state_dict,
    resolve_checkpoint_path,
)


def test_find_model_weights_file_from_directory(tmp_path) -> None:
    ckpt_dir = tmp_path / "step_1000"
    ckpt_dir.mkdir()
    pt_file = ckpt_dir / "pytorch_model.bin"
    torch.save({"weight": torch.tensor([1.0])}, pt_file)

    out = find_model_weights_file(ckpt_dir)
    assert out == pt_file


def test_load_checkpoint_file_torch_dispatch(tmp_path) -> None:
    path = tmp_path / "model.bin"
    expected = {"weight": torch.tensor([1.0, 2.0, 3.0])}
    torch.save(expected, path)

    loaded = load_checkpoint_file(path)
    assert torch.allclose(loaded["weight"], expected["weight"])


def test_load_checkpoint_file_safetensors_dispatch(tmp_path) -> None:
    path = tmp_path / "model.safetensors"
    expected = {"weight": torch.tensor([3.0, 2.0, 1.0])}
    save_file(expected, str(path))

    loaded = load_checkpoint_file(path)
    assert torch.allclose(loaded["weight"], expected["weight"])


def test_load_model_state_dict_unwraps_state_dict(tmp_path) -> None:
    path = tmp_path / "model.ckpt"
    inner = {"weight": torch.tensor([5.0])}
    torch.save({"state_dict": inner}, path)

    loaded = load_model_state_dict(path)
    assert torch.allclose(loaded["weight"], inner["weight"])


def test_has_unresolved_step_placeholder_detects_placeholder() -> None:
    assert has_unresolved_step_placeholder("checkpoints/phase1/step_XXXXX")
    assert has_unresolved_step_placeholder("checkpoints/phase1/step_xxx/model.safetensors")
    assert not has_unresolved_step_placeholder("checkpoints/phase1/step_1000")


def test_find_latest_step_checkpoint_selects_max_numeric_step(tmp_path) -> None:
    (tmp_path / "step_100").mkdir()
    (tmp_path / "step_250").mkdir()
    (tmp_path / "step_42").mkdir()
    (tmp_path / "latest").mkdir()

    latest = find_latest_step_checkpoint(tmp_path)
    assert latest == tmp_path / "step_250"


def test_resolve_checkpoint_path_auto_resolves_latest_step(tmp_path) -> None:
    step_10 = tmp_path / "step_10"
    step_25 = tmp_path / "step_25"
    step_10.mkdir()
    step_25.mkdir()

    target_file = step_25 / "model.safetensors"
    target_file.write_bytes(b"placeholder")

    resolved = resolve_checkpoint_path(
        tmp_path / "step_XXXXX" / "model.safetensors",
        auto_resolve_latest=True,
    )
    assert resolved == target_file


def test_resolve_checkpoint_path_raises_when_placeholder_cannot_resolve(tmp_path) -> None:
    with pytest.raises(ValueError, match="step_\\*"):
        resolve_checkpoint_path(
            tmp_path / "step_XXXXX",
            auto_resolve_latest=True,
        )
