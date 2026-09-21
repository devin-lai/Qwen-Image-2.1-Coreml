"""Regression checks for cache reuse and invalid generation inputs."""

import numpy as np
import pytest
import torch

from qwen_image_coreml import generate
from qwen_image_coreml import runtime as runtime_module
from qwen_image_coreml.scheduler import FlowMatchEulerScheduler


@pytest.fixture
def cached_prefix(tmp_path, monkeypatch):
    model = tmp_path / "models" / "QwenImage21_Embed.mlpackage" / "model.mlmodel"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model-v1")
    calls = []

    def compute(*args):
        calls.append(args)
        return np.full((2, 3), len(calls), dtype=np.float16), np.zeros((2, 3), dtype=np.float16)

    monkeypatch.setattr(runtime_module, "compute_prefix", compute)
    inputs = {
        "models_dir": tmp_path / "models",
        "chunks": 4,
        "compute_units": "cpu_and_gpu",
        "embeds": torch.zeros(1, 2, 4),
        "bias": torch.zeros(1, 1, 2, 2),
        "cos": torch.ones(2, 4),
        "sin": torch.zeros(2, 4),
        "cache_path": tmp_path / "prefix-cache",
    }
    return inputs, calls, model


def test_prefix_cache_reuses_matching_inputs(cached_prefix):
    inputs, calls, _ = cached_prefix
    first = runtime_module.prefix_cache(**inputs)
    second = runtime_module.prefix_cache(**inputs)
    assert len(calls) == 1
    assert inputs["cache_path"].is_file()
    np.testing.assert_array_equal(first[0], second[0])


@pytest.mark.parametrize("changed", ["embeds", "bias", "cos", "sin", "compute_units", "model"])
def test_prefix_cache_recomputes_when_inputs_change(cached_prefix, changed):
    inputs, calls, model = cached_prefix
    runtime_module.prefix_cache(**inputs)
    if changed == "model":
        model.write_bytes(b"replacement-model")
    elif changed == "compute_units":
        inputs[changed] = "cpu_only"
    else:
        inputs[changed] = inputs[changed] + 1
    actual, _ = runtime_module.prefix_cache(**inputs)
    assert len(calls) == 2
    np.testing.assert_array_equal(actual, np.full((2, 3), 2))


@pytest.mark.parametrize("legacy", [True, False])
def test_prefix_cache_rebuilds_legacy_or_corrupt_files(cached_prefix, legacy):
    inputs, calls, _ = cached_prefix
    path = inputs["cache_path"]
    if legacy:
        with path.open("wb") as handle:
            np.savez(handle, cache_k=np.ones(1), cache_v=np.ones(1))
    else:
        path.write_bytes(b"interrupted archive")
    runtime_module.prefix_cache(**inputs)
    assert len(calls) == 1


@pytest.mark.parametrize("steps", [-1, 0, 1])
def test_scheduler_rejects_undefined_short_schedules(steps):
    with pytest.raises(ValueError, match="at least 2"):
        FlowMatchEulerScheduler.make(steps, 4096)


@pytest.mark.parametrize("length", [0, 65])
def test_prompt_length_fails_before_loading_models(tmp_path, length):
    with pytest.raises(ValueError, match="1 to 64 tokens"):
        generate(torch.zeros(1, length, 4096), models_dir=tmp_path)


@pytest.mark.parametrize("shape", [(2, 31, 4096), (1, 31, 128), (31, 4096)])
def test_prompt_shape_fails_before_loading_models(tmp_path, shape):
    with pytest.raises(ValueError, match="shape"):
        generate(torch.zeros(shape), models_dir=tmp_path)


def test_unsupported_resolution_fails_before_loading_models(tmp_path):
    with pytest.raises(ValueError, match="1024x1024"):
        generate(torch.zeros(1, 31, 4096), height=512, models_dir=tmp_path)
