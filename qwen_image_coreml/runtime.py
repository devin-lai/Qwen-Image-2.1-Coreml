"""Loading and feeding the converted Core ML packages.

Multifunction models declare inputs on each function rather than on the
package. Feeds use the declared array dtypes and contiguous storage; strided
arrays can cause assertions in the Metal backend.
"""

from __future__ import annotations

import gc
import hashlib
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import torch

COMPUTE_UNITS = ("cpu_and_gpu", "all", "cpu_only", "cpu_and_ne")


def load_mlmodel(path: Path, compute_units: str = "cpu_and_gpu", function_name: str | None = None):
    """Open one `.mlpackage`, optionally selecting one function of a merged one."""
    import coremltools as ct

    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing — run `python download_models.py` to fetch the packages"
        )
    units = {
        "all": ct.ComputeUnit.ALL,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
    }[compute_units]
    kwargs = {"function_name": function_name} if function_name else {}
    return ct.models.MLModel(str(path), compute_units=units, **kwargs)


def _description(mlmodel):
    """The IO description, for a single-function or a multifunction package."""
    desc = mlmodel.get_spec().description
    functions = list(getattr(desc, "functions", []))
    if not functions:
        return desc
    wanted = getattr(mlmodel, "function_name", None) or desc.defaultFunctionName
    for function in functions:
        if function.name == wanted:
            return function
    return functions[0]


def input_names(mlmodel) -> list[str]:
    return [spec.name for spec in _description(mlmodel).input]


def output_names(mlmodel) -> list[str]:
    return [spec.name for spec in _description(mlmodel).output]


def n_inputs(mlmodel) -> int:
    return len(input_names(mlmodel))


def _feature_dtypes() -> dict:
    """Core ML's `ArrayFeatureType.ArrayDataType` enum -> numpy.

    Read from the proto rather than hardcoded: the values are bit-packed
    (`0x10000 | bits`), so FLOAT16 is 65552 and FLOAT32 is 65568.
    """
    from coremltools import proto

    enum = proto.FeatureTypes_pb2.ArrayFeatureType.ArrayDataType
    return {
        enum.FLOAT16: np.float16,
        enum.FLOAT32: np.float32,
        enum.DOUBLE: np.float64,
        enum.INT32: np.int32,
    }


def declared_dtypes(mlmodel) -> dict:
    out = {}
    for spec in _description(mlmodel).input:
        if spec.type.WhichOneof("Type") == "multiArrayType":
            out[spec.name] = _feature_dtypes().get(
                spec.type.multiArrayType.dataType, np.float32
            )
    return out


def _prepare(value, dtype):
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    return np.ascontiguousarray(array, dtype=dtype if dtype is not None else array.dtype)


def as_feed(mlmodel, *values) -> dict:
    """Positional values -> the model's named inputs, in its declared dtypes."""
    names = input_names(mlmodel)
    dtypes = declared_dtypes(mlmodel)
    return {
        name: _prepare(value, dtypes.get(name))
        for name, value in zip(names, values, strict=True)
    }


def predict(mlmodel, *values):
    out = mlmodel.predict(as_feed(mlmodel, *values))
    return [out[name] for name in output_names(mlmodel)]


def open_chunk(models_dir: Path, index: int, chunks: int, mode: str, compute_units: str):
    """Chunk `index`'s `prefix` or `decode` model, merged or single-function.

    The shipped packages are merged — the two functions share one copy of the
    weights — but an unmerged pair is equally usable, so try the merged name
    first and fall back rather than forcing one packaging choice on the caller.
    """
    merged = models_dir / f"QwenImage21_Blocks_{index}of{chunks}.mlpackage"
    if merged.exists():
        return load_mlmodel(merged, compute_units, function_name=mode)
    single = models_dir / f"QwenImage21_{mode.capitalize()}_{index}of{chunks}.mlpackage"
    if not single.exists():
        raise FileNotFoundError(f"neither {merged.name} nor {single.name} exists in {models_dir}")
    return load_mlmodel(single, compute_units)


class Denoiser:
    """The converted denoiser: one `Embed` model plus N decode chunks.

    Everything that does not change between denoising steps — the RoPE tables,
    the attention bias, and each chunk's slice of the KV cache — is converted to
    the exact numpy dtype each model declares **once**, at bind time. Rebuilding
    them per step costs ~130 MB of fp32 -> fp16 casts on every one of the 40
    iterations, for tensors that are identical each time.
    """

    def __init__(self, models_dir: Path, chunks: int = 4, compute_units: str = "cpu_and_gpu"):
        from .layout import time_proj

        self._time_proj = time_proj
        self.embed = load_mlmodel(models_dir / "QwenImage21_Embed.mlpackage", compute_units)
        self.chunks = [
            open_chunk(models_dir, i, chunks, "decode", compute_units) for i in range(chunks)
        ]
        self.n = chunks
        self._static: list[list] | None = None

    def bind(self, cos, sin, cache_k, cache_v, bias) -> None:
        per_chunk = np.asarray(cache_k).shape[0] // self.n
        self._static = []
        for i, chunk in enumerate(self.chunks):
            dtypes = declared_dtypes(chunk)
            names = input_names(chunk)
            values = [
                cos,
                sin,
                np.asarray(cache_k)[i * per_chunk : (i + 1) * per_chunk],
                np.asarray(cache_v)[i * per_chunk : (i + 1) * per_chunk],
            ]
            if n_inputs(chunk) == 8:
                values.append(bias)
            # names[:3] are the per-step inputs (hidden_states, modulation, temb).
            self._static.append(
                [
                    _prepare(value, dtypes.get(name))
                    for name, value in zip(names[3:], values, strict=True)
                ]
            )

    def step(self, latents, timestep) -> torch.Tensor:
        if self._static is None:
            raise RuntimeError("call bind() before step()")
        # time_proj runs on the host, in fp32: see layout.time_proj.
        temb, modulation = predict(self.embed, self._time_proj(timestep))
        hidden = latents
        for chunk, static in zip(self.chunks, self._static, strict=True):
            hidden = predict(chunk, hidden, modulation, temb, *static)[0]
        return torch.from_numpy(np.asarray(hidden)).float()


def compute_prefix(models_dir: Path, chunks: int, compute_units: str, embeds, bias, cos, sin):
    """Prompt embeddings -> the per-layer KV cache, through the prefix functions.

    Chunks are opened and dropped one at a time. Holding four prefix models and
    four decode models at once is 26 GB of resident weights, and Core ML aborts
    inside MetalPerformanceShadersGraph rather than failing an allocation.
    """
    embed = load_mlmodel(models_dir / "QwenImage21_Embed.mlpackage", compute_units)
    from .layout import time_proj

    _, modulation0 = predict(embed, time_proj(torch.zeros(1)))
    del embed
    gc.collect()

    keys, values = [], []
    hidden = embeds
    for i in range(chunks):
        model = open_chunk(models_dir, i, chunks, "prefix", compute_units)
        args = [hidden, modulation0, cos, sin]
        if n_inputs(model) == 5:
            args.append(bias)
        hidden, k, v = predict(model, *args)
        keys.append(np.asarray(k))
        values.append(np.asarray(v))
        del model
        gc.collect()
    return np.concatenate(keys, 0), np.concatenate(values, 0)


def _prefix_key(models_dir, chunks, compute_units, *inputs):
    """Identify prefix inputs and the local model files without rehashing weights."""
    models_dir = Path(models_dir).resolve()
    packages = [models_dir / "QwenImage21_Embed.mlpackage"]
    for index in range(chunks):
        merged = models_dir / f"QwenImage21_Blocks_{index}of{chunks}.mlpackage"
        single = models_dir / f"QwenImage21_Prefix_{index}of{chunks}.mlpackage"
        packages.append(merged if merged.exists() else single)
    files = []
    for package in packages:
        for path in sorted(package.rglob("*")):
            if path.is_file() and (path.name == "Manifest.json" or path.suffix in {".mlmodel", ".bin"}):
                stat = path.stat()
                files.append((str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    try:
        runtime_version = version("coremltools")
    except PackageNotFoundError:
        runtime_version = "unavailable"
    digest = hashlib.sha256(json.dumps(
        [1, chunks, compute_units, runtime_version, platform.mac_ver()[0], platform.machine(), files]
    ).encode())
    for value in inputs:
        array = _prepare(value, None)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def prefix_cache(
    models_dir: Path, chunks: int, compute_units: str, embeds, bias, cos, sin, cache_path
):
    """Reuse KV tensors only when the prompt, layout, models, and runtime match."""
    key = _prefix_key(models_dir, chunks, compute_units, embeds, bias, cos, sin) if cache_path else None
    if cache_path is not None and Path(cache_path).exists():
        try:
            with np.load(cache_path) as data:
                if "cache_key" in data and str(data["cache_key"].item()) == key:
                    keys, values = data["cache_k"], data["cache_v"]
                    print(f"  reusing {Path(cache_path).name}")
                    return keys, values
        except (OSError, ValueError, KeyError, EOFError, BadZipFile):
            pass
        print(f"  rebuilding {Path(cache_path).name}")
    keys, values = compute_prefix(models_dir, chunks, compute_units, embeds, bias, cos, sin)
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # A file handle preserves an explicitly supplied name without adding .npz.
        with cache_path.open("wb") as handle:
            np.savez(handle, cache_k=keys, cache_v=values, cache_key=np.array(key))
        print(f"  saved {cache_path.name}  ({cache_path.stat().st_size / 1e6:.0f} MB)")
    return keys, values
