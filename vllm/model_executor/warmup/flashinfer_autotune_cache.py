# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer autotune cache helpers."""

import hashlib
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

import vllm.envs as envs
from vllm.compilation.caching import aot_compile_hash_factors

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def flashinfer_autotune_cache_hash(runner: "GPUModelRunner") -> str:
    factors = aot_compile_hash_factors(runner.vllm_config)
    return hashlib.sha256(str(factors).encode()).hexdigest()


def resolve_flashinfer_autotune_file(runner: "GPUModelRunner") -> Path:
    override_dir = envs.VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR
    if override_dir:
        root = Path(override_dir).expanduser()
    else:
        from flashinfer.jit import env as flashinfer_jit_env

        flashinfer_workspace = flashinfer_jit_env.FLASHINFER_WORKSPACE_DIR
        root = (
            Path(envs.VLLM_CACHE_ROOT)
            / "flashinfer_autotune_cache"
            / flashinfer_workspace.parent.name
            / flashinfer_workspace.name
        )

    output_dir = root / flashinfer_autotune_cache_hash(runner)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "autotune_configs.json"


def sync_flashinfer_autotune_cache(
    runner: "GPUModelRunner",
    group: "GroupCoordinator",
) -> None:
    cache = None
    cache_path = None
    if runner.vllm_config.kernel_config.enable_flashinfer_autotune:
        from vllm.platforms import current_platform
        from vllm.utils.flashinfer import has_flashinfer

        if has_flashinfer() and current_platform.has_device_capability(90):
            cache_path = resolve_flashinfer_autotune_file(runner)
            if group.rank_in_group == 0:
                cache = cache_path.read_bytes() if cache_path.exists() else b""

    cache = group.broadcast_object(cache)
    if cache == b"":
        raise RuntimeError("FlashInfer autotune cache is unavailable")
    if cache is not None and group.rank_in_group != 0:
        from flashinfer.autotuner import AutoTuner

        assert cache_path is not None
        with tempfile.NamedTemporaryFile() as f:
            f.write(cache)
            f.flush()
            if not AutoTuner.get().load_configs(f.name):
                raise RuntimeError("FlashInfer autotune cache is incompatible")


def write_flashinfer_autotune_cache(cache_path: Path, contents: bytes) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=cache_path.parent, suffix=".tmp", prefix=f".{cache_path.name}."
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(contents)
        os.replace(tmp_path, cache_path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_path)
        raise
