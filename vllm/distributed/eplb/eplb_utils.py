# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility functions for EPLB (Expert Parallel Load Balancing)."""

import contextlib
import os
import threading

import torch

from vllm.config import ParallelConfig
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


@triton.jit(
    do_not_specialize=["num_physical_experts"],
    do_not_specialize_on_alignment=["window"],
)
def _record_expert_load(
    load_pass,
    mapping,
    window,
    num_physical_experts,
    PHYSICAL_STRIDE: tl.constexpr,
    LAYER_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    layer = tl.program_id(1).to(tl.int64)
    slots = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = slots < num_physical_experts
    offsets = layer * PHYSICAL_STRIDE + slots
    experts = tl.load(mapping + offsets, valid, other=-1)
    counts = tl.load(load_pass + offsets, valid, other=0)
    experts = tl.where(experts < 0, LAYER_STRIDE - 1, experts)
    tl.atomic_add(window + layer * LAYER_STRIDE + experts, counts, valid, sem="relaxed")
    tl.store(load_pass + offsets, 0, valid)


def record_expert_load(
    load_pass: torch.Tensor,
    mapping: torch.Tensor,
    window: torch.Tensor,
    window_step: int,
) -> None:
    """Record logical counts and clear counters; physical row strides must match."""
    window = window[window_step]
    window.zero_()
    if not load_pass.is_cuda:
        window.scatter_add_(
            -1, mapping.masked_fill(mapping < 0, window.shape[-1] - 1), load_pass
        )
        load_pass.zero_()
        return
    block_size = 256
    grid = (triton.cdiv(load_pass.shape[1], block_size), load_pass.shape[0])
    _record_expert_load[grid](
        load_pass,
        mapping,
        window,
        load_pass.shape[1],
        load_pass.stride(0),
        window.stride(0),
        block_size,
    )


@contextlib.contextmanager
def device_stream(stream: torch.Stream | None):
    """Platform-agnostic context manager that activates *stream* as the
    current accelerator stream for the duration of the ``with`` block.
    A no-op when *stream* is ``None``."""
    if stream is None:
        yield
        return
    prev = torch.accelerator.current_stream()
    torch.accelerator.set_stream(stream)
    try:
        yield
    finally:
        torch.accelerator.set_stream(prev)


class CpuGpuEvent:
    """Combines a CUDA event with a CPU threading event to enforce record->wait
    ordering across two threads.

    This class is designed for exactly two threads: one producer that calls
    record() and one consumer that calls wait(). Using it with more than two
    threads is not supported and will produce undefined behavior.

    CUDA events alone are insufficient for cross-thread synchronization because
    waiting on an unrecorded CUDA event is a no-op. The wait will return
    immediately instead of blocking. This class adds a threading.Event so
    that the waiting thread blocks on the CPU side until record() is called, at
    which point the CUDA event is guaranteed to be in-flight and event.wait() will
    correctly synchronize the GPU stream.
    """

    def __init__(self):
        self._event = torch.Event()
        self._recorded = threading.Event()

    def wait(self, stream: torch.Stream | None = None):
        """Blocks the calling thread until record finishes. Used to guarantee that the
        record kernel is called before wait.

        Should only be called by the Async Eplb thread.
        """
        self._recorded.wait()
        self._event.wait(stream)
        self._recorded.clear()

    def record(self, stream: torch.Stream | None = None):
        """Unblocks the waiting thread after calling event.record().

        Should only be called by the main thread.
        """
        if self._recorded.is_set():
            raise RuntimeError(
                "CpuGpuEvent.record() called before the previous event was "
                "consumed by wait()"
            )
        self._event = torch.Event()
        self._event.record(stream)
        self._recorded.set()


def override_envs_for_eplb(
    parallel_config: ParallelConfig,
    moe_backend: str | None = None,
) -> None:
    """Override environment variables for EPLB when specific conditions are met.

    Args:
        parallel_config: The parallel configuration object.
        moe_backend: The configured MoE backend (e.g. ``deep_gemm_mega_moe``).

    """
    is_data_parallel = parallel_config.data_parallel_size > 1
    is_eplb_enabled = parallel_config.enable_eplb
    is_mega_moe = moe_backend == "deep_gemm_mega_moe"
    is_nccl_based_eplb_communicator = parallel_config.eplb_config.communicator in (
        "torch_nccl",
        "pynccl",
    )

    # Override NCCL_MAX_CTAS to avoid hangs when EPLB's NCCL weight exchange
    # contends with MoE backend's cooperative-launch on GPU SMs.
    #
    # DeepGEMM Mega MoE uses cooperative launch, which tries to reserve a
    # large fraction of the GPU's SMs. If those SMs are occupied by NCCL,
    # the cooperative launch blocks until enough SMs are freed, causing a
    # deadlock. Limiting NCCL occupancy via NCCL_MAX_CTAS leaves space for
    # the cooperative kernel to launch and complete.
    if (
        is_data_parallel
        and is_eplb_enabled
        and is_nccl_based_eplb_communicator
        and is_mega_moe
    ):
        current_value_str = os.getenv("NCCL_MAX_CTAS")

        if current_value_str and current_value_str.isdigit():
            return

        override_value = 8
        os.environ["NCCL_MAX_CTAS"] = str(override_value)
        logger.info_once(
            f"EPLB: Setting NCCL_MAX_CTAS={override_value} "
            f"for expert parallel with NCCL-based EPLB communicator and "
            f"cooperative MoE backend (deep_gemm_mega_moe)",
            scope="global",
        )
