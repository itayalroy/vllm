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


@triton.jit(do_not_specialize=["expert_load_window_step"])
def _record_expert_load(
    expert_load_pass,
    physical_to_logical_map,
    expert_load_window,
    expert_load_window_step,
    MAX_PHYSICAL_EXPERTS: tl.constexpr,
    NUM_LOGICAL_EXPERTS: tl.constexpr,
):
    """Each program records one layer's logical loads and clears its physical counts."""
    BLOCK_SIZE: tl.constexpr = triton.next_power_of_2(MAX_PHYSICAL_EXPERTS)
    expert_slots = tl.arange(0, BLOCK_SIZE)
    layer = tl.program_id(0)
    num_layers = tl.num_programs(0)

    # Clear this layer's counts in the window slot being overwritten.
    # expert_load_window shape: [window_size, num_layers, entries_per_layer].
    entries_per_layer = NUM_LOGICAL_EXPERTS + 1
    step_offset = expert_load_window_step.to(tl.int64) * num_layers * entries_per_layer
    layer_offset = layer.to(tl.int64) * entries_per_layer
    layer_window_ptr = expert_load_window + step_offset + layer_offset
    tl.store(layer_window_ptr + expert_slots, 0, expert_slots < NUM_LOGICAL_EXPERTS)
    tl.debug_barrier()

    physical_expert_offsets = layer * MAX_PHYSICAL_EXPERTS + expert_slots
    valid_slot = expert_slots < MAX_PHYSICAL_EXPERTS
    logical_expert_ids = tl.load(
        physical_to_logical_map + physical_expert_offsets, mask=valid_slot, other=-1
    )
    expert_load = tl.load(
        expert_load_pass + physical_expert_offsets, mask=valid_slot, other=0
    )

    tl.atomic_add(
        layer_window_ptr + logical_expert_ids,
        expert_load,
        mask=logical_expert_ids >= 0,
        sem="relaxed",
    )
    tl.store(expert_load_pass + physical_expert_offsets, 0, mask=valid_slot)


def record_expert_load(
    expert_load_pass: torch.Tensor,
    physical_to_logical_map: torch.Tensor,
    expert_load_window: torch.Tensor,
    expert_load_window_step: int,
) -> None:
    """Record logical loads and clear full-capacity physical counters.

    Inactive slots in the full-capacity mapping must contain -1.
    """
    num_logical_experts = expert_load_window.shape[-1] - 1
    if not expert_load_pass.is_cuda:
        expert_load_window[expert_load_window_step].zero_().scatter_add_(
            -1,
            physical_to_logical_map.masked_fill(
                physical_to_logical_map < 0, num_logical_experts
            ),
            expert_load_pass,
        )
        expert_load_pass.zero_()
        return
    _record_expert_load[(expert_load_pass.shape[0],)](
        expert_load_pass,
        physical_to_logical_map,
        expert_load_window,
        expert_load_window_step,
        expert_load_pass.stride(0),
        num_logical_experts,
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
