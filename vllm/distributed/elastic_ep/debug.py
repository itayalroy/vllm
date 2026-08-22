# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Any

from vllm.distributed.utils import get_cached_tcp_store_client
from vllm.logger import init_logger
from vllm.v1.engine import ReconfigureDistributedRequest

logger = init_logger(__name__)

_TRACE = os.getenv("VLLM_EEP_DEBUG_TRACE", "0") == "1"
_HOLD_PHASE = os.getenv("VLLM_EEP_DEBUG_HOLD_PHASE")
_HOLD_SECONDS = float(os.getenv("VLLM_EEP_DEBUG_HOLD_SECONDS", "120"))
_current_phase = "serving"
_identity: dict[str, Any] = {}
_history: deque[dict[str, Any]] = deque(maxlen=64)
_history_lock = threading.Lock()
_heartbeat_thread: threading.Thread | None = None


def enabled() -> bool:
    return _TRACE or _HOLD_PHASE is not None


def current_phase() -> str:
    return _current_phase


def activate(**fields: Any) -> None:
    global _heartbeat_thread
    if not enabled():
        return
    _identity.update(fields)
    trace_event("trace_activated", **fields)
    if _heartbeat_thread is None:
        _heartbeat_thread = threading.Thread(
            target=_heartbeat,
            name="ElasticEPDebugHeartbeat",
            daemon=True,
        )
        _heartbeat_thread.start()


def _heartbeat() -> None:
    while True:
        trace_event("heartbeat", **_identity)
        time.sleep(1)


def trace_event(event: str, **fields: Any) -> None:
    if not enabled():
        return
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info(
        "[Elastic EP debug] wall_ns=%d monotonic_ns=%d event=%s prepare_phase=%s %s",
        time.time_ns(),
        time.monotonic_ns(),
        event,
        current_phase(),
        details,
    )


def record_forward(event: str, **fields: Any) -> None:
    if not enabled():
        return
    record = {
        "monotonic_ns": time.monotonic_ns(),
        "event": event,
        "prepare_phase": current_phase(),
        **fields,
    }
    with _history_lock:
        _history.append(record)


def dump_forward_history(reason: str, **fields: Any) -> None:
    if not enabled():
        return
    with _history_lock:
        history = list(_history)
    logger.error(
        "[Elastic EP debug] forward_history reason=%s fields=%s records=%s",
        reason,
        fields,
        history,
    )


@contextmanager
def dump_forward_errors(reason: str, **fields: Any):
    try:
        yield
    except BaseException as error:
        dump_forward_history(reason, error=repr(error), **fields)
        raise


def trace_phase(phase: str, event: str, **fields: Any) -> None:
    global _current_phase
    if not enabled():
        return
    _current_phase = f"{phase}:{event}"
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info(
        "[Elastic EP debug] wall_ns=%d monotonic_ns=%d phase=%s event=%s %s",
        time.time_ns(),
        time.monotonic_ns(),
        phase,
        event,
        details,
    )


@contextmanager
def trace_phase_context(phase: str, **fields: Any):
    if not enabled():
        yield
        return
    trace_phase(phase, "begin", **fields)
    try:
        yield
    except BaseException:
        trace_phase(phase, "error", **fields)
        raise
    trace_phase(phase, "end", **fields)


def hold_after_phase(
    phase: str,
    reconfig_request: ReconfigureDistributedRequest,
    old_dp_size: int,
    model_parallel_size: int,
    dp_rank: int,
    worker_rank: int,
) -> None:
    if phase != _HOLD_PHASE:
        return

    store = get_cached_tcp_store_client(
        reconfig_request.new_data_parallel_master_ip,
        reconfig_request.coord_store_port,
    )
    prefix = (
        f"eep_debug/{old_dp_size}-{reconfig_request.new_data_parallel_size}/{phase}"
    )
    arrival_key = f"{prefix}/arrived/{dp_rank}/{worker_rank}"
    start_key = f"{prefix}/start"
    release_key = f"{prefix}/release"
    store.set(arrival_key, b"1")
    trace_phase(
        phase,
        "gate_arrived",
        dp_rank=dp_rank,
        worker_rank=worker_rank,
    )

    if dp_rank == 0 and worker_rank == 0:
        arrival_keys = [
            f"{prefix}/arrived/{rank}/{worker}"
            for rank in range(old_dp_size)
            for worker in range(model_parallel_size)
        ]
        store.wait(arrival_keys)
        store.set(start_key, b"1")
        trace_phase(phase, "gate_started", hold_seconds=_HOLD_SECONDS)
        time.sleep(_HOLD_SECONDS)
        store.set(release_key, b"1")
    else:
        store.wait([start_key])

    store.wait([release_key])
    trace_phase(
        phase,
        "gate_released",
        dp_rank=dp_rank,
        worker_rank=worker_rank,
    )
