# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextvars
import json
import time
from collections import OrderedDict
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class _StageTiming:
    calls: int = 0
    host_seconds: float = 0.0
    gpu_wait_seconds: float = 0.0

    @property
    def total_seconds(self) -> float:
        return self.host_seconds + self.gpu_wait_seconds


@dataclass
class _GapTiming:
    calls: int = 0
    total_seconds: float = 0.0
    max_seconds: float = 0.0

    def add(self, seconds: float) -> None:
        self.calls += 1
        self.total_seconds += seconds
        self.max_seconds = max(self.max_seconds, seconds)


class ElasticEPCommitTiming:
    def __init__(self, scope: str, operation: str, **metadata: Any) -> None:
        self.scope = scope
        self.operation = operation
        self.metadata = metadata
        self.stages: OrderedDict[str, _StageTiming] = OrderedDict()
        self.metrics: OrderedDict[str, float] = OrderedDict()
        self.counters: OrderedDict[str, int] = OrderedDict()
        self._stage_intervals: list[tuple[str, float, float]] = []
        self._start = 0.0
        self._token: contextvars.Token[ElasticEPCommitTiming | None] | None = None

    def __enter__(self) -> "ElasticEPCommitTiming":
        self._start = time.perf_counter()
        self._token = _active_timing.set(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        end = time.perf_counter()
        total_seconds = end - self._start
        if self._token is not None:
            _active_timing.reset(self._token)

        unattributed_seconds, gaps, small_gaps_seconds = self._find_unattributed_gaps(
            end
        )
        report = {
            "scope": self.scope,
            "operation": self.operation,
            **self.metadata,
            "status": "error" if exc_type is not None else "ok",
            "total_seconds": round(total_seconds, 6),
            "unattributed_seconds": round(unattributed_seconds, 6),
            "unattributed_gaps": gaps,
            "unattributed_small_gaps_seconds": round(small_gaps_seconds, 6),
            "stages": [
                {
                    "name": name,
                    "calls": stage.calls,
                    "host_seconds": round(stage.host_seconds, 6),
                    "gpu_wait_seconds": round(stage.gpu_wait_seconds, 6),
                    "total_seconds": round(stage.total_seconds, 6),
                }
                for name, stage in self.stages.items()
            ],
            "overlapping_metrics": {
                name: round(seconds, 6) for name, seconds in self.metrics.items()
            },
            "counters": self.counters,
        }
        logger.info("[Elastic EP commit timing] %s", json.dumps(report))

    def _find_unattributed_gaps(
        self, end: float
    ) -> tuple[float, list[dict[str, Any]], float]:
        grouped: OrderedDict[tuple[str, str], _GapTiming] = OrderedDict()
        cursor = self._start
        previous = "commit.start"
        for name, start, stage_end in sorted(
            self._stage_intervals, key=lambda interval: interval[1]
        ):
            if start > cursor:
                grouped.setdefault((previous, name), _GapTiming()).add(start - cursor)
            if stage_end > cursor:
                cursor = stage_end
                previous = name
        if end > cursor:
            grouped.setdefault((previous, "commit.end"), _GapTiming()).add(end - cursor)

        threshold = 0.001
        unattributed_seconds = sum(gap.total_seconds for gap in grouped.values())
        small_gaps_seconds = sum(
            gap.total_seconds
            for gap in grouped.values()
            if gap.total_seconds < threshold
        )
        gaps = [
            {
                "after": after,
                "before": before,
                "calls": gap.calls,
                "total_seconds": round(gap.total_seconds, 6),
                "max_seconds": round(gap.max_seconds, 6),
            }
            for (after, before), gap in sorted(
                grouped.items(),
                key=lambda item: item[1].total_seconds,
                reverse=True,
            )
            if gap.total_seconds >= threshold
        ]
        return unattributed_seconds, gaps, small_gaps_seconds

    def add_stage(
        self,
        name: str,
        host_seconds: float,
        gpu_wait_seconds: float,
        start: float,
        end: float,
    ) -> None:
        stage = self.stages.setdefault(name, _StageTiming())
        stage.calls += 1
        stage.host_seconds += host_seconds
        stage.gpu_wait_seconds += gpu_wait_seconds
        self._stage_intervals.append((name, start, end))

    def add_metric(self, name: str, seconds: float) -> None:
        self.metrics[name] = self.metrics.get(name, 0.0) + seconds

    def increment(self, name: str, value: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + value


_active_timing: contextvars.ContextVar[ElasticEPCommitTiming | None] = (
    contextvars.ContextVar("elastic_ep_commit_timing", default=None)
)


def is_commit_timing_enabled() -> bool:
    return envs.VLLM_ELASTIC_EP_COMMIT_TIMING


@contextmanager
def collect_commit_timing(
    scope: str, operation: str, **metadata: Any
) -> Generator[ElasticEPCommitTiming | None, None, None]:
    if not is_commit_timing_enabled():
        yield None
        return

    with ElasticEPCommitTiming(scope, operation, **metadata) as timing:
        yield timing


@contextmanager
def record_commit_stage(
    name: str, *, synchronize_gpu: bool = False
) -> Generator[None, None, None]:
    timing = _active_timing.get()
    if timing is None:
        yield
        return

    start = time.perf_counter()
    completed = False
    try:
        yield
        completed = True
    finally:
        host_end = time.perf_counter()
        stage_end = host_end
        gpu_wait_seconds = 0.0
        if synchronize_gpu and completed:
            import torch

            torch.accelerator.synchronize()
            stage_end = time.perf_counter()
            gpu_wait_seconds = stage_end - host_end
        timing.add_stage(name, host_end - start, gpu_wait_seconds, start, stage_end)


def record_commit_metric(name: str, seconds: float) -> None:
    timing = _active_timing.get()
    if timing is not None:
        timing.add_metric(name, seconds)


def increment_commit_counter(name: str, value: int = 1) -> None:
    timing = _active_timing.get()
    if timing is not None:
        timing.increment(name, value)
