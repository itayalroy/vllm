# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class StageTimer:
    def __init__(self, prefix: str, *, synchronize_gpu: bool = True) -> None:
        self.prefix = prefix
        self.synchronize_gpu = synchronize_gpu
        self.enabled = os.getenv("VLLM_ELASTIC_EP_COMMIT_TIMING", "0") == "1"
        if self.enabled and synchronize_gpu:
            torch.accelerator.synchronize()
        self.started = self.total_started = time.perf_counter()

    def mark(self, stage: str, *, synchronize_gpu: bool | None = None) -> None:
        if not self.enabled:
            return
        if synchronize_gpu is None:
            synchronize_gpu = self.synchronize_gpu
        if synchronize_gpu:
            torch.accelerator.synchronize()
        now = time.perf_counter()
        elapsed = now - self.started
        logger.info(
            "[Elastic EP diagnostic] stage=%s.%s seconds=%.6f",
            self.prefix,
            stage,
            elapsed,
        )
        self.started = time.perf_counter()

    def total(self) -> None:
        if not self.enabled:
            return
        if self.synchronize_gpu:
            torch.accelerator.synchronize()
        logger.info(
            "[Elastic EP diagnostic] stage=%s.total seconds=%.6f",
            self.prefix,
            time.perf_counter() - self.total_started,
        )
