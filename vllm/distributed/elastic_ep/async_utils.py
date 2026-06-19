# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from collections.abc import Callable
from concurrent.futures import Future

from vllm.logger import init_logger

logger = init_logger(__name__)


class SingleMethodAsyncRunner:
    def __init__(self):
        self._lock = threading.Lock()
        self._method: str | None = None
        self._thread: threading.Thread | None = None
        self._future: Future[None] | None = None

    def start(self, method: str, target: Callable[..., None], *args, **kwargs) -> None:
        with self._lock:
            if self._method is not None:
                if self._method == method:
                    return
                raise RuntimeError(
                    f"Elastic EP async method is already active: {self._method}"
                )

            future: Future[None] = Future()
            self._method = method
            self._future = future
            self._thread = threading.Thread(
                target=self._run,
                args=(method, target, future, args, kwargs),
                daemon=True,
                name=f"ElasticEPAsync-{method}",
            )
            self._thread.start()

    def clear(self, method: str) -> None:
        with self._lock:
            if self._method != method:
                raise RuntimeError(
                    "Elastic EP async method mismatch: "
                    f"expected {self._method}, got {method}"
                )
            future = self._future
            assert future is not None
            if not future.done():
                raise RuntimeError(f"Elastic EP async method {method} is not done")
            thread = self._thread
            self._method = self._thread = self._future = None
        if thread is not None:
            thread.join(timeout=0)

    def _run(
        self,
        method: str,
        target: Callable[..., None],
        future: Future[None],
        args: tuple,
        kwargs: dict,
    ) -> None:
        try:
            target(method, *args, **kwargs)
        except BaseException as e:
            logger.exception("[Elastic EP] Async worker method %s failed", method)
            future.set_exception(e)
            return

        future.set_result(None)
