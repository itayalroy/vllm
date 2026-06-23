# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import pytest
import requests

from ..evals.gsm8k.gsm8k_eval import evaluate_gsm8k
from ..utils import RemoteOpenAIServer, multi_gpu_test


@pytest.fixture(autouse=True)
def cleanup_ray_between_tests():
    """Force-stop any lingering Ray processes between tests."""
    subprocess.run(["ray", "stop", "--force"], timeout=30, capture_output=True)
    time.sleep(5)
    yield


MODEL_NAME = "deepseek-ai/DeepSeek-V2-Lite-Chat"

NUM_GSM8K_QUESTIONS = 256
EXPECTED_ACCURACY = 0.58
ACCURACY_TOL = 0.08
MAX_NUM_SEQS = 32
EagerMode = Literal["enforce_eager", "cuda_graphs"]
TrafficMode = Literal["none", "light", "heavy"]
TrafficCase = tuple[TrafficMode, TrafficMode]
TRAFFIC_MODES: tuple[TrafficMode, ...] = ("none", "light", "heavy")
TRAFFIC_CASES: tuple[TrafficCase, ...] = tuple(
    (traffic, traffic) for traffic in TRAFFIC_MODES
)
HEAVY_TRAFFIC_CASE: TrafficCase = ("heavy", "heavy")
TrafficEvent = tuple[float, float, int | None, int | None]


ELASTIC_EP_SCALING_CASES = [
    pytest.param(
        "enforce_eager",
        traffic_case,
        id=f"enforce_eager_{traffic_case[0]}_{traffic_case[1]}",
    )
    for traffic_case in TRAFFIC_CASES
] + [
    pytest.param(
        "cuda_graphs",
        HEAVY_TRAFFIC_CASE,
        id="cuda_graphs_heavy_heavy",
    ),
]


def _send_scale_command(server: RemoteOpenAIServer, new_dp_size: int) -> bool:
    url = server.url_for("scale_elastic_ep")
    payload = {"new_data_parallel_size": new_dp_size}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=300)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def _post_elastic_ep_command_timed(
    server: RemoteOpenAIServer, route: str, new_dp_size: int
) -> tuple[bool, float, int | None]:
    url = server.url_for(route)
    payload = {"new_data_parallel_size": new_dp_size}
    headers = {"Content-Type": "application/json"}
    start_time = time.perf_counter()
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=300)
        elapsed = time.perf_counter() - start_time
        return response.status_code == 200, elapsed, response.status_code
    except requests.exceptions.RequestException:
        elapsed = time.perf_counter() - start_time
        return False, elapsed, None


def _send_liveness_completion(
    server: RemoteOpenAIServer, dp_rank: int | None = None
) -> int:
    url = server.url_for("v1/chat/completions")
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Reply with the word ok."}],
        "max_tokens": 4,
        "temperature": 0,
    }
    headers = None
    if dp_rank is not None:
        headers = {"X-data-parallel-rank": str(dp_rank)}
    response = requests.post(url, json=payload, headers=headers, timeout=120)
    return response.status_code


def _traffic_ranks(mode: TrafficMode, dp_size: int) -> list[int]:
    if mode == "none":
        return []
    if mode == "light":
        return [0]
    return list(range(dp_size))


def _traffic_loop(
    server: RemoteOpenAIServer,
    dp_rank: int,
    stop_event: threading.Event,
    events: list[TrafficEvent],
    events_lock: threading.Lock,
) -> None:
    while not stop_event.is_set():
        start_time = time.perf_counter()
        try:
            status_code = _send_liveness_completion(server, dp_rank)
        except requests.exceptions.RequestException:
            status_code = None
        end_time = time.perf_counter()
        with events_lock:
            events.append((start_time, end_time, status_code, dp_rank))
        time.sleep(0.05)


def _wait_for_traffic(
    events: list[TrafficEvent],
    events_lock: threading.Lock,
    ranks: list[int],
    since: float = 0.0,
) -> None:
    deadline = time.perf_counter() + 120
    expected_ranks = set(ranks)
    while time.perf_counter() < deadline:
        with events_lock:
            ready_ranks = {
                rank
                for _, end_time, status_code, rank in events
                if status_code == 200 and rank is not None and end_time >= since
            }
        if expected_ranks <= ready_ranks:
            return
        time.sleep(0.05)
    raise TimeoutError(f"Timed out waiting for traffic on ranks {sorted(ranks)}")


def _503_downtime_span(events: list[TrafficEvent]) -> float:
    sorted_events = sorted(events, key=lambda event: event[1])
    first_503_index = next(
        (
            index
            for index, (_, _, status_code, _) in enumerate(sorted_events)
            if status_code == 503
        ),
        None,
    )
    if first_503_index is None:
        return 0.0

    last_503_index = (
        len(sorted_events)
        - 1
        - next(
            index
            for index, (_, _, status_code, _) in enumerate(reversed(sorted_events))
            if status_code == 503
        )
    )
    last_success_before_503 = next(
        (
            end_time
            for _, end_time, status_code, _ in reversed(sorted_events[:first_503_index])
            if status_code == 200
        ),
        None,
    )
    first_success_after_503 = next(
        (
            end_time
            for _, end_time, status_code, _ in sorted_events[last_503_index + 1 :]
            if status_code == 200
        ),
        None,
    )
    if last_success_before_503 is None or first_success_after_503 is None:
        return 0.0
    return first_success_after_503 - last_success_before_503


def _record_liveness_completion(
    server: RemoteOpenAIServer,
    events: list[TrafficEvent],
    dp_rank: int | None = None,
) -> float:
    start_time = time.perf_counter()
    status_code = _send_liveness_completion(server, dp_rank)
    end_time = time.perf_counter()
    assert status_code == 200
    events.append((start_time, end_time, status_code, dp_rank))
    return end_time


def _run_command_with_traffic(
    server: RemoteOpenAIServer,
    route: str,
    source_dp_size: int,
    target_dp_size: int,
    traffic_mode: TrafficMode,
) -> tuple[bool, float, int | None, list[TrafficEvent]]:
    ranks = _traffic_ranks(traffic_mode, source_dp_size)
    events: list[TrafficEvent] = []
    events_lock = threading.Lock()
    if not ranks:
        ok, seconds, status = _post_elastic_ep_command_timed(
            server, route, target_dp_size
        )
        return ok, seconds, status, events

    with ThreadPoolExecutor(max_workers=len(ranks)) as executor:
        stop_events = {rank: threading.Event() for rank in ranks}
        futures = [
            executor.submit(
                _traffic_loop,
                server,
                rank,
                stop_events[rank],
                events,
                events_lock,
            )
            for rank in ranks
        ]
        _wait_for_traffic(events, events_lock, ranks)
        try:
            ok, seconds, status = _post_elastic_ep_command_timed(
                server, route, target_dp_size
            )
            if route == "scale_elastic_ep" and ok:
                if target_dp_size < source_dp_size:
                    for rank in ranks:
                        if rank >= target_dp_size:
                            stop_events[rank].set()
                post_scale_ranks = [rank for rank in ranks if rank < target_dp_size]
                _wait_for_traffic(
                    events, events_lock, post_scale_ranks, time.perf_counter()
                )
        finally:
            for rank_stop_event in stop_events.values():
                rank_stop_event.set()
        for future in futures:
            future.result()

    return ok, seconds, status, events


def _prepare_and_scale_elastic_ep(
    server: RemoteOpenAIServer,
    source_dp_size: int,
    target_dp_size: int,
    mode: str,
    prepare_traffic: TrafficMode,
    scale_traffic: TrafficMode,
) -> None:
    total_start = time.perf_counter()
    prepare_ok, prepare_seconds, prepare_status, prepare_events = (
        _run_command_with_traffic(
            server,
            "prepare_elastic_ep",
            source_dp_size,
            target_dp_size,
            prepare_traffic,
        )
    )
    bad_statuses = sorted(
        {
            str(status_code)
            for _, _, status_code, _ in prepare_events
            if status_code != 200
        }
    )
    assert not bad_statuses, f"prepare traffic got unexpected statuses {bad_statuses}"
    assert prepare_ok, f"prepare_elastic_ep failed with status {prepare_status}"

    scale_events: list[TrafficEvent] = []
    last_completion_at = None
    if scale_traffic == "none":
        last_completion_at = _record_liveness_completion(server, scale_events)
    scale_ok, switch_seconds, scale_status, traffic_events = _run_command_with_traffic(
        server,
        "scale_elastic_ep",
        source_dp_size,
        target_dp_size,
        scale_traffic,
    )
    scale_events.extend(traffic_events)
    downtime_seconds = float("nan")
    if scale_ok:
        first_completion_at = _record_liveness_completion(server, scale_events)
        downtime_seconds = (
            first_completion_at - last_completion_at
            if last_completion_at is not None
            else _503_downtime_span(scale_events)
        )
    bad_statuses = sorted(
        {
            str(status_code)
            for _, _, status_code, rank in scale_events
            if status_code not in (200, 503)
            and not (
                status_code == 400
                and target_dp_size < source_dp_size
                and rank is not None
                and rank >= target_dp_size
            )
        }
    )
    assert not bad_statuses, f"scale traffic got unexpected statuses {bad_statuses}"

    total_seconds = time.perf_counter() - total_start
    print(
        f"[Elastic EP timing][{source_dp_size}->{target_dp_size}]"
        f"[{mode}]"
        f"[prepare_traffic={prepare_traffic}] "
        f"[scale_traffic={scale_traffic}] "
        f"prepare_seconds={prepare_seconds:.3f} "
        f"switch_seconds={switch_seconds:.3f} "
        f"downtime_seconds={downtime_seconds:.3f} "
        f"total_seconds={total_seconds:.3f}"
    )
    assert scale_ok, f"scale_elastic_ep failed with status {scale_status}"


def _run_gsm8k_eval(server: RemoteOpenAIServer, stage: str) -> float:
    assert server.port is not None
    result = evaluate_gsm8k(
        num_questions=NUM_GSM8K_QUESTIONS,
        host=f"http://{server.host}",
        port=server.port,
    )
    accuracy = result["accuracy"]
    print(
        f"[{stage}] GSM8K accuracy: {accuracy:.3f} "
        f"({result['num_questions']} questions)"
    )
    assert accuracy >= EXPECTED_ACCURACY, (
        f"[{stage}] GSM8K accuracy {accuracy:.3f} is below "
        f"expected threshold {EXPECTED_ACCURACY}"
    )
    return accuracy


def _base_serve_args(
    data_parallel_size: int = 2,
    enforce_eager: bool = False,
) -> list[str]:
    args = [
        "--trust-remote-code",
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        "0.8",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--enable-expert-parallel",
        "--all2all-backend",
        "allgather_reducescatter",
        "--enable-elastic-ep",
        "--enable-eplb",
        "--eplb-config.num_redundant_experts",
        "0",
        "--eplb-config.use_async",
        "true",
        "--eplb-config.step_interval",
        "300",
        "--eplb-config.window_size",
        "5",
        "--data-parallel-backend",
        "ray",
        "--data-parallel-size",
        str(data_parallel_size),
        "--api-server-count",
        "1",
    ]
    leader_address = os.environ.get("LEADER_ADDRESS")
    if leader_address:
        args.extend(["--data-parallel-address", leader_address])
    if enforce_eager:
        args.append("--enforce-eager")

    return args


@pytest.mark.parametrize(
    ("eager_mode", "traffic_case"),
    ELASTIC_EP_SCALING_CASES,
)
@multi_gpu_test(num_gpus=4)
def test_elastic_ep_scaling(
    eager_mode: EagerMode,
    traffic_case: TrafficCase,
):
    from vllm.distributed.eplb.eplb_communicator import has_nixl

    if not has_nixl():
        pytest.skip("Async EPLB with elastic EP requires NIXL (not installed)")

    initial_dp_size = int(os.getenv("VLLM_TEST_ELASTIC_EP_INITIAL_DP", "2"))
    target_dp_size = int(os.getenv("VLLM_TEST_ELASTIC_EP_TARGET_DP", "4"))
    assert target_dp_size > initial_dp_size
    vllm_serve_args = _base_serve_args(
        initial_dp_size,
        eager_mode == "enforce_eager",
    )
    prepare_traffic, scale_traffic = traffic_case

    with RemoteOpenAIServer(
        MODEL_NAME, vllm_serve_args, env_dict={}, max_wait_seconds=1200
    ) as server:
        initial_accuracy = _run_gsm8k_eval(server, f"Initial ({initial_dp_size} GPUs)")

        _prepare_and_scale_elastic_ep(
            server,
            initial_dp_size,
            target_dp_size,
            eager_mode,
            prepare_traffic,
            scale_traffic,
        )
        scale_up_stage = (
            f"After scale up to {target_dp_size} GPUs "
            f"({prepare_traffic}->{scale_traffic})"
        )
        scale_up_accuracy = _run_gsm8k_eval(server, scale_up_stage)
        assert scale_up_accuracy >= initial_accuracy - ACCURACY_TOL, (
            f"[{scale_up_stage}] accuracy {scale_up_accuracy:.3f} dropped more "
            f"than {ACCURACY_TOL} below initial accuracy {initial_accuracy:.3f}"
        )

        _prepare_and_scale_elastic_ep(
            server,
            target_dp_size,
            initial_dp_size,
            eager_mode,
            prepare_traffic,
            scale_traffic,
        )
        scale_down_stage = (
            f"After scale down to {initial_dp_size} GPUs "
            f"({prepare_traffic}->{scale_traffic})"
        )
        scale_down_accuracy = _run_gsm8k_eval(server, scale_down_stage)
        assert scale_down_accuracy >= initial_accuracy - ACCURACY_TOL, (
            f"[{scale_down_stage}] accuracy {scale_down_accuracy:.3f} dropped more "
            f"than {ACCURACY_TOL} below initial accuracy {initial_accuracy:.3f}"
        )

        print("\nAccuracy Summary:")
        print(f"  Initial:    {initial_accuracy:.3f}")
        print(
            f"  {scale_up_stage}: {scale_up_accuracy:.3f} "
            f"(diff: {scale_up_accuracy - initial_accuracy:+.3f})"
        )
        print(
            f"  {scale_down_stage}: {scale_down_accuracy:.3f} "
            f"(diff: {scale_down_accuracy - initial_accuracy:+.3f})"
        )
        print(f"  Tolerance:  {ACCURACY_TOL:.3f}")


@multi_gpu_test(num_gpus=4)
def test_elastic_ep_scaling_uneven():
    """Test scale up with uneven worker distribution.

    This tests the case where num_new_workers % old_dp_size != 0,
    specifically 2 -> 3 where remainder = 1 % 2 = 1.
    This exercises the remainder handling in sender-receiver pairing.
    """
    from vllm.distributed.eplb.eplb_communicator import has_nixl

    if not has_nixl():
        pytest.skip("Async EPLB with elastic EP requires NIXL (not installed)")

    vllm_serve_args = _base_serve_args()

    with RemoteOpenAIServer(
        MODEL_NAME, vllm_serve_args, env_dict={}, max_wait_seconds=1200
    ) as server:
        initial_accuracy = _run_gsm8k_eval(server, "Initial (2 GPUs)")

        # Scale 2 -> 3: This has remainder = 1 % 2 = 1
        # Tests uneven sender-receiver pairing
        prepare_ok, _, prepare_status = _post_elastic_ep_command_timed(
            server, "prepare_elastic_ep", 3
        )
        assert prepare_ok, f"prepare_elastic_ep failed with status {prepare_status}"
        assert _send_scale_command(server, 3)
        scale_up_accuracy = _run_gsm8k_eval(server, "After scale up (3 GPUs)")

        assert scale_up_accuracy >= initial_accuracy - ACCURACY_TOL, (
            f"Scale up accuracy {scale_up_accuracy:.3f} dropped more than "
            f"{ACCURACY_TOL} below initial accuracy {initial_accuracy:.3f}"
        )

        # Scale back down to 2
        prepare_ok, _, prepare_status = _post_elastic_ep_command_timed(
            server, "prepare_elastic_ep", 2
        )
        assert prepare_ok, f"prepare_elastic_ep failed with status {prepare_status}"
        assert _send_scale_command(server, 2)
        scale_down_accuracy = _run_gsm8k_eval(server, "After scale down (2 GPUs)")

        assert scale_down_accuracy >= initial_accuracy - ACCURACY_TOL, (
            f"Scale down accuracy {scale_down_accuracy:.3f} dropped more than "
            f"{ACCURACY_TOL} below initial accuracy {initial_accuracy:.3f}"
        )

        print("\nAccuracy Summary (Uneven Scaling):")
        print(f"  Initial:    {initial_accuracy:.3f}")
        print(
            f"  Scale up:   {scale_up_accuracy:.3f} "
            f"(diff: {scale_up_accuracy - initial_accuracy:+.3f})"
        )
        print(
            f"  Scale down: {scale_down_accuracy:.3f} "
            f"(diff: {scale_down_accuracy - initial_accuracy:+.3f})"
        )
        print(f"  Tolerance:  {ACCURACY_TOL:.3f}")
