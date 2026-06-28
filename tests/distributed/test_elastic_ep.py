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
TRAFFIC_MODES: tuple[TrafficMode, ...] = ("none", "light", "heavy")
HEAVY_TRAFFIC_MODE: TrafficMode = "heavy"
PinnedDPRank = int | None
TrafficEvent = tuple[float, float, int | None, int]


ELASTIC_EP_SCALING_CASES = [
    pytest.param(
        "enforce_eager",
        traffic_mode,
        id=f"enforce_eager_{traffic_mode}",
    )
    for traffic_mode in TRAFFIC_MODES
] + [
    pytest.param(
        "cuda_graphs",
        HEAVY_TRAFFIC_MODE,
        id="cuda_graphs_heavy",
    ),
]


def _post_elastic_ep_command_timed(
    server: RemoteOpenAIServer, route: str, new_dp_size: int | None = None
) -> tuple[bool, float, int | None]:
    url = server.url_for(route)
    payload = {"new_data_parallel_size": new_dp_size} if new_dp_size else {}
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
    server: RemoteOpenAIServer, pinned_dp_rank: int | None = None
) -> int:
    url = server.url_for("v1/chat/completions")
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Reply with the word ok."}],
        "max_tokens": 4,
        "temperature": 0,
    }
    headers = None
    if pinned_dp_rank is not None:
        headers = {"X-data-parallel-rank": str(pinned_dp_rank)}
    response = requests.post(url, json=payload, headers=headers, timeout=120)
    return response.status_code


def _traffic_clients(mode: TrafficMode, dp_size: int) -> list[PinnedDPRank]:
    if mode == "none":
        return []
    if mode == "light":
        return [0]
    return [None] * dp_size


def _traffic_loop(
    server: RemoteOpenAIServer,
    client_id: int,
    pinned_dp_rank: PinnedDPRank,
    stop_event: threading.Event,
    events: list[TrafficEvent],
    events_lock: threading.Lock,
) -> None:
    while not stop_event.is_set():
        start_time = time.perf_counter()
        try:
            status_code = _send_liveness_completion(server, pinned_dp_rank)
        except requests.exceptions.RequestException:
            status_code = None
        end_time = time.perf_counter()
        with events_lock:
            events.append((start_time, end_time, status_code, client_id))
        time.sleep(0.05)


def _wait_for_traffic(
    events: list[TrafficEvent],
    events_lock: threading.Lock,
    client_count: int,
    since: float = 0.0,
) -> None:
    deadline = time.perf_counter() + 120
    expected_clients = set(range(client_count))
    while time.perf_counter() < deadline:
        with events_lock:
            ready_clients = {
                client_id
                for _, end_time, status_code, client_id in events
                if status_code == 200 and end_time >= since
            }
        if expected_clients <= ready_clients:
            return
        time.sleep(0.05)
    raise TimeoutError("Timed out waiting for traffic")


def _503_downtime_span(events: list[TrafficEvent]) -> float:
    max_downtime = 0.0
    client_ids = {client_id for _, _, _, client_id in events}
    for client_id in client_ids:
        downtime_start = None
        client_events = (
            event
            for event in sorted(events, key=lambda event: event[1])
            if event[3] == client_id
        )
        for _, end_time, status_code, _ in client_events:
            if status_code == 503 and downtime_start is None:
                downtime_start = end_time
            elif status_code == 200 and downtime_start is not None:
                max_downtime = max(max_downtime, end_time - downtime_start)
                downtime_start = None
    return max_downtime


def _prepare_and_scale_elastic_ep(
    server: RemoteOpenAIServer,
    source_dp_size: int,
    target_dp_size: int,
    mode: str,
    traffic_mode: TrafficMode,
) -> None:
    total_start = time.perf_counter()
    traffic_clients = _traffic_clients(traffic_mode, source_dp_size)
    events: list[TrafficEvent] = []
    events_lock = threading.Lock()
    prepare_done_at = 0.0
    prepare_seconds = float("nan")
    commit_seconds = float("nan")

    with ThreadPoolExecutor(max_workers=max(len(traffic_clients), 1)) as executor:
        stop_events = [threading.Event() for _ in traffic_clients]
        futures = [
            executor.submit(
                _traffic_loop,
                server,
                client_id,
                pinned_dp_rank,
                stop_event,
                events,
                events_lock,
            )
            for client_id, (pinned_dp_rank, stop_event) in enumerate(
                zip(traffic_clients, stop_events)
            )
        ]
        if traffic_clients:
            _wait_for_traffic(events, events_lock, len(traffic_clients))
        try:
            prepare_ok, prepare_seconds, prepare_status = (
                _post_elastic_ep_command_timed(
                    server, "prepare_elastic_ep", target_dp_size
                )
            )
            prepare_done_at = time.perf_counter()
            assert prepare_ok, f"prepare_elastic_ep failed with status {prepare_status}"
            time.sleep(3)
            scale_ok, commit_seconds, scale_status = _post_elastic_ep_command_timed(
                server, "scale_elastic_ep"
            )
            scale_done_at = time.perf_counter()
            assert scale_ok, f"scale_elastic_ep failed with status {scale_status}"
            if traffic_clients:
                _wait_for_traffic(
                    events, events_lock, len(traffic_clients), scale_done_at
                )
        finally:
            for stop_event in stop_events:
                stop_event.set()
        for future in futures:
            future.result()

    prepare_bad_statuses = sorted(
        {
            str(status_code)
            for _, end_time, status_code, _client_id in events
            if end_time <= prepare_done_at and status_code != 200
        }
    )
    assert not prepare_bad_statuses, (
        f"prepare traffic got unexpected statuses {prepare_bad_statuses}"
    )
    bad_statuses = sorted(
        {
            str(status_code)
            for _, _, status_code, _client_id in events
            if status_code not in (200, 503)
        }
    )
    assert not bad_statuses, f"traffic got unexpected statuses {bad_statuses}"
    downtime_seconds = _503_downtime_span(events)

    total_seconds = time.perf_counter() - total_start
    print(
        f"[Elastic EP timing][{source_dp_size}->{target_dp_size}]"
        f"[{mode}]"
        f"[traffic={traffic_mode}] "
        f"prepare_seconds={prepare_seconds:.3f} "
        f"commit_seconds={commit_seconds:.3f} "
        f"downtime_seconds={downtime_seconds:.3f} "
        f"total_seconds={total_seconds:.3f}"
    )


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
    ("eager_mode", "traffic_mode"),
    ELASTIC_EP_SCALING_CASES,
)
@multi_gpu_test(num_gpus=4)
def test_elastic_ep_scaling(
    eager_mode: EagerMode,
    traffic_mode: TrafficMode,
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

    with RemoteOpenAIServer(
        MODEL_NAME, vllm_serve_args, env_dict={}, max_wait_seconds=1200
    ) as server:
        initial_accuracy = _run_gsm8k_eval(server, f"Initial ({initial_dp_size} GPUs)")

        _prepare_and_scale_elastic_ep(
            server,
            initial_dp_size,
            target_dp_size,
            eager_mode,
            traffic_mode,
        )
        scale_up_stage = (
            f"After scale up to {target_dp_size} GPUs ({traffic_mode} traffic)"
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
            traffic_mode,
        )
        scale_down_stage = (
            f"After scale down to {initial_dp_size} GPUs ({traffic_mode} traffic)"
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
        assert _post_elastic_ep_command_timed(server, "scale_elastic_ep")[0]
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
        assert _post_elastic_ep_command_timed(server, "scale_elastic_ep")[0]
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
