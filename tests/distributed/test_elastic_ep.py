# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

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


def _send_liveness_completion(server: RemoteOpenAIServer) -> int:
    url = server.url_for("v1/chat/completions")
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Reply with the word ok."}],
        "max_tokens": 4,
        "temperature": 0,
    }
    response = requests.post(url, json=payload, timeout=120)
    return response.status_code


def _prepare_and_scale_elastic_ep(
    server: RemoteOpenAIServer,
    source_dp_size: int,
    target_dp_size: int,
    mode: str,
    use_async_eplb: bool,
) -> bool:
    total_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as executor:
        prepare_future = executor.submit(
            _post_elastic_ep_command_timed,
            server,
            "prepare_elastic_ep",
            target_dp_size,
        )
        served_during_prepare = False
        while not prepare_future.done():
            status_code = _send_liveness_completion(server)
            assert status_code == 200
            served_during_prepare = True
            time.sleep(0.5)
        prepare_ok, prepare_seconds, prepare_status = prepare_future.result()
    assert prepare_ok, f"prepare_elastic_ep failed with status {prepare_status}"

    assert _send_liveness_completion(server) == 200
    last_completion_at = time.perf_counter()
    scale_ok, switch_seconds, scale_status = _post_elastic_ep_command_timed(
        server, "scale_elastic_ep", target_dp_size
    )
    downtime_seconds = float("nan")
    if scale_ok:
        assert _send_liveness_completion(server) == 200
        downtime_seconds = time.perf_counter() - last_completion_at
    total_seconds = time.perf_counter() - total_start
    print(
        f"[Elastic EP timing][{source_dp_size}->{target_dp_size}]"
        f"[{mode}]"
        f"[{'async' if use_async_eplb else 'sync'}_eplb] "
        f"prepare_seconds={prepare_seconds:.3f} "
        f"switch_seconds={switch_seconds:.3f} "
        f"downtime_seconds={downtime_seconds:.3f} "
        f"total_seconds={total_seconds:.3f}"
    )
    assert scale_ok, f"scale_elastic_ep failed with status {scale_status}"
    return served_during_prepare


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
    use_async_eplb: bool = False, data_parallel_size: int = 2
) -> list[str]:
    all2all_backend = os.getenv(
        "VLLM_TEST_ELASTIC_EP_ALL2ALL_BACKEND", "allgather_reducescatter"
    )
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
        all2all_backend,
        "--enable-elastic-ep",
        "--enable-eplb",
        "--eplb-config.num_redundant_experts",
        "0",
        "--eplb-config.use_async",
        "true" if use_async_eplb else "false",
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
    if os.getenv("VLLM_TEST_ELASTIC_EP_ENFORCE_EAGER") == "1":
        args.append("--enforce-eager")

    return args


@pytest.mark.parametrize(
    "use_async_eplb", [False, True], ids=["sync_eplb", "async_eplb"]
)
@multi_gpu_test(num_gpus=8)
def test_elastic_ep_cold_boot_timing(use_async_eplb: bool):
    dp_size = int(os.getenv("VLLM_TEST_ELASTIC_EP_TARGET_DP", "8"))
    assert dp_size <= 8
    if use_async_eplb:
        from vllm.distributed.eplb.eplb_communicator import has_nixl

        if not has_nixl():
            pytest.skip("Async EPLB with elastic EP requires NIXL (not installed)")

    vllm_serve_args = _base_serve_args(use_async_eplb, dp_size)
    mode = "enforce_eager" if "--enforce-eager" in vllm_serve_args else "cuda_graphs"

    start_time = time.perf_counter()
    with RemoteOpenAIServer(
        MODEL_NAME, vllm_serve_args, env_dict={}, max_wait_seconds=1200
    ) as server:
        ready_seconds = time.perf_counter() - start_time
        first_completion_start = time.perf_counter()
        assert _send_liveness_completion(server) == 200
        first_completion_seconds = time.perf_counter() - first_completion_start
        total_seconds = time.perf_counter() - start_time
        print(
            f"[Elastic EP cold boot timing][{dp_size}]"
            f"[{mode}]"
            f"[{'async' if use_async_eplb else 'sync'}_eplb] "
            f"ready_seconds={ready_seconds:.3f} "
            f"first_completion_seconds={first_completion_seconds:.3f} "
            f"total_seconds={total_seconds:.3f}"
        )


@pytest.mark.parametrize(
    "use_async_eplb", [False, True], ids=["sync_eplb", "async_eplb"]
)
@multi_gpu_test(num_gpus=4)
def test_elastic_ep_scaling(use_async_eplb: bool):
    if use_async_eplb:
        from vllm.distributed.eplb.eplb_communicator import has_nixl

        if not has_nixl():
            pytest.skip("Async EPLB with elastic EP requires NIXL (not installed)")

    initial_dp_size = int(os.getenv("VLLM_TEST_ELASTIC_EP_INITIAL_DP", "2"))
    target_dp_size = int(os.getenv("VLLM_TEST_ELASTIC_EP_TARGET_DP", "4"))
    assert target_dp_size > initial_dp_size
    vllm_serve_args = _base_serve_args(use_async_eplb, initial_dp_size)
    mode = "enforce_eager" if "--enforce-eager" in vllm_serve_args else "cuda_graphs"

    with RemoteOpenAIServer(
        MODEL_NAME, vllm_serve_args, env_dict={}, max_wait_seconds=1200
    ) as server:
        initial_accuracy = _run_gsm8k_eval(server, f"Initial ({initial_dp_size} GPUs)")

        served_during_prepare = _prepare_and_scale_elastic_ep(
            server,
            initial_dp_size,
            target_dp_size,
            mode,
            use_async_eplb,
        )
        assert served_during_prepare, "prepare completed before serving was checked"
        time.sleep(10)
        scale_up_accuracy = _run_gsm8k_eval(
            server, f"After scale up ({target_dp_size} GPUs)"
        )

        assert scale_up_accuracy >= initial_accuracy - ACCURACY_TOL, (
            f"Scale up accuracy {scale_up_accuracy:.3f} dropped more than "
            f"{ACCURACY_TOL} below initial accuracy {initial_accuracy:.3f}"
        )

        _prepare_and_scale_elastic_ep(
            server,
            target_dp_size,
            initial_dp_size,
            mode,
            use_async_eplb,
        )
        time.sleep(5)
        scale_down_accuracy = _run_gsm8k_eval(
            server, f"After scale down ({initial_dp_size} GPUs)"
        )

        assert scale_down_accuracy >= initial_accuracy - ACCURACY_TOL, (
            f"Scale down accuracy {scale_down_accuracy:.3f} dropped more than "
            f"{ACCURACY_TOL} below initial accuracy {initial_accuracy:.3f}"
        )

        print("\nAccuracy Summary:")
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


@pytest.mark.parametrize(
    "use_async_eplb", [False, True], ids=["sync_eplb", "async_eplb"]
)
@multi_gpu_test(num_gpus=4)
def test_elastic_ep_scaling_uneven(use_async_eplb: bool):
    """Test scale up with uneven worker distribution.

    This tests the case where num_new_workers % old_dp_size != 0,
    specifically 2 -> 3 where remainder = 1 % 2 = 1.
    This exercises the remainder handling in sender-receiver pairing.
    """
    if use_async_eplb:
        from vllm.distributed.eplb.eplb_communicator import has_nixl

        if not has_nixl():
            pytest.skip("Async EPLB with elastic EP requires NIXL (not installed)")

    vllm_serve_args = _base_serve_args(use_async_eplb)

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
        time.sleep(10)
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
        time.sleep(5)
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
