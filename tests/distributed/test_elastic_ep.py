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
    use_async_eplb: bool = False,
    data_parallel_size: int = 2,
    eplb_step_interval: int | None = 10,
    eplb_window_size: int | None = 5,
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
        "true" if use_async_eplb else "false",
        "--data-parallel-backend",
        "ray",
        "--data-parallel-size",
        str(data_parallel_size),
        "--api-server-count",
        "1",
    ]
    if eplb_step_interval is not None:
        args.extend(["--eplb-config.step_interval", str(eplb_step_interval)])
    if eplb_window_size is not None:
        args.extend(["--eplb-config.window_size", str(eplb_window_size)])

    leader_address = os.environ.get("LEADER_ADDRESS")
    if leader_address:
        args.extend(["--data-parallel-address", leader_address])

    return args


def _run_prepare_then_scale_timing(
    enforce_eager: bool,
    initial_dp_size: int,
    target_dp_size: int,
):
    vllm_serve_args = _base_serve_args(
        use_async_eplb=False,
        data_parallel_size=initial_dp_size,
        eplb_step_interval=None,
        eplb_window_size=None,
    )
    if enforce_eager:
        vllm_serve_args.append("--enforce-eager")

    with RemoteOpenAIServer(
        MODEL_NAME, vllm_serve_args, env_dict={}, max_wait_seconds=1200
    ) as server:
        initial_accuracy = _run_gsm8k_eval(server, f"Initial ({initial_dp_size} GPUs)")

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
                assert status_code != 503
                assert status_code == 200
                served_during_prepare = True
                time.sleep(0.5)

            prepare_ok, prepare_seconds, prepare_status = prepare_future.result()

        assert prepare_ok, f"prepare_elastic_ep failed with status {prepare_status}"
        assert served_during_prepare, "prepare completed before serving was checked"

        scale_ok, switch_seconds, scale_status = _post_elastic_ep_command_timed(
            server, "scale_elastic_ep", target_dp_size
        )
        total_seconds = time.perf_counter() - total_start

        mode = "enforce_eager" if enforce_eager else "cuda_graphs"
        print(
            f"[Elastic EP timing][{initial_dp_size}->{target_dp_size}][{mode}] "
            f"prepare_seconds={prepare_seconds:.3f} "
            f"switch_seconds={switch_seconds:.3f} "
            f"total_seconds={total_seconds:.3f}"
        )

        assert scale_ok, f"scale_elastic_ep failed with status {scale_status}"
        final_accuracy = _run_gsm8k_eval(
            server, f"After prepared scale up ({target_dp_size} GPUs)"
        )
        assert final_accuracy >= initial_accuracy - ACCURACY_TOL, (
            f"Prepared scale-up accuracy {final_accuracy:.3f} dropped more than "
            f"{ACCURACY_TOL} below initial accuracy {initial_accuracy:.3f}"
        )
        print(
            f"[Elastic EP accuracy][{initial_dp_size}->{target_dp_size}][{mode}] "
            f"initial={initial_accuracy:.3f} "
            f"final={final_accuracy:.3f} "
            f"diff={final_accuracy - initial_accuracy:+.3f}"
        )


@pytest.mark.parametrize(
    "enforce_eager", [False, True], ids=["cuda_graphs", "enforce_eager"]
)
@multi_gpu_test(num_gpus=4)
def test_elastic_ep_prepare_then_scale_timing(enforce_eager: bool):
    _run_prepare_then_scale_timing(enforce_eager, 2, 4)


@pytest.mark.parametrize(
    "enforce_eager", [False, True], ids=["cuda_graphs", "enforce_eager"]
)
@multi_gpu_test(num_gpus=8)
def test_elastic_ep_prepare_then_scale_timing_4_to_8(enforce_eager: bool):
    _run_prepare_then_scale_timing(enforce_eager, 4, 8)


@pytest.mark.parametrize(
    "use_async_eplb", [False, True], ids=["sync_eplb", "async_eplb"]
)
@multi_gpu_test(num_gpus=4)
def test_elastic_ep_scaling(use_async_eplb: bool):
    if use_async_eplb:
        from vllm.distributed.eplb.eplb_communicator import has_nixl

        if not has_nixl():
            pytest.skip("Async EPLB with elastic EP requires NIXL (not installed)")

    vllm_serve_args = _base_serve_args(use_async_eplb)

    with RemoteOpenAIServer(
        MODEL_NAME, vllm_serve_args, env_dict={}, max_wait_seconds=1200
    ) as server:
        initial_accuracy = _run_gsm8k_eval(server, "Initial (2 GPUs)")

        assert _send_scale_command(server, 4)
        time.sleep(10)
        scale_up_accuracy = _run_gsm8k_eval(server, "After scale up (4 GPUs)")

        assert scale_up_accuracy >= initial_accuracy - ACCURACY_TOL, (
            f"Scale up accuracy {scale_up_accuracy:.3f} dropped more than "
            f"{ACCURACY_TOL} below initial accuracy {initial_accuracy:.3f}"
        )

        assert _send_scale_command(server, 2)
        time.sleep(5)
        scale_down_accuracy = _run_gsm8k_eval(server, "After scale down (2 GPUs)")

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
        assert _send_scale_command(server, 3)
        time.sleep(10)
        scale_up_accuracy = _run_gsm8k_eval(server, "After scale up (3 GPUs)")

        assert scale_up_accuracy >= initial_accuracy - ACCURACY_TOL, (
            f"Scale up accuracy {scale_up_accuracy:.3f} dropped more than "
            f"{ACCURACY_TOL} below initial accuracy {initial_accuracy:.3f}"
        )

        # Scale back down to 2
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
