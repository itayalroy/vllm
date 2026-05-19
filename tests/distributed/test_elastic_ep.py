# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import subprocess
import time

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
SMOKE_PROMPT = "The capital of France is"

DECODE_BENCH_KV_TRANSFER_CONFIG = {
    "kv_connector": "DecodeBenchConnector",
    "kv_role": "kv_both",
}
FULL_DECODE_ONLY_COMPILATION_CONFIG = {
    "cudagraph_mode": "FULL_DECODE_ONLY",
}


def _send_scale_command(
    server: RemoteOpenAIServer, new_dp_size: int, stage: str
) -> bool:
    url = server.url_for("scale_elastic_ep")
    payload = {"new_data_parallel_size": new_dp_size}
    headers = {"Content-Type": "application/json"}

    start_time = time.perf_counter()
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=300)
        elapsed = time.perf_counter() - start_time
        print(
            f"[Elastic EP] {stage} to DP size {new_dp_size} returned "
            f"{response.status_code} in {elapsed:.2f}s"
        )
        return response.status_code == 200
    except requests.exceptions.RequestException as err:
        elapsed = time.perf_counter() - start_time
        print(
            f"[Elastic EP] {stage} to DP size {new_dp_size} failed "
            f"after {elapsed:.2f}s: {err}"
        )
        return False


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


def _run_decode_bench_smoke_completion(server: RemoteOpenAIServer, stage: str) -> None:
    response = requests.post(
        server.url_for("v1/completions"),
        json={
            "model": MODEL_NAME,
            "prompt": SMOKE_PROMPT,
            "max_tokens": 8,
            "temperature": 0.0,
            "ignore_eos": True,
        },
        timeout=60,
    )
    response.raise_for_status()
    result = response.json()
    text = result["choices"][0]["text"]
    print(f"[{stage}] DecodeBench smoke completion: {text!r}")
    assert isinstance(text, str)


@pytest.mark.parametrize("all2all_backend", ["allgather_reducescatter", "nixl_ep"])
@multi_gpu_test(num_gpus=4)
def test_elastic_ep_scaling(all2all_backend: str):
    vllm_serve_args = [
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
        "--data-parallel-backend",
        "ray",
        "--data-parallel-size",
        "2",
        "--api-server-count",
        "1",
    ]

    leader_address = os.environ.get("LEADER_ADDRESS")
    if leader_address:
        vllm_serve_args.extend(["--data-parallel-address", leader_address])

    with RemoteOpenAIServer(
        MODEL_NAME, vllm_serve_args, env_dict={}, max_wait_seconds=1200
    ) as server:
        initial_accuracy = _run_gsm8k_eval(
            server, f"Initial (2 GPUs, {all2all_backend})"
        )

        assert _send_scale_command(server, 4, "Scale up")
        time.sleep(10)
        scale_up_accuracy = _run_gsm8k_eval(
            server, f"After scale up (4 GPUs, {all2all_backend})"
        )

        assert scale_up_accuracy >= initial_accuracy - ACCURACY_TOL, (
            f"Scale up accuracy {scale_up_accuracy:.3f} dropped more than "
            f"{ACCURACY_TOL} below initial accuracy {initial_accuracy:.3f}"
        )

        assert _send_scale_command(server, 2, "Scale down")
        time.sleep(5)
        scale_down_accuracy = _run_gsm8k_eval(
            server, f"After scale down (2 GPUs, {all2all_backend})"
        )

        assert scale_down_accuracy >= initial_accuracy - ACCURACY_TOL, (
            f"Scale down accuracy {scale_down_accuracy:.3f} dropped more than "
            f"{ACCURACY_TOL} below initial accuracy {initial_accuracy:.3f}"
        )

        print("\nAccuracy Summary:")
        print(f"  Backend:    {all2all_backend}")
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


@multi_gpu_test(num_gpus=4)
def test_elastic_ep_decode_bench_full_decode_only():
    vllm_serve_args = [
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
        "--data-parallel-backend",
        "ray",
        "--data-parallel-size",
        "2",
        "--api-server-count",
        "1",
        "--kv-transfer-config",
        json.dumps(DECODE_BENCH_KV_TRANSFER_CONFIG),
        "--compilation-config",
        json.dumps(FULL_DECODE_ONLY_COMPILATION_CONFIG),
    ]

    leader_address = os.environ.get("LEADER_ADDRESS")
    if leader_address:
        vllm_serve_args.extend(["--data-parallel-address", leader_address])

    with RemoteOpenAIServer(
        MODEL_NAME, vllm_serve_args, env_dict={}, max_wait_seconds=1200
    ) as server:
        _run_decode_bench_smoke_completion(server, "Initial decode benchmark")

        assert _send_scale_command(server, 4, "Scale up")
        time.sleep(10)
        _run_decode_bench_smoke_completion(server, "After scale up decode benchmark")

        assert _send_scale_command(server, 2, "Scale down")
        time.sleep(5)
        _run_decode_bench_smoke_completion(server, "After scale down decode benchmark")


@multi_gpu_test(num_gpus=4)
def test_elastic_ep_scaling_uneven():
    """Test scale up with uneven worker distribution.

    This tests the case where num_new_workers % old_dp_size != 0,
    specifically 2 -> 3 where remainder = 1 % 2 = 1.
    This exercises the remainder handling in sender-receiver pairing.
    """
    vllm_serve_args = [
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
        "--data-parallel-backend",
        "ray",
        "--data-parallel-size",
        "2",
        "--api-server-count",
        "1",
    ]

    leader_address = os.environ.get("LEADER_ADDRESS")
    if leader_address:
        vllm_serve_args.extend(["--data-parallel-address", leader_address])

    with RemoteOpenAIServer(
        MODEL_NAME, vllm_serve_args, env_dict={}, max_wait_seconds=1200
    ) as server:
        initial_accuracy = _run_gsm8k_eval(server, "Initial (2 GPUs)")

        # Scale 2 -> 3: This has remainder = 1 % 2 = 1
        # Tests uneven sender-receiver pairing
        assert _send_scale_command(server, 3, "Scale up")
        time.sleep(10)
        scale_up_accuracy = _run_gsm8k_eval(server, "After scale up (3 GPUs)")

        assert scale_up_accuracy >= initial_accuracy - ACCURACY_TOL, (
            f"Scale up accuracy {scale_up_accuracy:.3f} dropped more than "
            f"{ACCURACY_TOL} below initial accuracy {initial_accuracy:.3f}"
        )

        # Scale back down to 2
        assert _send_scale_command(server, 2, "Scale down")
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
