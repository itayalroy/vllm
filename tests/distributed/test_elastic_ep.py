# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import pytest
import requests

from vllm.utils.import_utils import has_nixl_ep

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
NIXL_EP_FAULT_TEST_TIMEOUT_MS = 2000
NIXL_EP_FAULT_TEST_POST_KILL_WAIT_S = 15
SMOKE_PROMPT = "The capital of France is"
FAULT_TRAFFIC_PROMPT = (
    "Explain why distributed inference needs resilient expert parallel "
    "communication. " * 32
)


def _send_scale_command(server: RemoteOpenAIServer, new_dp_size: int) -> bool:
    url = server.url_for("scale_elastic_ep")
    payload = {"new_data_parallel_size": new_dp_size}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=300)
        return response.status_code == 200
    except requests.exceptions.RequestException:
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


def _run_smoke_completion(server: RemoteOpenAIServer, stage: str) -> None:
    response = requests.post(
        server.url_for("v1/completions"),
        json={
            "model": MODEL_NAME,
            "prompt": SMOKE_PROMPT,
            "max_tokens": 8,
            "temperature": 0.0,
        },
        timeout=120,
    )
    response.raise_for_status()
    result = response.json()
    text = result["choices"][0]["text"]
    print(f"[{stage}] Smoke completion: {text!r}")


def _is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _wait_for_rank_pid(pid_dir: Path, rank: int, timeout_s: float) -> int:
    pid_file = pid_dir / f"rank_{rank}.pid"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            pid = int(pid_file.read_text().strip())
        except (FileNotFoundError, ValueError):
            time.sleep(0.5)
            continue

        if _is_process_alive(pid):
            return pid
        time.sleep(0.5)

    raise TimeoutError(f"Timed out waiting for live NIXL EP rank {rank} PID")


def _run_fault_completion_traffic(
    server: RemoteOpenAIServer,
    stop_event: threading.Event,
    errors: list[str],
    thread_idx: int,
) -> None:
    request_idx = 0
    while not stop_event.is_set():
        payload = {
            "model": MODEL_NAME,
            "prompt": f"{FAULT_TRAFFIC_PROMPT}\nRequest {thread_idx}-{request_idx}:",
            "max_tokens": 64,
            "temperature": 0.0,
        }
        try:
            response = requests.post(
                server.url_for("v1/completions"),
                json=payload,
                timeout=180,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            if len(errors) < 20:
                errors.append(f"thread={thread_idx} request={request_idx}: {exc!r}")
            time.sleep(0.5)
        request_idx += 1


@multi_gpu_test(num_gpus=4)
def test_elastic_ep_scaling():
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


@pytest.mark.skipif(
    os.environ.get("VLLM_RUN_NIXL_EP_FAULT_TEST") != "1",
    reason=("manual NIXL EP fault-injection test; set VLLM_RUN_NIXL_EP_FAULT_TEST=1"),
)
@pytest.mark.skipif(not has_nixl_ep(), reason="nixl_ep is not installed")
@pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="requires SIGKILL")
@multi_gpu_test(num_gpus=4)
def test_nixl_ep_fault_detection_mask_manual(tmp_path: Path):
    pid_dir = tmp_path / "nixl_ep_rank_pids"
    trigger_path = tmp_path / "nixl_ep_fault_trigger"
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
        "nixl_ep",
        "--enable-elastic-ep",
        "--enable-eplb",
        "--eplb-config.num_redundant_experts",
        "0",
        "--data-parallel-backend",
        "ray",
        "--data-parallel-size",
        "4",
        "--api-server-count",
        "1",
    ]

    leader_address = os.environ.get("LEADER_ADDRESS")
    if leader_address:
        vllm_serve_args.extend(["--data-parallel-address", leader_address])

    env_dict = {
        "RAY_DEDUP_LOGS": "0",
        "VLLM_NIXL_EP_DEBUG_RANK_PID_DIR": str(pid_dir),
        "VLLM_NIXL_EP_LOG_MASK_AFTER_FORWARD": "1",
        "VLLM_NIXL_EP_MAX_NUM_RANKS": "4",
        "VLLM_NIXL_EP_TIMEOUT_MS": str(NIXL_EP_FAULT_TEST_TIMEOUT_MS),
        "VLLM_NIXL_EP_DELAY_ON_ACTOR_DIED_SECONDS": "10",
        "VLLM_NIXL_EP_FAULT_INJECTION_RANK": "1",
        "VLLM_NIXL_EP_FAULT_INJECTION_TRIGGER_PATH": str(trigger_path),
        "VLLM_KEEP_ALIVE_ON_ENGINE_DEATH": "1",
        "VLLM_USE_RAY_V2_EXECUTOR_BACKEND": "1",
    }

    with RemoteOpenAIServer(
        MODEL_NAME, vllm_serve_args, env_dict=env_dict, max_wait_seconds=1200
    ) as server:
        _run_smoke_completion(server, "Before rank 1 kill")
        rank1_pid = _wait_for_rank_pid(pid_dir, rank=1, timeout_s=300)
        print(f"[NIXL EP fault test] ep_rank=1 pid={rank1_pid}")

        stop_event = threading.Event()
        errors: list[str] = []
        threads = [
            threading.Thread(
                target=_run_fault_completion_traffic,
                args=(server, stop_event, errors, thread_idx),
                daemon=True,
            )
            for thread_idx in range(8)
        ]
        for thread in threads:
            thread.start()

        time.sleep(3)
        trigger_path.write_text("armed\n")
        print(
            "[NIXL EP fault test] Armed post-DP-coordination self-SIGKILL for "
            "ep_rank=1; inspect NIXL_EP_MASK logs on ranks 0, 2, and 3"
        )

        time.sleep(NIXL_EP_FAULT_TEST_POST_KILL_WAIT_S)
        stop_event.set()
        for thread in threads:
            thread.join(timeout=5)

        print(f"[NIXL EP fault test] Background traffic errors: {len(errors)}")
        for error in errors[:5]:
            print(f"[NIXL EP fault test] {error}")
