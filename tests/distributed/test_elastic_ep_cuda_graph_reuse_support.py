# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests

from ..utils import RemoteOpenAIServer, multi_gpu_test

PROMPTS = (
    "The capital of France is",
    "Two plus two equals",
    "The opposite of hot is",
    "Complete this sequence: 1, 2, 3,",
)


def _serve_args(initial_dp_size: int) -> list[str]:
    args = [
        "--trust-remote-code",
        "--tensor-parallel-size",
        os.getenv("VLLM_TEST_ELASTIC_EP_TP", "1"),
        "--data-parallel-size",
        str(initial_dp_size),
        "--data-parallel-backend",
        "ray",
        "--enable-expert-parallel",
        "--all2all-backend",
        "nixl_ep",
        "--enable-elastic-ep",
        "--enable-eplb",
        "--eplb-config.num_redundant_experts",
        "0",
        "--eplb-config.use_async",
        "true",
        "--eplb-config.step_interval",
        "100000",
        "--gpu-memory-utilization",
        "0.8",
        "--max-model-len",
        os.getenv("VLLM_TEST_ELASTIC_EP_MAX_MODEL_LEN", "1024"),
        "--max-num-seqs",
        os.getenv("VLLM_TEST_ELASTIC_EP_MAX_NUM_SEQS", "8"),
        "--no-enable-flashinfer-autotune",
        "--expert-placement-strategy",
        os.getenv("VLLM_TEST_ELASTIC_EP_PLACEMENT", "linear"),
    ]
    if quantization := os.getenv("VLLM_TEST_ELASTIC_EP_QUANTIZATION"):
        args.extend(("--quantization", quantization))
    if moe_backend := os.getenv("VLLM_TEST_ELASTIC_EP_MOE_BACKEND"):
        args.extend(("--moe-backend", moe_backend))
    return args


def _completion(server: RemoteOpenAIServer, rank: int, prompt: str) -> str:
    response = requests.post(
        server.url_for("v1/completions"),
        headers={"X-data-parallel-rank": str(rank)},
        json={
            "model": os.environ["VLLM_TEST_ELASTIC_EP_MODEL"],
            "prompt": prompt,
            "max_tokens": 16,
            "temperature": 0,
            "seed": 0,
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["text"]


def _rank_outputs(
    server: RemoteOpenAIServer, ranks: range
) -> dict[int, tuple[str, ...]]:
    with ThreadPoolExecutor(max_workers=len(ranks) * len(PROMPTS)) as executor:
        futures = {
            (rank, prompt_index): executor.submit(_completion, server, rank, prompt)
            for rank in ranks
            for prompt_index, prompt in enumerate(PROMPTS)
        }
    return {
        rank: tuple(futures[rank, i].result() for i in range(len(PROMPTS)))
        for rank in ranks
    }


def _assert_outputs_match_baseline(
    baseline: tuple[set[str], ...], outputs: dict[int, tuple[str, ...]]
) -> None:
    mismatches = {
        (rank, prompt_index): output
        for rank, rank_outputs in outputs.items()
        for prompt_index, output in enumerate(rank_outputs)
        if output not in baseline[prompt_index]
    }
    assert not mismatches, f"baseline={baseline!r}, mismatches={mismatches!r}"


def _scale(server: RemoteOpenAIServer, new_dp_size: int) -> None:
    response = requests.post(
        server.url_for("scale_elastic_ep"),
        json={"new_data_parallel_size": new_dp_size},
        timeout=1800,
    )
    response.raise_for_status()


@pytest.mark.skipif(
    "VLLM_TEST_ELASTIC_EP_MODEL" not in os.environ,
    reason="requires an explicit Elastic EP model",
)
@multi_gpu_test(num_gpus=4)
def test_elastic_ep_cuda_graph_reuse_support():
    model = os.environ["VLLM_TEST_ELASTIC_EP_MODEL"]
    initial_dp_size = int(os.getenv("VLLM_TEST_ELASTIC_EP_INITIAL_DP", "2"))
    target_dp_size = int(os.getenv("VLLM_TEST_ELASTIC_EP_TARGET_DP", "4"))

    with RemoteOpenAIServer(
        model,
        _serve_args(initial_dp_size),
        env_dict={},
        max_wait_seconds=1800,
    ) as server:
        initial_ranks = range(initial_dp_size)
        initial = _rank_outputs(server, initial_ranks)
        baseline = tuple(
            {outputs[prompt_index] for outputs in initial.values()}
            for prompt_index in range(len(PROMPTS))
        )

        _scale(server, target_dp_size)
        print("Checking newly prepared ranks", flush=True)
        _assert_outputs_match_baseline(
            baseline, _rank_outputs(server, range(initial_dp_size, target_dp_size))
        )
        print("Checking retained ranks with reused CUDA graphs", flush=True)
        _assert_outputs_match_baseline(baseline, _rank_outputs(server, initial_ranks))
        time.sleep(2)
        print("Checking all ranks after EPLB", flush=True)
        _assert_outputs_match_baseline(
            baseline, _rank_outputs(server, range(target_dp_size))
        )

        _scale(server, initial_dp_size)
        _assert_outputs_match_baseline(baseline, _rank_outputs(server, initial_ranks))
