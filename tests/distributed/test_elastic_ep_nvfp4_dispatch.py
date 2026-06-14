# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
import time

import pytest
import requests
import torch

from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_nixl_ep

from ..utils import RemoteOpenAIServer, multi_gpu_test

BF16_MODEL = "deepseek-ai/DeepSeek-V2-Lite-Chat"
NVFP4_MODEL = "nvidia/DeepSeek-R1-0528-FP4-v2"

MAX_NUM_SEQS = 16
INITIAL_DP_SIZE = 2
SCALE_UP_DP_SIZE = 4
SMOKE_PROMPT = "The capital of France is"

HF_OVERRIDES = {
    "bf16": {
        "num_hidden_layers": 2,
    },
    "nvfp4": {
        "num_layers": 2,
        "num_hidden_layers": 2,
    },
}


@pytest.fixture
def cleanup_ray_between_tests():
    """Force-stop any lingering Ray processes between tests."""
    subprocess.run(["ray", "stop", "--force"], timeout=30, capture_output=True)
    time.sleep(5)
    yield


def _make_quant_config(quant_dtype: torch.dtype | str | None) -> FusedMoEQuantConfig:
    scale = torch.ones(1, dtype=torch.float32) if quant_dtype is not None else None
    return FusedMoEQuantConfig.make(
        quant_dtype=quant_dtype,
        a1_scale=scale,
        a2_scale=scale,
        w1_scale=scale,
        w2_scale=scale,
        a1_gscale=scale if quant_dtype == "nvfp4" else None,
        a2_gscale=scale if quant_dtype == "nvfp4" else None,
        g1_alphas=scale if quant_dtype == "nvfp4" else None,
        g2_alphas=scale if quant_dtype == "nvfp4" else None,
    )


@pytest.mark.parametrize(
    ("quant_dtype", "expected_quantizer_dtype"),
    [
        ("nvfp4", None),
        (torch.float8_e4m3fn, torch.float8_e4m3fn),
        (None, None),
    ],
    ids=["nvfp4_raw", "fp8_quantized", "unquantized"],
)
def test_nixl_ep_receiver_uses_raw_dispatch_for_nvfp4(
    monkeypatch: pytest.MonkeyPatch,
    quant_dtype: torch.dtype | str | None,
    expected_quantizer_dtype: torch.dtype | str | None,
):
    nixl_ep = pytest.importorskip(
        "vllm.model_executor.layers.fused_moe.prepare_finalize.nixl_ep"
    )

    captured: dict[str, torch.dtype | str | None] = {}

    def fake_quantize_input(
        x: torch.Tensor,
        a_scale: torch.Tensor | None,
        q_dtype: torch.dtype | str | None,
        per_act_token_quant: bool,
        block_shape: list[int] | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        captured["q_dtype"] = q_dtype
        x_scales = torch.ones(1, dtype=torch.float32) if q_dtype is not None else None
        return x, x_scales

    monkeypatch.setattr(nixl_ep, "moe_kernel_quantize_input", fake_quantize_input)

    prepare_finalize = nixl_ep.NixlEPPrepareAndFinalize.__new__(
        nixl_ep.NixlEPPrepareAndFinalize
    )
    prepare_finalize.use_fp8_dispatch = False

    hidden_states = torch.randn(2, 3, 16, dtype=torch.bfloat16)
    prepare_finalize._do_quant(
        hidden_states,
        torch.bfloat16,
        _make_quant_config(quant_dtype),
    )

    assert captured["q_dtype"] == expected_quantizer_dtype


@pytest.mark.parametrize(
    ("all2all_backend", "expected"),
    [
        ("deepep_low_latency", True),
        ("nixl_ep", False),
    ],
    ids=["deepep_ll", "nixl_ep"],
)
def test_deepep_packed_nvfp4_env_is_scoped_to_deepep_ll(
    monkeypatch: pytest.MonkeyPatch,
    all2all_backend: str,
    expected: bool,
):
    from types import SimpleNamespace

    from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_batched_moe import (  # noqa: E501
        FlashInferCuteDSLBatchedExperts,
    )

    monkeypatch.setenv("VLLM_DEEPEPLL_NVFP4_DISPATCH", "1")
    moe_config = SimpleNamespace(
        in_dtype=torch.bfloat16,
        use_deepep_ll_kernels=all2all_backend == "deepep_low_latency",
    )

    experts = FlashInferCuteDSLBatchedExperts(
        moe_config,  # type: ignore[arg-type]
        _make_quant_config("nvfp4"),
        max_num_tokens=8,
        num_dispatchers=1,
    )

    assert experts.use_deep_ep_ll_nvfp4_dispatch is expected


def _skip_if_backend_unavailable(all2all_backend: str) -> None:
    if all2all_backend == "nixl_ep" and not has_nixl_ep():
        pytest.skip("NIXL EP kernels are not installed")


def _skip_if_quant_unavailable(quantization: str) -> None:
    if quantization != "nvfp4":
        return
    if not current_platform.is_device_capability_family(100):
        pytest.skip("NVFP4 MoE requires Blackwell GPUs")

    from vllm.utils.flashinfer import has_flashinfer_cutedsl_grouped_gemm_nt_masked

    if not has_flashinfer_cutedsl_grouped_gemm_nt_masked():
        pytest.skip("FlashInfer CUTEDSL grouped GEMM is not available")


def _build_vllm_serve_args(all2all_backend: str, quantization: str) -> list[str]:
    args = [
        "--trust-remote-code",
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        "0.8",
        "--max-model-len",
        "1024",
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--max-num-batched-tokens",
        "256",
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
        str(INITIAL_DP_SIZE),
        "--api-server-count",
        "1",
        "--load-format",
        "dummy",
    ]

    if quantization == "nvfp4":
        args.extend(
            [
                "--moe-backend",
                "flashinfer_cutedsl",
            ]
        )

    leader_address = os.environ.get("LEADER_ADDRESS")
    if leader_address:
        args.extend(["--data-parallel-address", leader_address])

    return args


def _send_scale_command(server: RemoteOpenAIServer, new_dp_size: int) -> bool:
    response = requests.post(
        server.url_for("scale_elastic_ep"),
        json={"new_data_parallel_size": new_dp_size},
        headers={"Content-Type": "application/json"},
        timeout=300,
    )
    return response.status_code == 200


def _run_smoke_completion(
    server: RemoteOpenAIServer,
    model: str,
    stage: str,
) -> None:
    response = requests.post(
        server.url_for("v1/completions"),
        json={
            "model": model,
            "prompt": SMOKE_PROMPT,
            "max_tokens": 8,
            "temperature": 0.0,
        },
        timeout=120,
    )
    response.raise_for_status()
    choices = response.json()["choices"]
    assert choices
    text = choices[0]["text"]
    print(f"[{stage}] Smoke completion: {text!r}")
    assert text is not None


def _elastic_ep_smoke(
    all2all_backend: str,
    quantization: str,
) -> None:
    _skip_if_backend_unavailable(all2all_backend)
    _skip_if_quant_unavailable(quantization)

    model = NVFP4_MODEL if quantization == "nvfp4" else BF16_MODEL
    serve_args = _build_vllm_serve_args(all2all_backend, quantization)
    env_dict = {"VLLM_DEEPEPLL_NVFP4_DISPATCH": "0"}

    with RemoteOpenAIServer(
        model,
        serve_args,
        env_dict=env_dict,
        max_wait_seconds=1500,
        override_hf_configs=HF_OVERRIDES[quantization],
    ) as server:
        _run_smoke_completion(server, model, "Initial")

        assert _send_scale_command(server, SCALE_UP_DP_SIZE)
        time.sleep(10)
        _run_smoke_completion(server, model, "After scale up")

        assert _send_scale_command(server, INITIAL_DP_SIZE)
        time.sleep(5)
        _run_smoke_completion(server, model, "After scale down")


@pytest.mark.parametrize(
    ("all2all_backend", "quantization"),
    [
        ("nixl_ep", "bf16"),
        ("nixl_ep", "nvfp4"),
    ],
    ids=[
        "nixl_ep_bf16",
        "nixl_ep_nvfp4",
    ],
)
@pytest.mark.usefixtures("cleanup_ray_between_tests")
@multi_gpu_test(num_gpus=4)
def test_nixl_ep_elastic_scaling_nvfp4_dispatch_matrix(
    all2all_backend: str,
    quantization: str,
):
    _elastic_ep_smoke(all2all_backend, quantization)
