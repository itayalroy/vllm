# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import sys
from types import SimpleNamespace

import pytest
import torch

import vllm.platforms as vllm_platforms
from vllm import envs
from vllm.platforms.interface import UnspecifiedPlatform


def _import_with_unspecified_platform(module_name: str):
    original_platform = vllm_platforms._current_platform
    vllm_platforms._current_platform = UnspecifiedPlatform()
    try:
        return importlib.import_module(module_name)
    finally:
        vllm_platforms._current_platform = original_platform


MoEActivation = _import_with_unspecified_platform(
    "vllm.model_executor.layers.fused_moe.activation"
).MoEActivation

NON_NVFP4_DTYPE = getattr(torch, "float8_e4m3fn", torch.int8)


def _import_prepare_finalize_modules(monkeypatch):
    monkeypatch.setitem(sys.modules, "deep_ep", SimpleNamespace(Buffer=object))
    monkeypatch.setitem(sys.modules, "nixl_ep", SimpleNamespace(Buffer=object))

    deepep_ll = _import_with_unspecified_platform(
        "vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ll"
    )
    nixl_ep = _import_with_unspecified_platform(
        "vllm.model_executor.layers.fused_moe.prepare_finalize.nixl_ep"
    )
    return deepep_ll, nixl_ep


def _quant_config(quant_dtype):
    return SimpleNamespace(
        quant_dtype=quant_dtype,
        a1_scale=torch.ones((), dtype=torch.float32),
        a2_scale=torch.ones((), dtype=torch.float32),
        per_act_token_quant=False,
        block_shape=None,
    )


def _capture_receiver_quant(monkeypatch, module):
    calls = []

    def fake_quantize_input(
        x,
        scale,
        quant_dtype,
        per_act_token_quant,
        block_shape,
    ):
        calls.append(
            {
                "quant_dtype": quant_dtype,
                "scale": scale,
                "per_act_token_quant": per_act_token_quant,
                "block_shape": block_shape,
            }
        )
        x_scales = None
        if quant_dtype is not None:
            x_scales = torch.ones((x.size(0),), dtype=torch.float32)
        return x, x_scales

    def fake_normalize_scales(x_scales, num_experts):
        calls[-1]["normalized_num_experts"] = num_experts
        return x_scales

    monkeypatch.setattr(module, "moe_kernel_quantize_input", fake_quantize_input)
    monkeypatch.setattr(module, "normalize_batched_scales_shape", fake_normalize_scales)
    return calls


@pytest.mark.parametrize(
    ("all2all_backend", "module_name", "class_name"),
    [
        (
            "deepep_low_latency",
            "deepep_ll",
            "DeepEPLLPrepareAndFinalize",
        ),
        (
            "nixl_ep",
            "nixl_ep",
            "NixlEPPrepareAndFinalize",
        ),
    ],
)
@pytest.mark.parametrize(
    ("quant_dtype", "expected_receiver_quant_dtype"),
    [
        pytest.param("nvfp4", None, id="nvfp4_raw_dispatch"),
        pytest.param(NON_NVFP4_DTYPE, NON_NVFP4_DTYPE, id="non_nvfp4_quantized"),
    ],
)
def test_receiver_quant_matches_upstream_deepep_contract(
    monkeypatch,
    all2all_backend,
    module_name,
    class_name,
    quant_dtype,
    expected_receiver_quant_dtype,
):
    monkeypatch.setattr(envs, "VLLM_DEEPEPLL_NVFP4_DISPATCH", False)
    deepep_ll, nixl_ep = _import_prepare_finalize_modules(monkeypatch)
    module = {"deepep_ll": deepep_ll, "nixl_ep": nixl_ep}[module_name]
    calls = _capture_receiver_quant(monkeypatch, module)

    prepare_finalize_cls = getattr(module, class_name)
    prepare_finalize = prepare_finalize_cls(
        buffer=object(),
        max_tokens_per_rank=4,
        num_dispatchers=1,
    )
    expert_x = torch.randn((2, 3, 8), dtype=torch.bfloat16)

    out, out_scales = prepare_finalize._do_quant(
        expert_x,
        torch.bfloat16,
        _quant_config(quant_dtype),
    )

    assert out.shape == expert_x.shape
    assert len(calls) == 1, all2all_backend
    call = calls[0]
    assert call["quant_dtype"] == expected_receiver_quant_dtype
    assert torch.equal(call["scale"], torch.ones((), dtype=torch.float32))
    assert call["per_act_token_quant"] is False
    assert call["block_shape"] is None
    if expected_receiver_quant_dtype is None:
        assert "normalized_num_experts" not in call
    else:
        assert call["normalized_num_experts"] == 2
    assert (out_scales is None) == (expected_receiver_quant_dtype is None)


def _cutedsl_quant_config(num_experts: int):
    return SimpleNamespace(
        quant_dtype="nvfp4",
        weight_quant_dtype="nvfp4",
        a1_gscale=torch.ones((num_experts,), dtype=torch.float32),
        a2_gscale=torch.ones((num_experts,), dtype=torch.float32),
        w1_scale=torch.ones((num_experts, 1, 1), dtype=torch.float32),
        w2_scale=torch.ones((num_experts, 1, 1), dtype=torch.float32),
        g1_alphas=torch.ones((num_experts,), dtype=torch.float32),
        g2_alphas=torch.ones((num_experts,), dtype=torch.float32),
    )


@pytest.mark.parametrize(
    ("all2all_backend", "use_deepep_nvfp4_dispatch", "expects_packed_inputs"),
    [
        ("deepep_low_latency", False, False),
        ("deepep_low_latency", True, True),
        ("nixl_ep", False, False),
        ("nixl_ep", True, False),
    ],
)
def test_cutedsl_batched_deepep_nvfp4_dispatch_env_is_backend_scoped(
    monkeypatch,
    all2all_backend,
    use_deepep_nvfp4_dispatch,
    expects_packed_inputs,
):
    cutedsl_module = _import_with_unspecified_platform(
        "vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_batched_moe"
    )

    monkeypatch.setattr(
        envs,
        "VLLM_DEEPEPLL_NVFP4_DISPATCH",
        use_deepep_nvfp4_dispatch,
    )

    num_experts = 2
    max_tokens = 3
    hidden_size = 8
    quant_config = _cutedsl_quant_config(num_experts)
    moe_config = SimpleNamespace(
        in_dtype=torch.bfloat16,
        use_deepep_ll_kernels=all2all_backend == "deepep_low_latency",
    )
    expert = cutedsl_module.FlashInferCuteDSLBatchedExperts(
        moe_config=moe_config,
        quant_config=quant_config,
        max_num_tokens=max_tokens,
        num_dispatchers=1,
    )

    workspace1, _, output_shape = expert.workspace_shapes(
        M=max_tokens,
        N=16,
        K=hidden_size,
        topk=1,
        global_num_experts=num_experts,
        local_num_experts=num_experts,
        expert_tokens_meta=None,
        activation=MoEActivation.SILU,
    )
    expected_hidden_dim = hidden_size * 2 if expects_packed_inputs else hidden_size
    assert workspace1[-1] == expected_hidden_dim
    assert output_shape[-1] == expected_hidden_dim

    captured = {}

    def fake_cutedsl_moe_masked(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        cutedsl_module,
        "flashinfer_cutedsl_moe_masked",
        fake_cutedsl_moe_masked,
    )

    hidden_states = torch.zeros(
        (num_experts, max_tokens, hidden_size), dtype=torch.bfloat16
    )
    a1q_scale = torch.ones((num_experts, max_tokens, 1), dtype=torch.float32)
    expert.apply(
        output=torch.empty(
            (num_experts, max_tokens, hidden_size), dtype=torch.bfloat16
        ),
        hidden_states=hidden_states,
        w1=torch.empty((num_experts, 1, 1), dtype=torch.uint8),
        w2=torch.empty((num_experts, 1, 1), dtype=torch.uint8),
        topk_weights=torch.ones((max_tokens, 1), dtype=torch.float32),
        topk_ids=torch.zeros((max_tokens, 1), dtype=torch.int64),
        activation=MoEActivation.SILU,
        global_num_experts=num_experts,
        expert_map=None,
        a1q_scale=a1q_scale,
        a2_scale=None,
        workspace13=None,
        workspace2=torch.empty((num_experts, max_tokens, 16), dtype=torch.bfloat16),
        expert_tokens_meta=SimpleNamespace(
            expert_num_tokens=torch.full((num_experts,), max_tokens, dtype=torch.int32)
        ),
        apply_router_weight_on_input=False,
    )

    if expects_packed_inputs:
        assert isinstance(captured["hidden_states"], tuple)
        assert captured["hidden_states"][0] is hidden_states
        assert captured["hidden_states"][1] is a1q_scale
        assert captured["input_global_scale"] is None
    else:
        assert captured["hidden_states"] is hidden_states
        assert captured["input_global_scale"] is quant_config.a1_gscale
