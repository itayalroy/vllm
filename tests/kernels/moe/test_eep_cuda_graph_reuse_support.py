# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
from types import SimpleNamespace

import pytest

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.eep_reconfigure import _make_eep_experts
from vllm.model_executor.layers.fused_moe.experts import (
    batched_deep_gemm_moe,
    cutlass_moe,
    flashinfer_cutedsl_batched_moe,
    fused_batched_moe,
    fused_humming_moe,
    marlin_moe,
)
from vllm.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe import (
    BatchedDeepGemmExperts,
)
from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
    CutlassBatchedExpertsFp8,
)
from vllm.model_executor.layers.fused_moe.experts.fused_batched_moe import (
    BatchedTritonExperts,
    NaiveBatchedExperts,
)
from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
    BatchedHummingGroupedExperts,
)
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
    BatchedMarlinExperts,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    backend_to_kernel_cls as fp8_experts,
)
from vllm.model_executor.layers.fused_moe.oracle.int8 import (
    Int8MoeBackend,
)
from vllm.model_executor.layers.fused_moe.oracle.int8 import (
    backend_to_kernel_cls as int8_experts,
)
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    WNA16MoEBackend,
)
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    backend_to_kernel_cls as wna16_experts,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    backend_to_kernel_cls as mxfp4_experts,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp8 import (
    _mxfp8_backend_to_kernel_cls,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    backend_to_kernel_cls as nvfp4_experts,
)
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
)
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    backend_to_kernel_cls as unquantized_experts,
)
from vllm.model_executor.layers.fused_moe.oracle.w4a8 import (
    W4A8MoeBackend,
)
from vllm.model_executor.layers.fused_moe.oracle.w4a8 import (
    backend_to_kernel_cls as w4a8_experts,
)

FlashInferCuteDSLBatchedExperts = (
    flashinfer_cutedsl_batched_moe.FlashInferCuteDSLBatchedExperts
)

BATCHED_EXPERTS = {
    BatchedDeepGemmExperts,
    BatchedHummingGroupedExperts,
    BatchedMarlinExperts,
    BatchedTritonExperts,
    CutlassBatchedExpertsFp8,
    FlashInferCuteDSLBatchedExperts,
    NaiveBatchedExperts,
}


def test_all_batched_expert_implementations_are_classified():
    modules = (
        batched_deep_gemm_moe,
        cutlass_moe,
        flashinfer_cutedsl_batched_moe,
        fused_batched_moe,
        fused_humming_moe,
        marlin_moe,
    )
    discovered = {
        expert
        for module in modules
        for expert in vars(module).values()
        if inspect.isclass(expert)
        and not inspect.isabstract(expert)
        and expert.__module__ == module.__name__
        and issubclass(expert, mk.FusedMoEExpertsModular)
        and expert.activation_format() == mk.FusedMoEActivationFormat.BatchedExperts
    }

    assert discovered == BATCHED_EXPERTS, (
        f"unclassified={discovered - BATCHED_EXPERTS}, "
        f"not_discovered={BATCHED_EXPERTS - discovered}"
    )


@pytest.mark.parametrize(
    ("experts", "unsupported_args"),
    [
        (BatchedDeepGemmExperts, set()),
        (BatchedTritonExperts, set()),
        (CutlassBatchedExpertsFp8, set()),
        (FlashInferCuteDSLBatchedExperts, set()),
        (NaiveBatchedExperts, set()),
        (BatchedHummingGroupedExperts, {"layer"}),
        (
            BatchedMarlinExperts,
            {
                "w13_g_idx",
                "w2_g_idx",
                "w13_g_idx_sort_indices",
                "w2_g_idx_sort_indices",
                "is_k_full",
            },
        ),
    ],
)
def test_eep_generic_staging_constructor_contract(experts, unsupported_args):
    generic_args = set(inspect.signature(mk.FusedMoEExperts.__init__).parameters)
    expert_args = set(inspect.signature(experts.__init__).parameters)

    assert expert_args - generic_args == unsupported_args


@pytest.mark.parametrize(
    "experts",
    [BatchedHummingGroupedExperts, BatchedMarlinExperts],
)
def test_eep_staging_rejects_experts_with_extra_constructor_state(experts):
    quant_method = SimpleNamespace(moe_quant_config=object())
    prepare_finalize = SimpleNamespace(
        activation_format=mk.FusedMoEActivationFormat.BatchedExperts,
        max_num_tokens_per_rank=lambda: 1,
        num_dispatchers=lambda: 2,
    )

    with pytest.raises(NotImplementedError, match="do not support Elastic EP"):
        _make_eep_experts(
            quant_method,
            object.__new__(experts),
            prepare_finalize,
            object(),
        )


@pytest.mark.parametrize(
    ("quant_family", "experts", "expected_batched"),
    [
        (
            "unquantized",
            unquantized_experts(UnquantizedMoeBackend.BATCHED_TRITON),
            {BatchedTritonExperts},
        ),
        (
            "fp8_block",
            fp8_experts(Fp8MoeBackend.BATCHED_DEEPGEMM),
            {BatchedDeepGemmExperts},
        ),
        (
            "fp8_triton",
            fp8_experts(Fp8MoeBackend.BATCHED_TRITON),
            {BatchedTritonExperts},
        ),
        (
            "fp8_cutlass",
            fp8_experts(Fp8MoeBackend.BATCHED_VLLM_CUTLASS),
            {CutlassBatchedExpertsFp8},
        ),
        (
            "fp8_humming",
            fp8_experts(Fp8MoeBackend.HUMMING),
            {BatchedHummingGroupedExperts},
        ),
        (
            "int8",
            int8_experts(Int8MoeBackend.HUMMING),
            {BatchedHummingGroupedExperts},
        ),
        (
            "wna16_marlin",
            wna16_experts(WNA16MoEBackend.BATCHED_MARLIN),
            {BatchedMarlinExperts},
        ),
        (
            "wna16_humming",
            wna16_experts(WNA16MoEBackend.HUMMING),
            {BatchedHummingGroupedExperts},
        ),
        (
            "mxfp4_marlin",
            mxfp4_experts(Mxfp4MoeBackend.BATCHED_MARLIN),
            {BatchedMarlinExperts},
        ),
        (
            "mxfp4_humming",
            mxfp4_experts(Mxfp4MoeBackend.HUMMING),
            {BatchedHummingGroupedExperts},
        ),
        (
            "mxfp8",
            _mxfp8_backend_to_kernel_cls(Fp8MoeBackend.HUMMING),
            {BatchedHummingGroupedExperts},
        ),
        (
            "nvfp4_cutedsl",
            nvfp4_experts(NvFp4MoeBackend.FLASHINFER_CUTEDSL_BATCHED),
            {FlashInferCuteDSLBatchedExperts},
        ),
        (
            "nvfp4_humming",
            nvfp4_experts(NvFp4MoeBackend.HUMMING),
            {BatchedHummingGroupedExperts},
        ),
        ("w4a8", w4a8_experts(W4A8MoeBackend.CUTLASS), set()),
    ],
)
def test_quant_family_batched_experts_are_classified(
    quant_family, experts, expected_batched
):
    del quant_family
    batched = {
        expert
        for expert in experts
        if expert.activation_format() == mk.FusedMoEActivationFormat.BatchedExperts
    }
    assert batched == expected_batched


def test_cuda_graph_reuse_classification_is_complete():
    reusable = {
        BatchedDeepGemmExperts,
        BatchedTritonExperts,
        FlashInferCuteDSLBatchedExperts,
    }
    owns_graph_visible_tensors = {CutlassBatchedExpertsFp8}
    cannot_stage = {BatchedHummingGroupedExperts, BatchedMarlinExperts}
    reference_only = {NaiveBatchedExperts}

    assert reusable | owns_graph_visible_tensors | cannot_stage | reference_only == (
        BATCHED_EXPERTS
    )
    assert not (
        reusable & owns_graph_visible_tensors
        or reusable & cannot_stage
        or reusable & reference_only
        or owns_graph_visible_tensors & cannot_stage
        or owns_graph_visible_tensors & reference_only
        or cannot_stage & reference_only
    )
