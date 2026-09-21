# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
import torch

from vllm.distributed.eplb.eplb_state import (
    EplbState,
    _commit_eplb_maps,
    _commit_eplb_maps_for_layer,
)


def _make_model_state(
    phy2log: torch.Tensor,
    log2phy: torch.Tensor,
    logcnt: torch.Tensor,
    phy2log_storage: torch.Tensor | None = None,
) -> MagicMock:
    """Build a minimal EplbModelState mock with its map tensors."""
    state = MagicMock()
    state.physical_to_logical_map = phy2log
    state.physical_to_logical_map_buffer = (
        phy2log if phy2log_storage is None else phy2log_storage
    )
    state.logical_to_physical_map = log2phy
    state.logical_replica_count = logcnt
    state.expert_load_window = torch.zeros((3, *phy2log.shape), dtype=torch.int32)
    state.expert_load_samples = 0
    state.expert_load_valid_after = torch.zeros(phy2log.shape[0], dtype=torch.int64)
    state.last_expert_load = None
    return state


def test_commit_eplb_maps_shape_change():
    """When the number of physical experts changes, resize the active map within its
    preallocated storage.
    """
    num_layers, num_logical, num_physical = 2, 4, 6
    max_replicas = 3

    # Build current state tensors
    storage = torch.full((num_layers, num_physical + 2), -1, dtype=torch.long)
    model_state = _make_model_state(
        phy2log=storage[:, :num_physical],
        phy2log_storage=storage,
        log2phy=torch.full(
            (num_layers, num_logical, max_replicas), -1, dtype=torch.long
        ),
        logcnt=torch.zeros(num_layers, num_logical, dtype=torch.long),
    )

    # The new map has two more physical experts. These new physical experts will
    # automatically map to the first two logical experts
    new_phy2log_larger = (
        (torch.arange(num_physical + 2, dtype=torch.long) % num_logical)
        .unsqueeze(0)
        .expand(num_layers, -1)
    )
    _commit_eplb_maps(model_state, new_phy2log_larger)

    # Check that the number of physical experts has been updated and that the values
    # match
    assert model_state.physical_to_logical_map.shape[1] == num_physical + 2
    assert model_state.physical_to_logical_map.data_ptr() == storage.data_ptr()
    assert torch.equal(model_state.physical_to_logical_map, new_phy2log_larger)


def test_commit_eplb_maps_for_layer_logical_padding():
    """Test that logical_to_physical_map is padded with -1 to fill the
    pre-allocated slots when the new map has fewer replicas than the max.
    """
    num_layers, num_logical, num_physical = 2, 4, 6
    max_replicas = 3

    model_state = _make_model_state(
        phy2log=torch.zeros(num_layers, num_physical, dtype=torch.long),
        log2phy=torch.full(
            (num_layers, num_logical, max_replicas), -1, dtype=torch.long
        ),
        logcnt=torch.zeros(num_layers, num_logical, dtype=torch.long),
    )

    new_phy2log = (
        (torch.arange(num_physical, dtype=torch.long) % num_logical)
        .unsqueeze(0)
        .expand(num_layers, -1)
        .contiguous()
    )
    layer = 0
    _commit_eplb_maps_for_layer(model_state, new_phy2log[layer], layer)

    assert torch.all(model_state.logical_to_physical_map[layer, :, 2] == -1)


def test_commit_eplb_maps_for_layer_shape_assert():
    """Test that a mismatched number of physical experts triggers an assertion error."""
    num_layers, num_logical, num_physical = 2, 4, 6

    model_state = _make_model_state(
        phy2log=torch.zeros(num_layers, num_physical, dtype=torch.long),
        log2phy=torch.full((num_layers, num_logical, 2), -1, dtype=torch.long),
        logcnt=torch.zeros(num_layers, num_logical, dtype=torch.long),
    )
    bad_phy2log = torch.zeros(num_layers, num_physical + 1, dtype=torch.long)
    with pytest.raises(AssertionError):
        _commit_eplb_maps_for_layer(model_state, bad_phy2log, layer=0)


def test_commit_eplb_maps():
    """Test that all values are copied correctly into model_state."""
    num_layers, num_logical, num_physical, max_replicas = 2, 3, 4, 2

    model_state = _make_model_state(
        phy2log=torch.zeros(num_layers, num_physical, dtype=torch.long),
        log2phy=torch.full(
            (num_layers, num_logical, max_replicas), -1, dtype=torch.long
        ),
        logcnt=torch.zeros(num_layers, num_logical, dtype=torch.long),
    )

    new_phy2log = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]], dtype=torch.long)
    new_log2phy = torch.tensor(
        [[[0, 3], [1, -1], [2, -1]], [[2, -1], [0, 3], [1, -1]]], dtype=torch.long
    )
    new_logcnt = torch.tensor([[2, 1, 1], [1, 2, 1]], dtype=torch.long)

    _commit_eplb_maps(model_state, new_phy2log)

    assert torch.equal(model_state.physical_to_logical_map, new_phy2log)
    assert torch.equal(model_state.logical_to_physical_map, new_log2phy)
    assert torch.equal(model_state.logical_replica_count, new_logcnt)


def test_commit_eplb_maps_for_layer():
    """Test that only the target layer is updated"""
    num_layers, num_logical, max_replicas = 2, 3, 2

    original_phy2log = torch.tensor([[9, 9, 9, 9], [8, 8, 8, 8]], dtype=torch.long)
    model_state = _make_model_state(
        phy2log=original_phy2log.clone(),
        log2phy=torch.full(
            (num_layers, num_logical, max_replicas), -1, dtype=torch.long
        ),
        logcnt=torch.zeros(num_layers, num_logical, dtype=torch.long),
    )

    new_phy2log = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]], dtype=torch.long)
    new_log2phy = torch.tensor(
        [[[0, 3], [1, -1], [2, -1]], [[2, -1], [0, 3], [1, -1]]], dtype=torch.long
    )
    new_logcnt = torch.tensor([[2, 1, 1], [1, 2, 1]], dtype=torch.long)

    _commit_eplb_maps_for_layer(model_state, new_phy2log[0], layer=0)

    # Layer 0 updated
    assert torch.equal(model_state.physical_to_logical_map[0], new_phy2log[0])
    assert torch.equal(model_state.logical_to_physical_map[0], new_log2phy[0])
    assert torch.equal(model_state.logical_replica_count[0], new_logcnt[0])

    # Layer 1 untouched
    assert torch.equal(model_state.physical_to_logical_map[1], original_phy2log[1])


@pytest.mark.parametrize("per_layer", [False, True])
def test_rearrange_reuses_load_until_history_matches_mapping(monkeypatch, per_layer):
    """Use saved demand until old samples are replaced, separately for each layer."""
    module = "vllm.distributed.eplb.eplb_state"
    group = MagicMock()
    group.device_group.rank.return_value = 0
    group.device_group.size.return_value = 1
    monkeypatch.setattr(f"{module}.get_ep_group", lambda: group)
    monkeypatch.setattr(f"{module}.get_node_count", lambda: 1)
    ms = _make_model_state(
        torch.tensor([[0, 1], [0, 1]]),
        torch.full((2, 2, 1), -1),
        torch.ones((2, 2)),
    )
    ms.model.num_logical_experts = 2
    ms.model.num_physical_experts = 2
    ms.model.num_expert_groups = 1
    ms.model.num_moe_layers = 2
    ms.expert_load_pass = torch.zeros((2, 2), dtype=torch.int32)
    ms.expert_load_window[:] = torch.tensor([[10, 1], [20, 2]])
    state = object.__new__(EplbState)
    state.model_states = {"model": ms}
    state.expert_load_window_size = 3
    state.expert_load_window_step = 0
    state.expert_rearrangement_step = 0
    state.expert_rearrangement_step_interval = 100
    state.should_record_tensor = None
    state._should_record_current_step = lambda **kwargs: True
    state._allreduce_list = lambda values: values
    state.rearrange_event = MagicMock()

    def demand():
        state.is_async = True
        state.rearrange()
        return ms.eplb_stats.global_expert_load_window

    old_demand = demand().clone()
    mapping = torch.tensor([[1, 0], [1, 0]])
    if per_layer:
        _commit_eplb_maps_for_layer(ms, mapping[0], 0)
    else:
        _commit_eplb_maps(ms, mapping)
    torch.testing.assert_close(demand(), old_demand)
    state.is_async = False
    state.step(is_dummy=True)
    assert ms.expert_load_samples == 0

    for sample in range(1, 6):
        state.is_async = False
        ms.expert_load_pass.copy_(torch.tensor([[2, 20], [4, 40]]))
        state.step()
        if per_layer and sample == 2:
            _commit_eplb_maps_for_layer(ms, mapping[1], 1)
        if sample == 2:
            # Reinstalling an unchanged mapping must not postpone fresh statistics.
            if per_layer:
                _commit_eplb_maps_for_layer(ms, mapping[0], 0)
            else:
                _commit_eplb_maps(ms, mapping)
        # Do not refresh the cached second layer before its delayed installation.
        if per_layer and sample == 1:
            continue
        expected = old_demand.clone()
        if sample >= 3:
            expected[0] = torch.tensor([60, 6])
        if sample >= (5 if per_layer else 3):
            expected[1] = torch.tensor([120, 12])
        torch.testing.assert_close(demand(), expected)
