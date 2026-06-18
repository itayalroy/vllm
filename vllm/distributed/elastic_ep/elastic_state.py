# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import enum
import time
import weakref
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import torch.distributed

from vllm.config import ParallelConfig
from vllm.distributed import (
    stateless_destroy_torch_distributed_process_group,
)
from vllm.distributed.utils import get_cached_tcp_store_client
from vllm.logger import init_logger
from vllm.v1.engine import (
    EEPNotificationType,
    ReconfigureDistributedRequest,
    ReconfigureRankType,
)
from vllm.v1.engine.core import DPEngineCoreProc

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.executor.abstract import Executor

logger = init_logger(__name__)

WorkerType = Literal["existing", "new", "removing"]
_PREPARE_ASYNC_POLL_INTERVAL_S = 0.05


class ScaleUpExistingEngineState(enum.IntEnum):
    WAIT_NEW_CORE_ENGINES_INIT = 0
    CREATE_STANDBY_GROUPS = 1
    TRANSFER_EXPERT_MAPPING = 2
    WAIT_NEW_CORE_ENGINES_WEIGHTS_INIT = 3
    TRANSFER_WEIGHTS = 4
    SYNC_KV_CACHE_MEMORY_SIZE = 5
    SWITCH_AND_PREPARE = 6
    COMPLETE = 7


class ScaleUpNewEngineState(enum.IntEnum):
    PRE_KV_INIT = 0
    PREPARE = 1
    COMPLETE = 2


class ScaleDownRemainingEngineState(enum.IntEnum):
    PREPARE = 0
    COMPLETE = 1


class ScaleDownRemovingEngineState(enum.IntEnum):
    PREPARE = 0
    COMPLETE = 1


EngineState: TypeAlias = (
    ScaleUpExistingEngineState
    | ScaleUpNewEngineState
    | ScaleDownRemainingEngineState
    | ScaleDownRemovingEngineState
)


class ElasticEPScalingState:
    def __init__(
        self,
        model_executor: "Executor",
        engine_core: "DPEngineCoreProc",
        vllm_config: "VllmConfig",
        new_parallel_config: ParallelConfig,
        worker_type: WorkerType,
        scale_type: Literal["scale_up", "scale_down"],
        reconfig_request: ReconfigureDistributedRequest | None = None,
        prepare_mode: bool = False,
    ):
        self.model_executor_ref = weakref.ref(model_executor)
        self.engine_core_ref = weakref.ref(engine_core)
        self.vllm_config = vllm_config
        self.old_dp_group = self.engine_core.dp_group if worker_type != "new" else None
        self.old_dp_store = self.engine_core.dp_store if worker_type != "new" else None
        self.new_parallel_config: ParallelConfig = new_parallel_config
        self.new_dp_group = self.engine_core.dp_group if worker_type == "new" else None
        self.new_dp_store = self.engine_core.dp_store if worker_type == "new" else None
        self.worker_type = worker_type
        self.scale_type = scale_type
        self.reconfig_request = reconfig_request
        self.prepare_mode = prepare_mode
        self._prepare_async_method: str | None = None
        self._prepare_async_last_poll = 0.0
        self._prepare_async_done_keys: list[str] | None = None
        self._kv_cache_memory_sync: tuple[object, Any] | None = None
        self._standby_groups_created = False
        self.state: EngineState
        if scale_type == "scale_up":
            self.state = (
                ScaleUpNewEngineState.PRE_KV_INIT
                if worker_type == "new"
                else ScaleUpExistingEngineState.WAIT_NEW_CORE_ENGINES_INIT
            )
        else:
            self.state = (
                ScaleDownRemovingEngineState.PREPARE
                if worker_type == "removing"
                else ScaleDownRemainingEngineState.PREPARE
            )

    @property
    def model_executor(self) -> "Executor":
        model_executor = self.model_executor_ref()
        if model_executor is None:
            raise RuntimeError("Model executor has been garbage collected")
        return model_executor

    @property
    def engine_core(self) -> "DPEngineCoreProc":
        engine_core = self.engine_core_ref()
        if engine_core is None:
            raise RuntimeError("Engine core has been garbage collected")
        return engine_core

    def _collective_rpc(self, *args, **kwargs):
        return self.model_executor.collective_rpc(*args, **kwargs)

    def _elastic_ep_execute(self, execute_method: str, *args) -> bool:
        if not self.prepare_mode:
            self._collective_rpc("elastic_ep_execute", args=(execute_method, *args))
            return True

        if self._prepare_async_method is None:
            done_keys = self._collective_rpc(
                "elastic_ep_execute",
                args=("start_async", execute_method, *args),
            )
            self._prepare_async_method = execute_method
            self._prepare_async_done_keys = list(done_keys)
            self._prepare_async_last_poll = 0.0
            return False

        if self._prepare_async_method != execute_method:
            raise RuntimeError(
                "Elastic EP prepare async method is already active: "
                f"{self._prepare_async_method}"
            )

        now = time.monotonic()
        if now - self._prepare_async_last_poll < _PREPARE_ASYNC_POLL_INTERVAL_S:
            return False
        self._prepare_async_last_poll = now

        assert self.reconfig_request is not None
        assert self._prepare_async_done_keys is not None
        coord_store = get_cached_tcp_store_client(
            self.reconfig_request.new_data_parallel_master_ip,
            self.reconfig_request.coord_store_port,
        )
        if not coord_store.check(self._prepare_async_done_keys):
            return False

        self._collective_rpc("elastic_ep_execute", args=("clear_async", execute_method))
        self._prepare_async_method = None
        self._prepare_async_done_keys = None
        return True

    def progress(self) -> bool:
        if self.scale_type == "scale_up":
            return (
                self._progress_new_engine()
                if self.worker_type == "new"
                else self._progress_existing_engine()
            )
        return (
            self._progress_removing_engine()
            if self.worker_type == "removing"
            else self._progress_remaining_engine()
        )

    def run_pre_kv_init_states(self) -> None:
        assert self.scale_type == "scale_up" and self.worker_type == "new"
        assert self.state == ScaleUpNewEngineState.PRE_KV_INIT
        assert self.progress()
        assert self.state == ScaleUpNewEngineState.PREPARE

    def _old_dp_wave_drained(self) -> bool:
        return (
            not self.engine_core.engines_running
            and not self.engine_core.scheduler.has_requests()
        )

    def _progress_existing_engine(self) -> bool:
        state = self.state
        assert self.old_dp_group is not None and self.old_dp_store is not None

        if state == ScaleUpExistingEngineState.WAIT_NEW_CORE_ENGINES_INIT:
            return False

        elif state == ScaleUpExistingEngineState.CREATE_STANDBY_GROUPS:
            if not self._create_standby_groups():
                return False
            self.state = ScaleUpExistingEngineState.TRANSFER_EXPERT_MAPPING
            return True

        elif state == ScaleUpExistingEngineState.TRANSFER_EXPERT_MAPPING:
            if not self._transfer_expert_mapping():
                return False
            self.state = ScaleUpExistingEngineState.WAIT_NEW_CORE_ENGINES_WEIGHTS_INIT
            return True

        elif state == ScaleUpExistingEngineState.WAIT_NEW_CORE_ENGINES_WEIGHTS_INIT:
            return False

        elif state == ScaleUpExistingEngineState.TRANSFER_WEIGHTS:
            if not self._transfer_weights():
                return False
            self.state = ScaleUpExistingEngineState.SYNC_KV_CACHE_MEMORY_SIZE
            return True

        elif state == ScaleUpExistingEngineState.SYNC_KV_CACHE_MEMORY_SIZE:
            if not self._sync_kv_cache_memory_size():
                return False
            self.state = ScaleUpExistingEngineState.SWITCH_AND_PREPARE
            return True

        elif state == ScaleUpExistingEngineState.SWITCH_AND_PREPARE:
            if self.prepare_mode:
                return False
            if not self._old_dp_wave_drained():
                return False
            self._switch_and_prepare()
            self._eplb_reshuffle()
            self.state = ScaleUpExistingEngineState.COMPLETE
            self._update_parallel_config()
            self._send_reconfigure_finished()
            return True

        else:
            assert self.state == ScaleUpExistingEngineState.COMPLETE
            return True

    def _progress_new_engine(self) -> bool:
        state = self.state
        assert self.new_dp_group is not None and self.new_dp_store is not None

        if state == ScaleUpNewEngineState.PRE_KV_INIT:
            self.engine_core._eep_send_engine_core_notification(
                EEPNotificationType.NEW_CORE_ENGINES_WEIGHTS_INIT_READY
            )
            self._collective_rpc("elastic_ep_execute", args=("receive_weights",))
            self.engine_core.available_gpu_memory_for_kv_cache = (
                ParallelConfig.sync_kv_cache_memory_size(self.new_dp_group, -1)
            )
            self._collective_rpc("elastic_ep_execute", args=("prepare_new_worker",))
            self.state = ScaleUpNewEngineState.PREPARE
            return True

        elif state == ScaleUpNewEngineState.PREPARE:
            tensor = torch.tensor([0, 0, 0], dtype=torch.int32, device="cpu")
            torch.distributed.all_reduce(
                tensor,
                op=torch.distributed.ReduceOp.MAX,
                group=self.new_dp_group,
            )
            data = tensor.tolist()
            self.engine_core.engines_running = bool(data[0])
            self.engine_core.current_wave = int(data[1])
            self.engine_core.step_counter = int(data[2])
            self._eplb_reshuffle()
            self.state = ScaleUpNewEngineState.COMPLETE
            return True

        else:
            assert self.state == ScaleUpNewEngineState.COMPLETE
            return True

    def _progress_remaining_engine(self) -> bool:
        state = self.state
        assert self.old_dp_group is not None and self.old_dp_store is not None

        if state == ScaleDownRemainingEngineState.PREPARE:
            if self.prepare_mode:
                if self._standby_groups_created:
                    return False
                return self._create_standby_groups()
            if not self._old_dp_wave_drained():
                return False
            self._eplb_reshuffle_before_scale_down()
            # NOTE(yongji): currently, after EPLB reshuffle
            # that redistributes experts to remaining workers, workers
            # to be removed will immediately initiate shutdown;
            # existing workers can no longer execute forward steps using
            # the old setup. In the future, we may keep
            # the removing workers alive a bit longer,
            # e.g., to drain in-batch requests.
            if not self._standby_groups_created:
                self._create_standby_groups()
            self._switch_and_prepare()
            self._collective_rpc("elastic_ep_execute", args=("warm_and_capture",))
            self._update_parallel_config()
            self.state = ScaleDownRemainingEngineState.COMPLETE
            self._send_reconfigure_finished()
            return True

        else:
            assert self.state == ScaleDownRemainingEngineState.COMPLETE
            return True

    def _progress_removing_engine(self) -> bool:
        state = self.state
        assert self.old_dp_group is not None and self.old_dp_store is not None

        if state == ScaleDownRemovingEngineState.PREPARE:
            if not self._old_dp_wave_drained():
                return False
            assert self.old_dp_group.rank() > 0
            self._eplb_reshuffle_before_scale_down()
            self._switch_and_remove()
            self.state = ScaleDownRemovingEngineState.COMPLETE
            self.engine_core._eep_send_engine_core_notification(
                EEPNotificationType.SHUTDOWN_COMPLETE
            )
            return True

        else:
            assert self.state == ScaleDownRemovingEngineState.COMPLETE
            return True

    def handle_notification(self, notification_type: EEPNotificationType):
        assert self.worker_type != "new"
        if (
            notification_type == EEPNotificationType.NEW_CORE_ENGINES_INIT_READY
            and self.state == ScaleUpExistingEngineState.WAIT_NEW_CORE_ENGINES_INIT
        ):
            self.state = ScaleUpExistingEngineState.CREATE_STANDBY_GROUPS
        elif (
            notification_type == EEPNotificationType.NEW_CORE_ENGINES_WEIGHTS_INIT_READY
            and self.state
            == ScaleUpExistingEngineState.WAIT_NEW_CORE_ENGINES_WEIGHTS_INIT
        ):
            self.state = ScaleUpExistingEngineState.TRANSFER_WEIGHTS

    def is_ready_for_switch(self) -> bool:
        if (
            self.scale_type == "scale_up"
            and self.worker_type == "existing"
            and self.state == ScaleUpExistingEngineState.SWITCH_AND_PREPARE
        ):
            return True
        return (
            self.scale_type == "scale_down"
            and self.worker_type == "existing"
            and self.state == ScaleDownRemainingEngineState.PREPARE
            and self._standby_groups_created
        )

    def is_complete(self) -> bool:
        if self.scale_type == "scale_up":
            return (
                self.state == ScaleUpNewEngineState.COMPLETE
                if self.worker_type == "new"
                else self.state == ScaleUpExistingEngineState.COMPLETE
            )
        return (
            self.state == ScaleDownRemovingEngineState.COMPLETE
            if self.worker_type == "removing"
            else self.state == ScaleDownRemainingEngineState.COMPLETE
        )

    def _create_standby_groups(self) -> bool:
        assert self.old_dp_group is not None
        if self._prepare_async_method != "create_standby_groups":
            self.new_dp_group, self.new_dp_store = (
                self.new_parallel_config.stateless_init_dp_group(return_store=True)
            )
        if not self._elastic_ep_execute("create_standby_groups", self.reconfig_request):
            return False
        self._standby_groups_created = True
        if self.old_dp_group.rank() == 0:
            logger.info("[Elastic EP] Created standby communication groups")
        return True

    def _transfer_weights(self) -> bool:
        assert self.reconfig_request is not None and self.old_dp_group is not None
        old_dp_size = self.old_dp_group.size()
        new_dp_size = self.reconfig_request.new_data_parallel_size

        if not self._elastic_ep_execute("transfer_weights", old_dp_size, new_dp_size):
            return False
        if self.old_dp_group.rank() == 0:
            logger.info("[Elastic EP] Transferred weights to new workers")
        return True

    def _transfer_expert_mapping(self) -> bool:
        assert self.old_dp_group is not None
        if not self._elastic_ep_execute("broadcast_expert_mapping"):
            return False
        if self.old_dp_group.rank() == 0:
            logger.info("[Elastic EP] Broadcasted expert mapping to new workers")
        return True

    def _sync_kv_cache_memory_size(self) -> bool:
        assert self.engine_core.available_gpu_memory_for_kv_cache > 0
        assert self.new_dp_group is not None and self.old_dp_group is not None

        if self._kv_cache_memory_sync is None:
            tensor = torch.tensor(
                [self.engine_core.available_gpu_memory_for_kv_cache],
                dtype=torch.int64,
                device="cpu",
            )
            work = torch.distributed.all_reduce(
                tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=self.new_dp_group,
                async_op=True,
            )
            self._kv_cache_memory_sync = (tensor, work)
            return False

        _, work = self._kv_cache_memory_sync
        if not work.is_completed():
            return False
        work.wait()
        self._kv_cache_memory_sync = None
        if self.old_dp_group.rank() == 0:
            logger.info("[Elastic EP] Synced KV cache memory size to new workers")
        return True

    def _switch_and_prepare(self):
        self._collective_rpc("elastic_ep_execute", args=("switch_and_prepare",))
        old_dp_group = self.old_dp_group
        stateless_destroy_torch_distributed_process_group(old_dp_group)
        assert self.new_dp_group is not None
        new_dp_group = self.new_dp_group
        self.engine_core.dp_group = new_dp_group
        self.engine_core.dp_rank = new_dp_group.rank()
        self.engine_core.dp_store = self.new_dp_store
        engines_running = int(self.engine_core.engines_running)
        current_wave = self.engine_core.current_wave
        step_counter = self.engine_core.step_counter
        tensor = torch.tensor(
            [engines_running, current_wave, step_counter],
            dtype=torch.int32,
            device="cpu",
        )
        torch.distributed.all_reduce(
            tensor, op=torch.distributed.ReduceOp.MAX, group=new_dp_group
        )
        data = tensor.tolist()
        self.engine_core.engines_running = bool(data[0])
        self.engine_core.current_wave = int(data[1])
        self.engine_core.step_counter = int(data[2])
        if new_dp_group.rank() == 0:
            logger.info("[Elastic EP] Switched to new setup")

    def _send_reconfigure_finished(self):
        assert self.new_dp_group is not None
        if self.new_dp_group.rank() == 0:
            self.engine_core._eep_send_engine_core_notification(
                EEPNotificationType.RECONFIGURE_FINISHED
            )

    def _eplb_reshuffle(self):
        self._collective_rpc("elastic_ep_execute", args=("perform_eplb_reshuffle",))
        self._collective_rpc("elastic_ep_execute", args=("warm_and_capture",))
        assert self.new_dp_group is not None
        if self.new_dp_group.rank() == 0:
            logger.info("[Elastic EP] EPLB reshuffle completed")

    def _eplb_reshuffle_before_scale_down(self):
        assert self.reconfig_request is not None and self.old_dp_group is not None
        self._collective_rpc(
            "elastic_ep_execute",
            args=(
                "perform_scale_down_eplb_reshuffle",
                self.reconfig_request.new_data_parallel_size,
            ),
        )
        if self.old_dp_group.rank() == 0:
            logger.info("[Elastic EP] EPLB reshuffle completed")

    def _switch_and_remove(self):
        self._collective_rpc("elastic_ep_execute", args=("switch_and_remove",))

    def _update_parallel_config(self):
        assert self.reconfig_request is not None
        reconfig_request = self.reconfig_request
        parallel_config = self.vllm_config.parallel_config
        parallel_config.data_parallel_size = reconfig_request.new_data_parallel_size
        if (
            reconfig_request.new_data_parallel_rank
            != ReconfigureRankType.KEEP_CURRENT_RANK
        ):
            parallel_config.data_parallel_rank = reconfig_request.new_data_parallel_rank
        if (
            reconfig_request.new_data_parallel_rank_local
            != ReconfigureRankType.KEEP_CURRENT_RANK
        ):
            parallel_config.data_parallel_rank_local = (
                reconfig_request.new_data_parallel_rank_local
            )
        parallel_config.data_parallel_master_ip = (
            reconfig_request.new_data_parallel_master_ip
        )
        parallel_config.data_parallel_master_port = (
            reconfig_request.new_data_parallel_master_port
        )
        parallel_config._data_parallel_master_port_list = (
            reconfig_request.new_data_parallel_master_port_list
        )
        parallel_config._coord_store_port = reconfig_request.coord_store_port
