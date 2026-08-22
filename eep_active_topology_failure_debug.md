# Elastic EP Active-Topology Failure Debug

Last updated: 2026-08-23

## Goal

Find the first point at which retained Elastic EP ranks stop agreeing during
scale-up preparation. The observed failure is a NIXL EP dispatch/combine timeout
on the still-active DP16 topology, before the DP24 configuration is committed.

The investigation must answer one concrete question:

> At the first failed old-topology operation, did a rank fail to launch the
> operation, launch a different operation, use different topology state, or
> launch the same operation and fail in the transport/runtime layer?

## Frozen Reproducer

Do not change these inputs while isolating the failure:

| Component | Revision/configuration |
| --- | --- |
| vLLM | `a46cfedd073333faf35198572bce6fbfe0154305` |
| NIXL | `8e9d233b35383ad89e6929760985f52b24d27075` |
| UCX | `4916a6f24c71cf3832465b5c1915adbf137ede7c` |
| Model | `deepseek-ai/DeepSeek-V3` |
| Reconfiguration | DP `16 -> 24`, TP 1 |
| Serving | Heavy traffic, CUDA-graph reuse, NIXL EP |
| EPLB | Async, step interval 300 |
| CUDA IPC cache | Off for initial reproduction |
| Compilation | Warm Mega AOT cache; FlashInfer autotuning disabled |
| Capacity | Default model length and max sequences |
| Memory | `--gpu-memory-utilization 0.80` |

Every attempt starts a fresh server. Cache-population and infrastructure
failures do not count as reproductions.

## Known Evidence

A controlled paired campaign did not reproduce the failure:

| Flow | IPC cache on | IPC cache off | Accuracy checks | EP errors |
| --- | ---: | ---: | ---: | ---: |
| Upstream graph recapture | 15/15 | 15/15 | 90/90 | 0 |
| CUDA-graph reuse | 15/15 | 15/15 | 90/90 | 0 |

Historical failures were intermittent and allocation-sensitive:

| Cluster/run | Allocation | Result |
| --- | --- | --- |
| Ptyche `2631589_15` | `ptyche[0056-0057,0062,0065,0067-0068]` | 8/10 passed; attempts 1 and 6 failed |
| Lyris `2753950` | `lyris[0127-0129,0131-0133]` | Warmup and measured attempt 5 failed |

The timeout partitions changed between failures and split GPUs within a node:

| Failure | Ranks on smaller timeout side |
| --- | --- |
| Ptyche attempt 1 | `0,7,12,13,14,15` |
| Ptyche attempt 6 | `12,15` |
| Lyris warmup | `4,5,7,14,15` |
| Lyris attempt 5 | `0,2,5` |

This does not fit a deterministic bad node, local GPU index, NVL block, or IB
leaf. Cache-off alone is also not a demonstrated root cause.

## Preparation Timeline

On retained workers, scale-up preparation currently runs in this order:

1. Wait for any retired process-group cleanup.
2. Create standby process groups.
3. Stage the target NIXL EP size with `connect_ranks(..., activate=False)`.
4. Prepare the target EPLB communicator.
5. Transfer weights to new workers.
6. Warm the standby DP and EP groups.
7. Return from worker preparation.
8. Synchronize KV-cache memory size over the target group.
9. Mark the engine ready and park at commit.

Retained workers continue serving with the old DP16 topology throughout these
steps. No target groups, expert mapping, MoE runtime, or CUDA graphs are
installed until commit.

## Diagnostic Patch

The diagnostic branch will add temporary, opt-in instrumentation only.

### Phase Timeline

Record monotonic and wall-clock timestamps per rank for:

- standby-group creation begin/end
- EP `connect_ranks` begin/end
- EPLB communicator creation begin/end
- weight transfer begin/end
- standby DP warmup begin/end
- standby EP warmup begin/end
- worker preparation return
- KV-cache synchronization begin/end
- ready-for-commit publication

### Retained-Forward History

Keep a bounded history of the last 64 real and dummy forward events per rank:

- forward type and sequence number
- launch and completion timestamps
- current preparation phase

Dump the history with the device mask and active/connected EP sizes when output
waiting raises or EP fault detection fires. Disable Ray log deduplication so
rank-specific evidence is retained.

Exact dispatch/combine sequence and memory-view generation must be recorded by
the NIXL runtime. These operations are replayed from a CUDA graph, so wrapping
the Python `Buffer.dispatch()` method would trace capture rather than the
failing runtime operation.

### Controlled Phase Gates

A diagnostic controller can hold retained workers after a selected phase. Each
worker publishes arrival to the coordination store and waits for an explicit
release key. The hold timer starts only after every retained rank has arrived.

Supported gates:

- after staged EP connection
- after EPLB communicator creation
- after weight transfer
- after standby DP/EP warmup

The normal path remains unchanged when no gate is configured.

### Diagnostic Environment

```bash
RAY_DEDUP_LOGS=0 \
VLLM_EEP_DEBUG_TRACE=1 \
NIXL_EP_CONNECT_TIMELINE=1 \
<test command>
```

To run a coordinated 120-second hold after a phase:

```bash
VLLM_EEP_DEBUG_HOLD_PHASE=ep_connect \
VLLM_EEP_DEBUG_HOLD_SECONDS=120 \
<test command>
```

Valid hold names are `ep_connect`, `eplb_communicator`, `weight_transfer`, and
`target_group_warmup`. Rank 0 starts the timer only after all retained worker
processes have published arrival.

## Experiment Sequence

### 1. Recover a Susceptible Allocation

Try, in order:

1. The exact historical Ptyche node list if available.
2. Another allocation in `nvlblk04`.
3. Up to three concurrent allocations, screened with five natural attempts
   each. Keep the first allocation that reproduces and leave its job alive.

Run at most 20 natural attempts before reassessing the reproducer.

A valid reproduction requires:

- healthy initial serving and accuracy
- a failure before commit
- old-topology DP16 NIXL dispatch/combine receive timeouts
- no infrastructure, cold-AOT, or new-rank startup failure first

### 2. Isolate the First Harmful Phase

Use fresh processes for each cell:

| Cell | Hold point | Hold duration | Attempts | Result |
| --- | --- | ---: | ---: | --- |
| Control | No scale request | 120 s | 3 | Pending |
| A | After staged EP connection | 120 s | 3 | Pending |
| B | After EPLB communicator creation | 120 s | 3 | Pending |
| C | After weight transfer | 120 s | 3 | Pending |
| D | After standby DP/EP warmup | 120 s | 3 | Pending |

Stop a cell after the first valid reproduction. The earliest failing hold point
identifies the phase that made the active topology unsafe.

### 3. Classify the Failure

At the first mismatched NIXL sequence:

| Observation | Classification |
| --- | --- |
| A rank never launched the operation | Worker starvation or phase skew |
| Ranks launched different operation sequences | Serving-loop divergence |
| Same operation, different mask/view generation | Staging-state divergence |
| Same operation and state on every rank | Transport or memory-lifetime failure |

### 4. Verify the Suspect Operation

Once a phase is identified, split it into its internal operations:

- EP connection: remote-agent exchange, memory metadata exchange, staged
  memory-view creation, and connected-rank bookkeeping.
- EPLB creation: agent creation, buffer registration, remote-agent exchange,
  and remote-memory metadata exchange.
- Group warmup: DP collective and EP collective separately.

### 5. Same-Allocation A/B

Alternate cells on one retained allocation, up to ten attempts each and stop
after two valid failures:

| NIXL | IPC cache | Attempts | Valid failures | Result |
| --- | --- | ---: | ---: | --- |
| Historical frozen revision | On | 0/10 | 0 | Pending |
| Historical frozen revision | Off | 0/10 | 0 | Pending |
| PR 2138 (`09cf765b...`) | On | 0/10 | 0 | Pending |
| PR 2138 (`09cf765b...`) | Off | 0/10 | 0 | Pending |

## Required Artifacts

For every attempt, preserve:

- Slurm job, nodes, GPU/rank placement, NVL block, and IB leaf
- exact vLLM/NIXL/UCX revisions and complete environment
- complete serve command and effective cache setting
- per-rank phase timeline and NIXL operation history
- server/client logs, result JSON, accuracy, and exit status
- UCX transport selection
- Xid, NVLink, NIC, and fabric-health snapshots around the failure

## Progress Log

| Date | Action | Outcome |
| --- | --- | --- |
| 2026-08-23 | Froze reproducer and wrote isolation plan | Complete |
| 2026-08-23 | Created `eep_active_topology_debug` | Complete |
| 2026-08-23 | Added opt-in phase tracing and coordinated holds | Complete |
| 2026-08-23 | Added bounded retained-forward fault history | Complete |
| 2026-08-23 | Ruff, byte compilation, mypy, and diff checks | Passed |
| 2026-08-23 | Add exact NIXL runtime operation history | In progress |
