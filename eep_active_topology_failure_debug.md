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

Diagnostic-only revisions add tracing without changing these implementations:

- vLLM `3301bea3046eb58bdff2d92c0314b6423d25f4bf`, branch
  `itay/eep_active_topology_debug`.
- NIXL `d3cfdb673293769481a568f9f50f3eacd42ded84`, branch
  `itay/eep-active-topology-debug`; its direct parent is the frozen NIXL
  revision above.

The first campaign reuses a previously built NIXL runtime at `06fade77`. Its
parent is `d3cfdb67`; the only additional change is an environment-gated GIL
experiment. `NIXL_EP_CONNECT_RELEASE_GIL` is explicitly unset, so that branch
executes the original connection path.

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

## First Diagnostic Reproduction

Lyris job `2763561`, measured attempt 1, reproduced the target failure on
`lyris[0279-0284]`. All 24 GB100 GPUs were in the same reported fabric clique.

Observed sequence:

1. Initial DP16 serving and accuracy were healthy.
2. Every retained worker entered `create_standby_groups()`.
3. No worker reached staged EP connection, EPLB creation, weight transfer, or
   target-group warmup.
4. Rank 0 entered a one-token dummy forward and remained inside it for
   approximately `60.5 s`.
5. Peers timed out after approximately `30 s` waiting for rank 0's NIXL EP
   dispatch. Ranks 2 and 10 explicitly reported dispatch-receive timeouts from
   source rank 0.
6. Fault detection produced mask `[1, 0, ..., 0]` on the still-active DP16
   manager (`manager=connected=active=16`).

This rules out all later preparation phases for this reproduction. It also
rules out simple Python-thread starvation: rank 0 launched the dummy forward,
then stalled inside model execution. The next experiment must split standby
group creation into world, DP, EP, and EPLB groups, and split each group into
device process group, CPU process group, TCP store, and device communicator
construction.

A separate screen (`2763563` warmup) completed scale-up but failed its
post-scale 64-question accuracy check (`0.969 -> 0.469`). It did not reproduce
the pre-commit timeout and is tracked separately until its overlap with the
post-commit asynchronous expert reshuffle is understood.

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

Current Ptyche requests:

| Job | Request | Status |
| ---: | --- | --- |
| `2638351` | Exact historical nodes | Pending resources |
| `2638352` | General screen A | Pending resources |
| `2638353` | General screen B | Pending resources |
| `2638354` | General screen C | Pending resources |
| `2638355` | General backfill screen | Pending resources |
| `2763553` | Exact historical Lyris nodes | Pending resources |
| `2763561` | Lyris screen B | Warmup passed 16->24 and post-scale accuracy; measured attempt running |
| `2763563` | Lyris screen A | Warmup scaled 16->24 but failed post-scale accuracy; measured attempt running |
| `2763557` | Lyris backfill screen | Pending resources |

Excluded setup attempts:

- Lyris `2763555`: the host-absolute source-bundle symlink was not visible
  inside the container mount.
- Lyris `2763556` and `2763559`: Ray-node checkout omitted the incremental
  diagnostic bundle. The API-server checkout succeeded, but workers exited
  before Ray startup.

No vLLM server or model process started in these attempts, so they provide no
product correctness evidence.

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
| 2026-08-23 | Froze incremental vLLM/NIXL diagnostic bundles | Complete |
| 2026-08-23 | Created isolated Ptyche harness | Complete |
| 2026-08-23 | Found matching prebuilt diagnostic runtime | Reusing `06fade77`; redundant builds canceled |
| 2026-08-23 | Submitted natural-reproduction screens | Jobs `2638351`-`2638355` pending |
| 2026-08-23 | Harness submission checks | Fixed stale partition and time-limit settings before any test ran |
| 2026-08-23 | Added parallel Lyris screens | Jobs `2763561` and `2763563` running; exact-node and backfill screens pending |
| 2026-08-23 | Lyris container checkout preflight | Fixed source symlink and Ray-node incremental fetch; three setup attempts excluded |
| 2026-08-23 | Add exact NIXL runtime operation history | Deferred until a phase hold reproduces |
| 2026-08-23 | Lyris screen B warmup (`2763561`) | Passed: initial accuracy `0.969`, post-up accuracy `0.953`, no active-topology timeout |
| 2026-08-23 | Lyris screen A warmup (`2763563`) | Separate correctness failure: initial accuracy `0.969`, post-up accuracy `0.469`; scale-up itself completed and no pre-commit timeout was observed |
