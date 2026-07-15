# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

from vllm.distributed.elastic_ep import timing


def test_elastic_ep_commit_timing_report(monkeypatch):
    monkeypatch.setenv("VLLM_ELASTIC_EP_COMMIT_TIMING", "1")
    timestamps = iter((0.0, 0.2, 0.3, 0.5, 0.7, 1.0))
    monkeypatch.setattr(timing.time, "perf_counter", lambda: next(timestamps))
    reports = []
    monkeypatch.setattr(
        timing.logger, "info", lambda _message, report: reports.append(report)
    )

    with timing.collect_commit_timing("worker", "scale_up", rank=3):
        with timing.record_commit_stage("capture"):
            pass
        with timing.record_commit_stage("restore"):
            pass
        timing.record_commit_metric("torch_compile_seconds", 1.25)
        timing.increment_commit_counter("graphs", 7)

    report = json.loads(reports.pop())
    assert report["scope"] == "worker"
    assert report["operation"] == "scale_up"
    assert report["rank"] == 3
    assert report["stages"][0]["name"] == "capture"
    assert report["unattributed_seconds"] == 0.7
    assert report["unattributed_gaps"] == [
        {
            "after": "restore",
            "before": "commit.end",
            "calls": 1,
            "total_seconds": 0.3,
            "max_seconds": 0.3,
        },
        {
            "after": "commit.start",
            "before": "capture",
            "calls": 1,
            "total_seconds": 0.2,
            "max_seconds": 0.2,
        },
        {
            "after": "capture",
            "before": "restore",
            "calls": 1,
            "total_seconds": 0.2,
            "max_seconds": 0.2,
        },
    ]
    assert report["overlapping_metrics"]["torch_compile_seconds"] == 1.25
    assert report["counters"]["graphs"] == 7
