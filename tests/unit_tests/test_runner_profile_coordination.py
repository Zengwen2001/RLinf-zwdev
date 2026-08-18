# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for coordinated runner profiling windows."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from rlinf.runners.async_ppo_embodied_runner import AsyncPPOEmbodiedRunner
from rlinf.runners.embodied_runner import (
    EmbodiedRunner,
    close_profile_window_on_error,
)


@dataclass
class _Handle:
    name: str
    events: list[str]
    result: object = None

    def wait(self):
        self.events.append(f"wait:{self.name}")
        return self.result


class _Group:
    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        start_dispatch_error: BaseException | None = None,
        stop_dispatch_error: Exception | None = None,
        node_rank: int = 0,
        timestamp_ns: int = 0,
    ):
        self.name = name
        self.events = events
        self.start_dispatch_error = start_dispatch_error
        self.stop_dispatch_error = stop_dispatch_error
        self.node_rank = node_rank
        self.timestamp_ns = timestamp_ns

    def start_profile(self, step_idx: int):
        self.events.append(f"dispatch:start:{self.name}:{step_idx}")
        if self.start_dispatch_error is not None:
            raise self.start_dispatch_error
        return _Handle(
            f"start:{self.name}",
            self.events,
            result=[(self.node_rank, self.timestamp_ns)],
        )

    def stop_profile(self):
        self.events.append(f"dispatch:stop:{self.name}")
        if self.stop_dispatch_error is not None:
            raise self.stop_dispatch_error
        return _Handle(
            f"stop:{self.name}",
            self.events,
            result=[(self.node_rank, self.timestamp_ns)],
        )


class _Logger:
    def info(self, _message: str):
        pass

    def exception(self, _message: str):
        pass


class _StopAfterMetrics(Exception):
    pass


class _StopAtMetricsTimer:
    def __init__(self, events: list[str]):
        self.events = events

    def __call__(self, _name: str):
        return nullcontext()

    def consume_durations(self):
        self.events.append("metrics")
        raise _StopAfterMetrics


def _runner(
    events: list[str], *, actor_start_dispatch_error: BaseException | None = None
) -> EmbodiedRunner:
    runner = EmbodiedRunner.__new__(EmbodiedRunner)
    runner.actor = _Group(
        "actor",
        events,
        start_dispatch_error=actor_start_dispatch_error,
        timestamp_ns=1_000_000_000,
    )
    runner.rollout = _Group("rollout", events, timestamp_ns=1_001_000_000)
    runner.env = _Group("env", events, timestamp_ns=1_002_000_000)
    runner.reward = _Group("reward", events)
    runner.cfg = SimpleNamespace(cluster={"num_nodes": 1})
    runner.logger = _Logger()
    runner._active_profile_step = None
    runner._profile_start_skew_ms = None
    runner._profile_stop_skew_ms = None
    return runner


def test_open_dispatches_all_compute_groups_before_waiting():
    events: list[str] = []
    runner = _runner(events)

    runner._open_profiling_window(7)

    assert events == [
        "dispatch:start:actor:7",
        "dispatch:start:rollout:7",
        "dispatch:start:env:7",
        "wait:start:actor",
        "wait:start:rollout",
        "wait:start:env",
    ]
    assert runner._active_profile_step == 7
    assert runner._profile_start_skew_ms == pytest.approx(2.0)


def test_close_dispatches_all_compute_groups_before_waiting():
    events: list[str] = []
    runner = _runner(events)
    runner._active_profile_step = 7

    runner._close_profiling_window(7)

    assert events == [
        "dispatch:stop:actor",
        "dispatch:stop:rollout",
        "dispatch:stop:env",
        "wait:stop:actor",
        "wait:stop:rollout",
        "wait:stop:env",
    ]
    assert runner._active_profile_step is None
    assert runner._profile_stop_skew_ms == pytest.approx(2.0)


def test_runner_error_closes_all_groups():
    events: list[str] = []
    runner = _runner(events)

    @close_profile_window_on_error
    def run_profiled_step(target):
        target._open_profiling_window(3)
        events.append("step")
        raise RuntimeError("step failed")

    with pytest.raises(RuntimeError, match="step failed"):
        run_profiled_step(runner)

    assert events == [
        "dispatch:start:actor:3",
        "dispatch:start:rollout:3",
        "dispatch:start:env:3",
        "wait:start:actor",
        "wait:start:rollout",
        "wait:start:env",
        "step",
        "dispatch:stop:actor",
        "dispatch:stop:rollout",
        "dispatch:stop:env",
        "wait:stop:actor",
        "wait:stop:rollout",
        "wait:stop:env",
    ]


def test_failed_start_dispatch_cleans_up_all_compute_groups():
    events: list[str] = []
    runner = _runner(
        events, actor_start_dispatch_error=RuntimeError("start dispatch failed")
    )

    with pytest.raises(RuntimeError, match="start dispatch failed"):
        runner._open_profiling_window(9)

    assert events == [
        "dispatch:start:actor:9",
        "dispatch:start:rollout:9",
        "dispatch:start:env:9",
        "wait:start:rollout",
        "wait:start:env",
        "dispatch:stop:actor",
        "dispatch:stop:rollout",
        "dispatch:stop:env",
        "wait:stop:actor",
        "wait:stop:rollout",
        "wait:stop:env",
    ]
    assert runner._active_profile_step is None


def test_base_exception_during_start_dispatch_cleans_up_all_compute_groups():
    events: list[str] = []
    runner = _runner(events, actor_start_dispatch_error=KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        runner._open_profiling_window(9)

    assert events == [
        "dispatch:start:actor:9",
        "dispatch:stop:actor",
        "dispatch:stop:rollout",
        "dispatch:stop:env",
        "wait:stop:actor",
        "wait:stop:rollout",
        "wait:stop:env",
    ]


def test_failed_stop_keeps_window_active_for_error_recovery():
    events: list[str] = []
    runner = _runner(events)
    runner._active_profile_step = 4
    runner.actor.stop_dispatch_error = RuntimeError("stop dispatch failed")

    with pytest.raises(RuntimeError, match="stop dispatch failed"):
        runner._close_profiling_window(4)

    assert runner._active_profile_step == 4


def test_multi_node_timestamps_do_not_report_a_cross_node_skew():
    events: list[str] = []
    runner = _runner(events)
    runner.cfg.cluster["num_nodes"] = 2
    runner.env.node_rank = 1

    runner._open_profiling_window(5)

    assert runner._profile_start_skew_ms is None


def test_runner_error_without_active_window_does_not_stop_groups():
    events: list[str] = []
    runner = _runner(events)

    @close_profile_window_on_error
    def run_unprofiled_step(_target):
        events.append("step")
        raise RuntimeError("step failed")

    with pytest.raises(RuntimeError, match="step failed"):
        run_unprofiled_step(runner)

    assert events == ["step"]


@pytest.mark.parametrize(
    "run_method",
    [
        EmbodiedRunner.run,
        EmbodiedRunner.run_pipeline,
        AsyncPPOEmbodiedRunner.run,
    ],
)
def test_all_profiled_runner_entrypoints_install_error_cleanup(run_method):
    assert hasattr(run_method, "__wrapped__")


def test_async_ppo_window_surrounds_measured_step_before_metrics():
    events: list[str] = []
    runner = AsyncPPOEmbodiedRunner.__new__(AsyncPPOEmbodiedRunner)
    base = _runner(events)
    runner.__dict__.update(base.__dict__)
    runner.global_step = 0
    runner.max_steps = 1
    runner.recompute_logprobs = False
    runner.timer = _StopAtMetricsTimer(events)
    runner.env_channel = object()
    runner.rollout_channel = object()
    runner.reward_channel = None
    runner.actor_channel = object()
    runner.env_metric_channel = object()
    runner.rollout_metric_channel = object()
    runner._should_profile_step = lambda _step: True
    runner.update_rollout_weights = lambda: events.append("weight-sync")

    runner.actor.set_global_step = lambda _step: _Handle("set-actor-step", events)
    runner.actor.construct_rollout_batch = lambda: _Handle(
        "construct-rollout-batch", events, result={}
    )
    runner.actor.compute_advantages_and_returns = lambda: _Handle(
        "advantages", events, result=[]
    )
    runner.actor.run_training = lambda: _Handle("train", events, result=[{}])
    runner.actor.recv_rollout_trajectories = lambda **_kwargs: _Handle(
        "recv-trajectories", events
    )
    runner.rollout.set_global_step = lambda _step: _Handle("set-rollout-step", events)
    runner.rollout.generate = lambda **_kwargs: _Handle("generate", events)
    runner.env.interact = lambda **_kwargs: _Handle("interact", events)

    with pytest.raises(_StopAfterMetrics):
        runner.run()

    assert events.index("wait:start:env") < events.index("wait:construct-rollout-batch")
    assert events.index("wait:construct-rollout-batch") < events.index("wait:train")
    assert events.index("wait:train") < events.index("dispatch:stop:actor")
    assert events.index("wait:stop:env") < events.index("metrics")
