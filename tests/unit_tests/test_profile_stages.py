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

"""Tests for stable profiling stage annotations."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from rlinf.profiling import ProfileStage, profile_range
from rlinf.scheduler.hardware import AcceleratorType, AcceleratorUtil
from rlinf.scheduler.hardware.accelerators import nvidia_gpu
from rlinf.scheduler.worker.worker import Worker


def test_profile_range_is_transparent_without_current_worker(monkeypatch):
    result = []
    monkeypatch.setattr(Worker, "current_worker", None)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("accelerator backend must not run outside a worker")

    monkeypatch.setattr(AcceleratorUtil, "profiling_range", fail_if_called)

    with profile_range(ProfileStage.ACTOR_FORWARD):
        result.append("executed")

    assert result == ["executed"]


def test_profile_range_delegates_worker_stage_to_accelerator(monkeypatch):
    events = []
    worker = SimpleNamespace(accelerator_type=AcceleratorType.NV_GPU)
    monkeypatch.setattr(Worker, "current_worker", worker)

    @contextmanager
    def record_range(accelerator_type, label):
        events.append(("enter", accelerator_type, label))
        try:
            yield
        finally:
            events.append(("exit", accelerator_type, label))

    monkeypatch.setattr(AcceleratorUtil, "profiling_range", record_range)

    with profile_range(ProfileStage.ACTOR_FORWARD):
        events.append(("body",))

    assert events == [
        ("enter", AcceleratorType.NV_GPU, "ACTOR/FORWARD"),
        ("body",),
        ("exit", AcceleratorType.NV_GPU, "ACTOR/FORWARD"),
    ]


def test_profile_range_worker_is_transparent_when_profiling_inactive(monkeypatch):
    events = []
    monkeypatch.setattr(
        Worker,
        "current_worker",
        SimpleNamespace(accelerator_type=AcceleratorType.NV_GPU),
    )
    monkeypatch.setattr(nvidia_gpu, "_nv_profiling_active", False)
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", events.append)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: events.append("pop"))

    with profile_range(ProfileStage.ACTOR_FORWARD):
        pass

    assert events == []


def test_profile_range_closes_backend_range_when_body_raises(monkeypatch):
    events = []
    worker = SimpleNamespace(accelerator_type=AcceleratorType.NV_GPU)
    monkeypatch.setattr(Worker, "current_worker", worker)

    @contextmanager
    def record_range(accelerator_type, label):
        events.append(("enter", accelerator_type, label))
        try:
            yield
        finally:
            events.append(("exit", accelerator_type, label))

    monkeypatch.setattr(AcceleratorUtil, "profiling_range", record_range)

    with pytest.raises(RuntimeError, match="stage failed"):
        with profile_range(ProfileStage.ACTOR_FORWARD):
            raise RuntimeError("stage failed")

    assert events == [
        ("enter", AcceleratorType.NV_GPU, "ACTOR/FORWARD"),
        ("exit", AcceleratorType.NV_GPU, "ACTOR/FORWARD"),
    ]


def test_profile_range_rejects_dynamic_string_labels():
    with pytest.raises(TypeError, match="ProfileStage"):
        with profile_range("ACTOR/FORWARD"):
            pass


def test_profile_stage_vocabulary_is_stable():
    assert {stage.name: stage.value for stage in ProfileStage} == {
        "ACTOR_BATCH_PREP": "ACTOR/BATCH_PREP",
        "ACTOR_H2D": "ACTOR/H2D",
        "ACTOR_FORWARD": "ACTOR/FORWARD",
        "ACTOR_LOSS": "ACTOR/LOSS",
        "ACTOR_BACKWARD_NO_SYNC": "ACTOR/BACKWARD_NO_SYNC",
        "ACTOR_BACKWARD_SYNC": "ACTOR/BACKWARD_SYNC",
        "ACTOR_EMPTY_CACHE": "ACTOR/EMPTY_CACHE",
        "ACTOR_OPTIMIZER": "ACTOR/OPTIMIZER",
        "ACTOR_METRICS_REDUCE": "ACTOR/METRICS_REDUCE",
        "ROLLOUT_RECV_OBS": "ROLLOUT/RECV_OBS",
        "ROLLOUT_OBS_PREPROCESS": "ROLLOUT/OBS_PREPROCESS",
        "ROLLOUT_VLM_PREFIX": "ROLLOUT/VLM_PREFIX",
        "ROLLOUT_AE_DENOISE": "ROLLOUT/AE_DENOISE",
        "ROLLOUT_BOOTSTRAP_VALUE": "ROLLOUT/BOOTSTRAP_VALUE",
        "ROLLOUT_POSTPROCESS": "ROLLOUT/POSTPROCESS",
        "ROLLOUT_SEND_ACTION": "ROLLOUT/SEND_ACTION",
        "ENV_RECV_ACTION": "ENV/RECV_ACTION",
        "ENV_BOOTSTRAP_REWARD": "ENV/BOOTSTRAP_REWARD",
        "ENV_TRAJECTORY_APPEND": "ENV/TRAJECTORY_APPEND",
        "ENV_STEP": "ENV/STEP",
        "ENV_SEND_OBS": "ENV/SEND_OBS",
        "WEIGHT_SYNC_ACTOR_SEND": "WEIGHT_SYNC/ACTOR_SEND",
        "WEIGHT_SYNC_ROLLOUT_APPLY": "WEIGHT_SYNC/ROLLOUT_APPLY",
    }
