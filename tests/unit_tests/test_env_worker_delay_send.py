# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from collections import defaultdict

import torch
from omegaconf import OmegaConf

from rlinf.scheduler.manager import WorkerAddress
from rlinf.utils.delay_sampler import ConstantDelaySampler
from rlinf.workers.env.env_worker import EnvWorker


class _DummyChannel:
    pass


def _make_worker(delay_sampler=None, env_decoupled_mode=True):
    worker = EnvWorker.__new__(EnvWorker)
    worker._rank = 1
    worker._worker_address = WorkerAddress("EnvGroup", ranks=1)
    worker.cfg = OmegaConf.create({"rollout": {"group_name": "RolloutGroup"}})
    worker.env_decoupled_mode = env_decoupled_mode
    worker.delay_sampler = delay_sampler
    worker.send_to_calls = []

    def send_to(**kwargs):
        worker.send_to_calls.append(kwargs)

    worker.send_to = send_to
    return worker


def test_send_to_rollout_with_delay_sends_one_rank_batch_for_eval():
    worker = _make_worker(delay_sampler=ConstantDelaySampler(delay=0.0))
    channel = _DummyChannel()
    env_metrics = defaultdict(list)
    data = {
        "obs": {"state": torch.arange(4).reshape(4, 1)},
        "final_obs": None,
    }

    asyncio.run(
        worker._send_to_rollout_with_delay(
            channel,
            data,
            env_metrics=env_metrics,
            mode="eval",
        )
    )

    assert len(worker.send_to_calls) == 1
    call = worker.send_to_calls[0]
    assert call["group_name"] == "RolloutGroup"
    assert call["channel"] is channel
    assert call["data"] is data
    assert call["mode"] == "eval"
    assert call["tag"] == "rollout_results"
    assert call["env_decoupled_mode"] is True
    assert torch.equal(env_metrics["interact_delay"][0], torch.zeros(1))


def test_send_to_rollout_without_delay_does_not_record_delay_metric():
    worker = _make_worker(delay_sampler=None)
    channel = _DummyChannel()
    env_metrics = defaultdict(list)
    data = {
        "obs": {"state": torch.arange(4).reshape(4, 1)},
        "final_obs": None,
    }

    asyncio.run(
        worker._send_to_rollout_with_delay(
            channel,
            data,
            env_metrics=env_metrics,
            mode="train",
        )
    )

    assert len(worker.send_to_calls) == 1
    assert worker.send_to_calls[0]["data"] is data
    assert "interact_delay" not in env_metrics


def test_send_to_rollout_with_delay_sync_inside_running_loop():
    worker = _make_worker(delay_sampler=ConstantDelaySampler(delay=0.0))
    channel = _DummyChannel()
    env_metrics = defaultdict(list)
    data = {
        "obs": {"state": torch.arange(2).reshape(2, 1)},
        "final_obs": None,
    }

    async def call_sync_send():
        worker._send_to_rollout_with_delay_sync(
            channel,
            data,
            env_metrics=env_metrics,
            mode="eval",
        )

    asyncio.run(call_sync_send())

    assert len(worker.send_to_calls) == 1
    assert worker.send_to_calls[0]["data"] is data
    assert torch.equal(env_metrics["interact_delay"][0], torch.zeros(1))
