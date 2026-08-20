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

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from rlinf.profiling.fixture import ProfileFixtureCapture, ProfileFixtureKind
from toolkits.overlap_profiler.adapters import (
    ActorFSDPReplayAdapter,
    OpenPIReplayAdapter,
)


class _FakeOptimizer:
    def __init__(self):
        self.zero_grad_calls = 0
        self.param_groups = [{"lr": 1e-3}]

    def zero_grad(self):
        self.zero_grad_calls += 1


class _FakeScaler:
    @staticmethod
    def scale(loss):
        return loss


class _FakeActor:
    def __init__(self):
        self.device = "cpu"
        self.model = torch.nn.Linear(2, 1)
        self.optimizer = _FakeOptimizer()
        self.grad_scaler = _FakeScaler()
        self.gradient_accumulation = 1
        self.forward_calls = 0
        self.optimizer_calls = 0

    @staticmethod
    def before_micro_batch(_model, is_last_micro_batch):
        return nullcontext()

    def forward_and_loss_micro_batch(self, data):
        self.forward_calls += 1
        loss = self.model(data["forward_inputs"]["x"]).sum()
        return loss, {"loss": float(loss.detach())}, torch.tensor(0.0)

    def forward_micro_batch(self, data):
        self.forward_calls += 1
        return {"output": self.model(data["forward_inputs"]["x"])}, torch.ones(1)

    def optimizer_step(self):
        self.optimizer_calls += 1
        return 1.0, [0.0]


class _FakeOpenPIModel:
    def __init__(self):
        self.config = SimpleNamespace(
            num_steps=3,
            action_horizon=2,
            action_dim=2,
        )
        self.eval_calls = 0
        self.prefix_calls = 0
        self.denoise_calls = 0

    def eval(self):
        self.eval_calls += 1

    @staticmethod
    def _preprocess_observation(observation, train=False):
        state = observation["state"]
        return (state, state, state, state, state)

    def _build_prefix_cache(self, *_inputs):
        self.prefix_calls += 1
        value = torch.ones(1)
        return value, value, (value,)

    @staticmethod
    def sample_noise(shape, device):
        return torch.zeros(shape, device=device)

    def _sample_actions_with_prefix_cache(self, *args, **kwargs):
        self.denoise_calls += 1
        return {"actions": kwargs["noise"]}

    @staticmethod
    def predict_action_batch(env_obs, mode):
        return env_obs["states"], {"mode": mode}


class _TestOpenPIAdapter(OpenPIReplayAdapter):
    def _build_observation(self):
        return {"state": self.env_obs["states"]}


def _capture_fixture(tmp_path, kind, payload, metadata=None):
    capture = ProfileFixtureCapture(
        enabled=True,
        output_dir=tmp_path,
        kinds=[kind.value],
        role="test",
    )
    return capture.maybe_dump(
        kind,
        version=2,
        payload=payload,
        metadata=metadata,
    )


def test_actor_adapter_reuses_production_forward_and_loss(tmp_path):
    fixture = _capture_fixture(
        tmp_path,
        ProfileFixtureKind.ACTOR_MICROBATCH,
        {"data": {"forward_inputs": {"x": torch.ones(2, 2)}}},
        {"gradient_accumulation": 4, "micro_batch_size": 2},
    )
    actor = _FakeActor()
    adapter = ActorFSDPReplayAdapter(actor, str(fixture))

    adapter.prepare("actor.forward")
    output = adapter.run_once()

    assert output["output"].shape == (2, 1)
    assert actor.forward_calls == 1
    assert actor.gradient_accumulation == 4
    assert adapter.work_signature()["stage"] == "actor.forward"
    assert adapter.work_units() == {"microbatches": 1, "samples": 2}


def test_actor_backward_rebuilds_graph_before_timed_stage(tmp_path):
    fixture = _capture_fixture(
        tmp_path,
        ProfileFixtureKind.ACTOR_MICROBATCH,
        {"data": {"forward_inputs": {"x": torch.ones(2, 2)}}},
    )
    actor = _FakeActor()
    adapter = ActorFSDPReplayAdapter(actor, str(fixture))

    adapter.prepare("actor.backward_sync")
    adapter.prepare_iteration()
    assert actor.forward_calls == 1
    adapter.run_once()

    assert actor.model.weight.grad is not None


def test_actor_optimizer_replay_uses_zero_lr_and_restores_it(tmp_path):
    fixture = _capture_fixture(
        tmp_path,
        ProfileFixtureKind.ACTOR_MICROBATCH,
        {"data": {"forward_inputs": {"x": torch.ones(2, 2)}}},
    )
    actor = _FakeActor()
    adapter = ActorFSDPReplayAdapter(actor, str(fixture))

    adapter.prepare("actor.optimizer")
    assert actor.optimizer.param_groups[0]["lr"] == 0.0
    adapter.close()

    assert actor.optimizer.param_groups[0]["lr"] == 1e-3


def test_openpi_ae_prepare_keeps_prefix_outside_replayed_stage(tmp_path):
    fixture = _capture_fixture(
        tmp_path,
        ProfileFixtureKind.ROLLOUT_OBSERVATION,
        {
            "env_obs": {"states": torch.ones(2, 3)},
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": None,
        },
        {"mode": "train", "batch_size": 2, "num_action_chunks": 5},
    )
    model = _FakeOpenPIModel()
    adapter = _TestOpenPIAdapter(model, str(fixture))

    adapter.prepare("rollout.ae_denoise")
    assert model.prefix_calls == 1
    adapter.prepare_iteration()
    assert model.prefix_calls == 2
    result = adapter.run_once()

    assert model.prefix_calls == 2
    assert model.denoise_calls == 1
    assert result["actions"].shape == (2, 2, 2)
    assert adapter.work_signature()["num_steps"] == 3
    assert adapter.work_units()["denoise_iterations"] == 6
