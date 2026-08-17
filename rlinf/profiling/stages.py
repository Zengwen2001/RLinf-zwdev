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

"""Stable names and context management for execution-stage profiling."""

from contextlib import contextmanager
from enum import Enum
from typing import Iterator


class ProfileStage(str, Enum):
    """Stable execution-stage names emitted to profiling backends."""

    ACTOR_BATCH_PREP = "ACTOR/BATCH_PREP"
    ACTOR_H2D = "ACTOR/H2D"
    ACTOR_FORWARD = "ACTOR/FORWARD"
    ACTOR_LOSS = "ACTOR/LOSS"
    ACTOR_BACKWARD_NO_SYNC = "ACTOR/BACKWARD_NO_SYNC"
    ACTOR_BACKWARD_SYNC = "ACTOR/BACKWARD_SYNC"
    ACTOR_EMPTY_CACHE = "ACTOR/EMPTY_CACHE"
    ACTOR_OPTIMIZER = "ACTOR/OPTIMIZER"
    ACTOR_METRICS_REDUCE = "ACTOR/METRICS_REDUCE"

    ROLLOUT_RECV_OBS = "ROLLOUT/RECV_OBS"
    ROLLOUT_OBS_PREPROCESS = "ROLLOUT/OBS_PREPROCESS"
    ROLLOUT_VLM_PREFIX = "ROLLOUT/VLM_PREFIX"
    ROLLOUT_AE_DENOISE = "ROLLOUT/AE_DENOISE"
    ROLLOUT_POSTPROCESS = "ROLLOUT/POSTPROCESS"
    ROLLOUT_SEND_ACTION = "ROLLOUT/SEND_ACTION"

    ENV_REWARD_PREPARE = "ENV/REWARD_PREPARE"
    ENV_REWARD_WAIT = "ENV/REWARD_WAIT"
    ENV_RECV_ACTION = "ENV/RECV_ACTION"
    ENV_TRAJECTORY_APPEND = "ENV/TRAJECTORY_APPEND"
    ENV_STEP = "ENV/STEP"
    ENV_SEND_OBS = "ENV/SEND_OBS"

    REWARD_PREPARE = "REWARD/PREPARE"
    REWARD_MODEL_FORWARD = "REWARD/MODEL_FORWARD"
    REWARD_API_WAIT = "REWARD/API_WAIT"
    REWARD_PARSE = "REWARD/PARSE"

    WEIGHT_SYNC_ACTOR_SEND = "WEIGHT_SYNC/ACTOR_SEND"
    WEIGHT_SYNC_ROLLOUT_APPLY = "WEIGHT_SYNC/ROLLOUT_APPLY"


@contextmanager
def profile_range(stage: ProfileStage) -> Iterator[None]:
    """Annotate a stage when called from a worker, otherwise act as a no-op."""
    if not isinstance(stage, ProfileStage):
        raise TypeError(f"stage must be a ProfileStage, got {type(stage).__name__}")

    from rlinf.scheduler.hardware import AcceleratorUtil
    from rlinf.scheduler.worker.worker import Worker

    worker = Worker.current_worker
    if worker is None:
        yield
        return

    with AcceleratorUtil.profiling_range(worker.accelerator_type, stage.value):
        yield
