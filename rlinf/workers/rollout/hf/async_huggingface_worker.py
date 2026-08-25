# Copyright 2025 The RLinf Authors.
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

from omegaconf.omegaconf import DictConfig

from rlinf.profiling import ProfileStage, profile_range
from rlinf.scheduler import Channel, Worker
from rlinf.scheduler.stage_gate import InterProcessStageGate
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class AsyncMultiStepRolloutWorker(MultiStepRolloutWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._generate_task: asyncio.Task = None
        self.staleness_threshold = cfg.algorithm.get("staleness_threshold", None)
        # set the decoupled rollout worker sync weight time
        self.sync_rollout_weight_time = (
            self.num_pipeline_stages * self.n_train_chunk_steps * self.rollout_epoch
        )

        assert not self.enable_offload, (
            "Offload not supported in AsyncMultiStepRolloutWorker"
        )

        self._background_weight_sync_active = self.cfg.actor.get(
            "sync_weight_no_wait", False
        )
        self._weight_sync_requested = False
        self._weight_sync_work = None
        self._weight_sync_apply_total = 0
        self._weight_sync_coalesced_total = 0
        self._weight_sync_request_total = 0
        self._inference_pause_requested = False
        self._inference_resume_event = asyncio.Event()
        self._inference_resume_event.set()
        self._actor_vlm_stage_gate: InterProcessStageGate | None = None

    @Worker.timer("rollout/generate")
    async def generate(
        self,
        input_channel: Channel,
        output_channel: Channel,
        metric_channel: Channel,
    ):
        assert self._generate_task is None, (
            "generate task is not None but generate function is called."
        )
        self._generate_task = asyncio.create_task(
            self._generate(input_channel, output_channel, metric_channel)
        )
        try:
            await self._generate_task
        except asyncio.CancelledError:
            pass

    async def _generate(
        self,
        input_channel: Channel,
        output_channel: Channel,
        metric_channel: Channel,
    ):
        if self.env_decoupled_mode:
            await self.decoupled_generate_one_epoch(input_channel, output_channel)
        else:
            while True:
                if self._background_weight_sync_active:
                    await self._poll_background_weight_sync()

                for _ in range(self.rollout_epoch):
                    await self.generate_one_epoch(input_channel, output_channel)
                if self.finished_episodes is not None:
                    self.finished_episodes += (
                        self.total_num_train_envs * self.rollout_epoch
                    )
                rollout_metrics = self.pop_execution_times()
                rollout_metrics = {
                    f"time/rollout/{k}": v for k, v in rollout_metrics.items()
                }
                metric_channel.put(
                    {"rank": self._rank, "time": rollout_metrics},
                    async_op=True,
                )

    async def wait_if_stale(self) -> None:
        if self.staleness_threshold is None:
            return
        assert self.finished_episodes is not None, (
            "finished_episodes should be initialized."
        )
        while True:
            capacity = (
                (self.staleness_threshold + self.version + 1)
                * self.total_num_train_envs
                * self.rollout_epoch
            )
            if (
                self.finished_episodes + self.total_num_train_envs * self.rollout_epoch
                <= capacity
            ):
                break
            await asyncio.sleep(0.01)

    def stop(self):
        self._inference_pause_requested = False
        self._inference_resume_event.set()
        if self._generate_task is not None and not self._generate_task.done():
            self._generate_task.cancel()

    def configure_actor_vlm_stage_gate(self, gate_id: str) -> None:
        """Configure VLM exclusion against actor compute on this GPU."""
        self._actor_vlm_stage_gate = InterProcessStageGate(
            gate_id,
            node_rank=self._cluster_node_rank,
            accelerator_rank=self._local_accelerator_rank,
            wait_stage=ProfileStage.ROLLOUT_WAIT_ACTOR_COMPUTE,
            synchronize=self.torch_platform.synchronize,
            record_wait=lambda duration: self._record_stage_gate_wait(
                "vlm_actor_gate_wait", duration
            ),
            priority_waiter=True,
        )
        self.hf_model.set_rollout_vlm_stage_gate(self._actor_vlm_stage_gate)

    def _record_stage_gate_wait(self, tag: str, duration: float) -> None:
        self._timer_metrics[tag] = self._timer_metrics.get(tag, 0.0) + duration

    async def pause_inference_at_boundary(self) -> None:
        """Prevent the next policy inference from starting.

        Async worker methods share an event loop. If inference is currently
        running, this method is serviced only after that synchronous call
        returns, so its completion acknowledges a safe inference boundary.
        """
        self._inference_pause_requested = True
        self._inference_resume_event.clear()
        self.torch_platform.synchronize()

    async def resume_inference(self) -> None:
        self._inference_pause_requested = False
        self._inference_resume_event.set()

    async def _wait_for_inference_permission(self) -> None:
        while self._inference_pause_requested:
            await self._inference_resume_event.wait()

    async def _recv_and_apply_actor_sync(self) -> int:
        await super().sync_model_from_actor()
        return self.version

    def _start_background_weight_sync_if_needed(self):
        if (
            not self._background_weight_sync_active
            or not self._weight_sync_requested
            or self._weight_sync_work is not None
        ):
            return

        self._weight_sync_requested = False
        self._weight_sync_work = asyncio.create_task(self._recv_and_apply_actor_sync())

    @Worker.timer("rollout/poll_weight_sync")
    async def _poll_background_weight_sync(self):
        self._start_background_weight_sync_if_needed()
        if self._weight_sync_work is None:
            return

        if not self._weight_sync_work.done():
            return

        await self._weight_sync_work
        self._weight_sync_work = None
        self._weight_sync_apply_total += 1

        self._start_background_weight_sync_if_needed()

    @Worker.timer("rollout/request_weight_sync")
    async def request_actor_sync_model(self):
        self._weight_sync_request_total += 1
        if self._weight_sync_requested or self._weight_sync_work is not None:
            self._weight_sync_coalesced_total += 1
        self._weight_sync_requested = True
        self._start_background_weight_sync_if_needed()

    async def decoupled_generate_one_epoch(
        self, input_channel: Channel, output_channel: Channel
    ):
        self.update_dagger_beta()
        decoupled_generate_time = 1
        while True:
            if decoupled_generate_time % self.sync_rollout_weight_time == 0:
                self.update_dagger_beta()
                if self._background_weight_sync_active:
                    await self._poll_background_weight_sync()
                await self.wait_if_stale()
            decoupled_generate_time = decoupled_generate_time + 1
            with profile_range(ProfileStage.ROLLOUT_RECV_OBS):
                (
                    env_output,
                    split_sizes,
                ) = await self.recv_from_and_record_batch_routes_with_timeout(
                    group_name=self.cfg.env.group_name,
                    channel=input_channel,
                    tag="rollout_results",
                    batch_size=self.train_batch_size,
                    merge_fn=self._merge_obs_batches,
                    infer_batch_size_fn=self._infer_env_batch_size,
                    timeout_time=0.02,
                    recv_queue_size=self.rollout_queue_size,
                )
            await self._wait_for_inference_permission()
            actions, result = self._predict_rollout_actions(
                env_output["obs"],
                final_obs=env_output.get("final_obs", None),
                rlt_switch_flags=env_output.get("rlt_switch_flags", None),
                intervene_requested=env_output.get("intervene_flags", None),
            )
            policy_output = self._build_policy_output(
                actions,
                result,
                final_obs=env_output.get("final_obs", None),
            )
            with profile_range(ProfileStage.ROLLOUT_SEND_ACTION):
                self.send_to_recorded_batch_routes(
                    group_name=self.cfg.env.group_name,
                    channel=output_channel,
                    data=policy_output,
                    tag="rollout_results",
                    split_fn=self._split_policy_output,
                    split_sizes=split_sizes,
                )
