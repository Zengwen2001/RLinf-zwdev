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

"""
VLM-AE Pipeline Worker with 2-Stage Async Pipeline

Architecture:
    - 2 micro-batches pipeline: VLM(N+1) || AE(N)
    - VLM workers (rank 0,1,2): Process micro-batch, send KV to AE
    - AE worker (rank 3): Receive KV, run denoise, generate rollout results

Pipeline Timeline (TRUE OVERLAP with async recv):

    Time ─────────────────────────────────────────────────────────────────────►

    VLM Rank 0: ┌─ KV0 ─┐ ┌─ KV1 ─┐ ┌─ KV2 ─┐ ┌─ KV3 ─┐
                │ send  │ │ send  │ │ send  │ │ send  │
                └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘
                    │         │         │         │
                    ▼         ▼         ▼         ▼
    AE:           ┌─────────┐ ┌─────────┐ ┌─────────┐
                  │ Recv+   │ │ Recv+   │ │ Recv+   │
                  │ Denoise │ │ Denoise │ │ Denoise │
                  │ MB0     │ │ MB1     │ │ MB2     │
                  └────┬────┘ └────┬────┘ └────┬────┘
                       │           │           │
                       ▼           ▼           ▼
                  Rollout     Rollout     Rollout
                  Result      Result      Result

Key Insight for Overlap:
    1. VLM workers compute KV sequentially and send (synchronous NCCL send)
    2. AE worker starts receiving AS SOON AS first KV arrives
    3. While AE is denoising MB(N), VLM can compute and send MB(N+1)
    4. The overlap is achieved by:
       - AE using non-blocking recv to accumulate KV data
       - AE processing complete micro-batches as soon as all VLM data arrives
       - VLM continuing to compute next micro-batch

Implementation Notes:
    - NCCL send/recv is blocking from the sender side
    - AE can issue recv requests before VLM sends
    - The key is to pipeline the recv+dense sequence

Usage:
    export VLM_AE_DISAGG=1
    export VLM_AE_VLM_GPUS=0,1,2
    export VLM_AE_AE_GPU=3
    export VLM_AE_NUM_MICRO_BATCHES=2
"""

import os
import time
import threading
import asyncio
from typing import Any, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import queue

import torch

from rlinf.models.embodiment.openpi.vlm_ae_disagg import (
    MicroSliceSpec,
    VLM_AE_DISAGG_ENABLED,
    VLM_AE_NUM_MICRO_BATCHES,
    VLM_AE_VLM_GPUS,
    VLM_AE_AE_GPU,
    VLM_AE_DEBUG,
    WorkerRole,
    get_worker_role,
    get_num_vlm_workers,
    get_ae_worker_rank,
    get_ae_to_env_micro_dst_slices,
    get_vlm_env_src_ranks,
    get_vlm_env_micro_src_slices,
    get_ae_to_env_dst_ranks,
)
from rlinf.utils.logging import get_logger
from rlinf.utils.comm_mapping import CommMapper
from rlinf.utils.timeline_trace import append_timeline_event

_logger = None
BASELINE_MICROPIPE_ENABLED = os.environ.get("VLM_AE_BASELINE_MICROPIPE", "0") == "1"

def _get_logger():
    global _logger
    if _logger is None:
        _logger = get_logger()
    return _logger


def _append_worker_timeline_event(
    worker,
    *,
    component: str,
    tag: str,
    t0: float,
    t1: float,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    append_timeline_event(
        worker.cfg,
        component=component,
        rank=worker._rank,
        tag=tag,
        t0=t0,
        t1=t1,
        global_step=getattr(worker, "version", None),
        extra=extra,
    )


def _append_ae_model_phase_events(
    worker,
    *,
    stage_start_time: float,
    rollout_epoch_idx: int,
    chunk_step: int,
    stage_idx: int,
    micro_id: int,
) -> None:
    timing = getattr(worker.hf_model, "_last_ae_stage_timing", None)
    if not timing:
        return
    for phase_name, rel_t0, rel_t1 in timing.get("phases", []):
        _append_worker_timeline_event(
            worker,
            component="rollout_ae_detail",
            tag=(
                f"{phase_name}_e{rollout_epoch_idx}_cs{chunk_step}_st{stage_idx}"
                f"_mb{micro_id}"
            ),
            t0=stage_start_time + rel_t0,
            t1=stage_start_time + rel_t1,
            extra={
                "chunk_step": chunk_step,
                "stage_idx": stage_idx,
                "micro_batch_id": micro_id,
                "phase": phase_name,
                "role": "ae",
            },
        )


@dataclass
class PipelineConfig:
    """Configuration for VLM-AE pipeline."""
    enabled: bool = False
    num_micro_batches: int = 2
    batch_size: int = 24
    micro_batch_size: int = 12
    vlm_gpus: list = field(default_factory=lambda: [0, 1, 2])
    ae_gpu: int = 3

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        num_micro_batches = VLM_AE_NUM_MICRO_BATCHES
        batch_size = int(os.environ.get("VLM_AE_BATCH_SIZE", 48))
        return cls(
            enabled=VLM_AE_DISAGG_ENABLED,
            num_micro_batches=num_micro_batches,
            batch_size=batch_size,
            micro_batch_size=batch_size // num_micro_batches,
            vlm_gpus=VLM_AE_VLM_GPUS,
            ae_gpu=VLM_AE_AE_GPU,
        )


@dataclass
class KVPipelineMessage:
    """Tensor-only payload for VLM -> AE KV transfer."""

    request_id: int
    chunk_step: int
    micro_batch_id: int
    vlm_rank: int
    kv_keys: tuple[torch.Tensor, ...]
    kv_values: tuple[torch.Tensor, ...]
    prefix_output: Optional[torch.Tensor] = None
    prefix_pad_masks: Optional[torch.Tensor] = None
    state: Optional[torch.Tensor] = None
    vlm_value: Optional[torch.Tensor] = None
    obs_main_images: Optional[torch.Tensor] = None
    obs_wrist_images: Optional[torch.Tensor] = None
    obs_extra_view_images: Optional[torch.Tensor] = None
    obs_states: Optional[torch.Tensor] = None
    obs_task_descriptions: Optional[list[str]] = None
    final_main_images: Optional[torch.Tensor] = None
    final_wrist_images: Optional[torch.Tensor] = None
    final_extra_view_images: Optional[torch.Tensor] = None
    final_states: Optional[torch.Tensor] = None
    final_task_descriptions: Optional[list[str]] = None


class PipelineOrchestrator:
    """Coordinates 2-stage pipeline between VLM and AE workers."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._lock = threading.Lock()
        self._request_counter = 0

    def get_next_request_id(self) -> int:
        with self._lock:
            self._request_counter += 1
            return self._request_counter

    def split_obs_for_micro_batch(self, env_obs: dict, micro_batch_id: int) -> dict:
        """Split environment observations for a specific micro-batch."""
        micro_batch_size = self.config.micro_batch_size
        start = micro_batch_id * micro_batch_size
        end = start + micro_batch_size

        result = {}
        for key, value in env_obs.items():
            if isinstance(value, torch.Tensor):
                result[key] = value[start:end]
            elif isinstance(value, list):
                result[key] = value[start:end]
            else:
                result[key] = value
        return result


def _make_contiguous(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def _pack_kv_message(
    *,
    request_id: int,
    chunk_step: int,
    micro_batch_id: int,
    vlm_rank: int,
    kv_data: dict[str, Any],
    env_output: Optional[dict[str, Any]] = None,
) -> KVPipelineMessage:
    def _extract_obs_fields(obs: Optional[dict[str, Any]], *, prefix: str) -> dict[str, Any]:
        if obs is None:
            return {
                f"{prefix}_main_images": None,
                f"{prefix}_wrist_images": None,
                f"{prefix}_extra_view_images": None,
                f"{prefix}_states": None,
                f"{prefix}_task_descriptions": None,
            }
        task_descriptions = obs.get("task_descriptions")
        return {
            f"{prefix}_main_images": _make_contiguous(obs.get("main_images")),
            f"{prefix}_wrist_images": _make_contiguous(obs.get("wrist_images")),
            f"{prefix}_extra_view_images": _make_contiguous(obs.get("extra_view_images")),
            f"{prefix}_states": _make_contiguous(obs.get("states")),
            f"{prefix}_task_descriptions": (
                list(task_descriptions) if task_descriptions is not None else None
            ),
        }

    kv_cache = kv_data["kv_cache"]
    kv_keys = tuple(_make_contiguous(layer[0]) for layer in kv_cache)
    kv_values = tuple(_make_contiguous(layer[1]) for layer in kv_cache)
    env_fields = _extract_obs_fields(env_output.get("obs") if env_output else None, prefix="obs")
    final_obs_fields = _extract_obs_fields(
        env_output.get("final_obs") if env_output else None,
        prefix="final",
    )
    if (
        final_obs_fields["final_task_descriptions"] is None
        and env_fields["obs_task_descriptions"] is not None
    ):
        final_obs_fields["final_task_descriptions"] = list(
            env_fields["obs_task_descriptions"]
        )
    env_fields.update(final_obs_fields)
    return KVPipelineMessage(
        request_id=request_id,
        chunk_step=chunk_step,
        micro_batch_id=micro_batch_id,
        vlm_rank=vlm_rank,
        kv_keys=kv_keys,
        kv_values=kv_values,
        prefix_output=_make_contiguous(kv_data.get("prefix_output")),
        prefix_pad_masks=_make_contiguous(kv_data.get("prefix_pad_masks")),
        state=_make_contiguous(kv_data.get("state")),
        vlm_value=_make_contiguous(kv_data.get("vlm_value")),
        obs_main_images=env_fields["obs_main_images"],
        obs_wrist_images=env_fields["obs_wrist_images"],
        obs_extra_view_images=env_fields["obs_extra_view_images"],
        obs_states=env_fields["obs_states"],
        obs_task_descriptions=env_fields["obs_task_descriptions"],
        final_main_images=env_fields["final_main_images"],
        final_wrist_images=env_fields["final_wrist_images"],
        final_extra_view_images=env_fields["final_extra_view_images"],
        final_states=env_fields["final_states"],
        final_task_descriptions=env_fields["final_task_descriptions"],
    )


def _unpack_kv_message(message: KVPipelineMessage | dict[str, Any]) -> dict[str, Any]:
    if isinstance(message, dict):
        return message
    kv_cache = list(zip(message.kv_keys, message.kv_values, strict=False))
    return {
        "request_id": message.request_id,
        "chunk_step": message.chunk_step,
        "micro_batch_id": message.micro_batch_id,
        "vlm_rank": message.vlm_rank,
        "kv_cache": kv_cache,
        "prefix_output": message.prefix_output,
        "prefix_pad_masks": message.prefix_pad_masks,
        "state": message.state,
        "vlm_value": message.vlm_value,
        "obs_main_images": message.obs_main_images,
        "obs_wrist_images": message.obs_wrist_images,
        "obs_extra_view_images": message.obs_extra_view_images,
        "obs_states": message.obs_states,
        "obs_task_descriptions": message.obs_task_descriptions,
        "final_main_images": message.final_main_images,
        "final_wrist_images": message.final_wrist_images,
        "final_extra_view_images": message.final_extra_view_images,
        "final_states": message.final_states,
        "final_task_descriptions": message.final_task_descriptions,
    }


async def _await_kv_recv(
    recv_work,
    *,
    src_rank: int,
    micro_batch_id: int,
    t0: float,
) -> dict[str, Any]:
    payload = await recv_work.async_wait()
    return {
        "payload": payload,
        "src_rank": src_rank,
        "micro_batch_id": micro_batch_id,
        "t0": t0,
        "t_ready": time.time(),
    }


def _wait_for_kv_recv(
    recv_work,
    completion_queue: "queue.Queue[dict[str, Any]]",
    *,
    src_rank: int,
    micro_batch_id: int,
    t0: float,
) -> None:
    payload = recv_work.wait()
    completion_queue.put(
        {
            "payload": payload,
            "src_rank": src_rank,
            "micro_batch_id": micro_batch_id,
            "t0": t0,
            "t_ready": time.time(),
        }
    )


# Global orchestrator
_orchestrator: Optional[PipelineOrchestrator] = None

def get_orchestrator(config: Optional[PipelineConfig] = None) -> PipelineOrchestrator:
    global _orchestrator
    if not VLM_AE_DISAGG_ENABLED:
        return None
    if config is None:
        config = PipelineConfig.from_env()
    if (
        _orchestrator is None
        or _orchestrator.config.batch_size != config.batch_size
        or _orchestrator.config.num_micro_batches != config.num_micro_batches
    ):
        if config is None:
            config = PipelineConfig.from_env()
        _orchestrator = PipelineOrchestrator(config)
    return _orchestrator


def patch_huggingface_worker_for_pipeline():
    """
    Patch MultiStepRolloutWorker for 2-stage VLM-AE pipeline.

    Key changes:
    - VLM workers: Process micro-batches, send KV to AE via NCCL P2P
    - AE worker: Receive KV from VLM workers, run denoise, send rollout results
    - Pipeline overlap: AE starts denoise as soon as KV for a micro-batch is complete
    """
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker
    from rlinf.scheduler import Channel
    from rlinf.data.embodied_io_struct import RolloutResult

    if hasattr(MultiStepRolloutWorker, '_vlm_ae_patched'):
        _get_logger().info("[VLM-AE Patch] Already patched, skipping")
        return

    original_init = MultiStepRolloutWorker.init_worker
    original_generate_one_epoch = MultiStepRolloutWorker.generate_one_epoch

    NUM_VLM_WORKERS = get_num_vlm_workers()
    AE_WORKER_RANK = get_ae_worker_rank()

    def patched_init_worker(self):
        role_hint = get_worker_role() if VLM_AE_DISAGG_ENABLED else WorkerRole.UNIFIED
        if self.cfg.rollout.get("enable_torch_compile", False) and (
            VLM_AE_DISAGG_ENABLED or BASELINE_MICROPIPE_ENABLED
        ):
            # Isolate Inductor/Triton caches per rollout rank to avoid concurrent
            # autotune writers corrupting shared benchmark cache files.
            cache_root = os.environ.get(
                "RLINF_TORCH_COMPILE_CACHE_ROOT",
                os.path.join("/tmp", "rlinf_torch_compile_cache"),
            )
            rank_cache_dir = os.path.join(cache_root, f"rollout_rank_{self._rank}")
            os.makedirs(rank_cache_dir, exist_ok=True)
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(
                rank_cache_dir, "inductor"
            )
            os.environ["TRITON_CACHE_DIR"] = os.path.join(rank_cache_dir, "triton")
            os.makedirs(os.environ["TORCHINDUCTOR_CACHE_DIR"], exist_ok=True)
            os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)
        original_compile_mode = self.cfg.rollout.get(
            "torch_compile_mode", "max-autotune-no-cudagraphs"
        )
        if (
            VLM_AE_DISAGG_ENABLED
            and role_hint == WorkerRole.AE
            and self.cfg.rollout.get("enable_torch_compile", False)
            and original_compile_mode == "max-autotune"
        ):
            # Keep AE on torch.compile, but avoid CUDA graph pools for the cache-heavy
            # denoise path. This preserves compile while preventing graph-pool OOMs.
            self.cfg.rollout.torch_compile_mode = "max-autotune-no-cudagraphs"

        original_init(self)
        if VLM_AE_DISAGG_ENABLED:
            base_config = PipelineConfig.from_env()
            self._vlm_ae_role = role_hint
            env_world_size = self.placement.get_world_size("env")
            stage_batch_size = self.total_num_train_envs // self.num_pipeline_stages
            vlm_local_batch_size = stage_batch_size // NUM_VLM_WORKERS

            if self._world_size != NUM_VLM_WORKERS + 1:
                raise ValueError(
                    "VLM-AE disaggregation requires rollout world size to equal "
                    f"num_vlm_workers + 1, got rollout_world_size={self._world_size}, "
                    f"num_vlm_workers={NUM_VLM_WORKERS}."
                )
            if stage_batch_size % NUM_VLM_WORKERS != 0:
                raise ValueError(
                    "Per-stage env batch size must be divisible by the number of VLM workers, "
                    f"got stage_batch_size={stage_batch_size}, num_vlm_workers={NUM_VLM_WORKERS}."
                )
            if vlm_local_batch_size % base_config.num_micro_batches != 0:
                raise ValueError(
                    "Per-VLM-worker batch size must be divisible by num_micro_batches, "
                    f"got vlm_local_batch_size={vlm_local_batch_size}, "
                    f"num_micro_batches={base_config.num_micro_batches}."
                )
            if stage_batch_size % base_config.num_micro_batches != 0:
                raise ValueError(
                    "AE stage batch size must be divisible by num_micro_batches, "
                    f"got stage_batch_size={stage_batch_size}, "
                    f"num_micro_batches={base_config.num_micro_batches}."
                )
            if self.enable_cuda_graph:
                raise ValueError(
                    "VLM-AE disaggregation does not support cuda graph capture yet."
                )

            role_batch_size = (
                stage_batch_size
                if self._vlm_ae_role == WorkerRole.AE
                else vlm_local_batch_size
            )
            self.train_batch_size = role_batch_size
            self._vlm_ae_config = PipelineConfig(
                enabled=base_config.enabled,
                num_micro_batches=base_config.num_micro_batches,
                batch_size=role_batch_size,
                micro_batch_size=role_batch_size // base_config.num_micro_batches,
                vlm_gpus=base_config.vlm_gpus,
                ae_gpu=base_config.ae_gpu,
            )
            self._vlm_ae_orchestrator = get_orchestrator(self._vlm_ae_config)
            self._vlm_ae_request_counter = 0
            self._vlm_ae_lock = threading.Lock()
            self._vlm_ae_pending_sends = []
            self._vlm_ae_send_warmup_done = False
            self._vlm_ae_max_pending_sends = max(
                1,
                int(
                    os.environ.get(
                        "VLM_AE_MAX_PENDING_SENDS",
                        str(self._vlm_ae_config.num_micro_batches + 1),
                    )
                ),
            )

        # Rewire env<->rollout communication only in disagg mode:
        # - VLM ranks consume sharded env obs
        # - AE rank consumes only VLM bundles and is the only result producer
            if self._vlm_ae_role == WorkerRole.VLM:
                self.src_ranks["train"] = get_vlm_env_src_ranks(
                    stage_batch_size, env_world_size, self._rank
                )
                self.dst_ranks["train"] = []
                self._vlm_ae_env_micro_src_specs = {
                    "train": get_vlm_env_micro_src_slices(
                        stage_batch_size,
                        env_world_size,
                        self._rank,
                        self._vlm_ae_config.num_micro_batches,
                    )
                }
            elif self._vlm_ae_role == WorkerRole.AE:
                self.src_ranks["train"] = []
                self.dst_ranks["train"] = get_ae_to_env_dst_ranks(
                    stage_batch_size, env_world_size
                )
                self._ae_env_micro_dst_specs = {
                    "train": get_ae_to_env_micro_dst_slices(
                        stage_batch_size,
                        env_world_size,
                        self._vlm_ae_config.num_micro_batches,
                    )
                }

            if hasattr(self.hf_model, 'setup_vlm_ae_disagg'):
                self.hf_model.setup_vlm_ae_disagg(batch_size=role_batch_size)

            _get_logger().info(f"[VLM-AE Patch] Worker role={self._vlm_ae_role}, rank={self._rank}, "
                              f"num_micro_batches={self._vlm_ae_config.num_micro_batches}, "
                              f"torch_compile_enabled={self.enable_torch_compile}, "
                              f"src_ranks={self.src_ranks.get('train')}, "
                              f"dst_ranks={self.dst_ranks.get('train')}")
        elif BASELINE_MICROPIPE_ENABLED:
            num_micro_batches = int(os.environ.get("VLM_AE_NUM_MICRO_BATCHES", "2"))
            if self.train_batch_size % num_micro_batches != 0:
                raise ValueError(
                    f"Baseline micro pipeline requires train_batch_size={self.train_batch_size} "
                    f"to be divisible by num_micro_batches={num_micro_batches}."
                )
            if len(self.src_ranks["train"]) != 1 or len(self.dst_ranks["train"]) != 1:
                raise ValueError(
                    "Baseline micro pipeline currently requires 1:1 env<->rollout mapping."
                )
            self._baseline_micro_num_micro_batches = num_micro_batches
            self._baseline_micro_batch_size = self.train_batch_size // num_micro_batches
            _get_logger().info(
                "[VLM-AE Patch] Enabled baseline micro pipeline, rank=%s, "
                "num_micro_batches=%s, micro_batch_size=%s",
                self._rank,
                self._baseline_micro_num_micro_batches,
                self._baseline_micro_batch_size,
            )

    def _wait_one_pending_vlm_send(self, pending_send: dict[str, Any]) -> float:
        pending_send["work"].wait()
        t_done = time.time()
        _append_worker_timeline_event(
            self,
            component="kv_transfer",
            tag=pending_send["tag"],
            t0=pending_send["t0"],
            t1=t_done,
            extra={
                "chunk_step": pending_send["chunk_step"],
                "stage_idx": pending_send["stage_idx"],
                "micro_batch_id": pending_send["micro_batch_id"],
                "src_rank": self._rank,
                "dst_rank": AE_WORKER_RANK,
                "direction": "vlm_to_ae",
            },
        )
        return t_done - pending_send["t0"]

    def _flush_pending_vlm_sends(self, keep: int = 0) -> float:
        pending_sends = getattr(self, "_vlm_ae_pending_sends", None)
        if not pending_sends:
            return 0.0

        total_wait = 0.0
        while len(pending_sends) > keep:
            total_wait += self._wait_one_pending_vlm_send(pending_sends.pop(0))
        return total_wait

    def _dispatch_vlm_send(
        self,
        send_data: KVPipelineMessage,
        *,
        rollout_epoch_idx: int,
        chunk_step: int,
        stage_idx: int,
        micro_id: int,
        tail: bool = False,
    ) -> float:
        if not hasattr(self, "_vlm_ae_pending_sends"):
            self._vlm_ae_pending_sends = []

        tag = (
            f"send_tail_e{rollout_epoch_idx}_st{stage_idx}_mb{micro_id}"
            if tail
            else f"send_e{rollout_epoch_idx}_cs{chunk_step}_st{stage_idx}_mb{micro_id}"
        )
        t0 = time.time()

        # Keep the first transfer synchronous to eagerly establish the NCCL P2P path.
        if not getattr(self, "_vlm_ae_send_warmup_done", False):
            self.send(
                send_data,
                dst_group_name=self._group_name,
                dst_rank=AE_WORKER_RANK,
                async_op=False,
            )
            self._vlm_ae_send_warmup_done = True
            t1 = time.time()
            _append_worker_timeline_event(
                self,
                component="kv_transfer",
                tag=tag,
                t0=t0,
                t1=t1,
                extra={
                    "chunk_step": chunk_step,
                    "stage_idx": stage_idx,
                    "micro_batch_id": micro_id,
                    "src_rank": self._rank,
                    "dst_rank": AE_WORKER_RANK,
                    "direction": "vlm_to_ae",
                },
            )
            return t1 - t0

        send_work = self.send(
            send_data,
            dst_group_name=self._group_name,
            dst_rank=AE_WORKER_RANK,
            async_op=True,
        )
        self._vlm_ae_pending_sends.append(
            {
                "work": send_work,
                "t0": t0,
                "tag": tag,
                "chunk_step": chunk_step,
                "stage_idx": stage_idx,
                "micro_batch_id": micro_id,
            }
        )

        max_pending = getattr(self, "_vlm_ae_max_pending_sends", 1)
        if len(self._vlm_ae_pending_sends) > max_pending:
            return self._flush_pending_vlm_sends(keep=max_pending)
        return 0.0

    async def _recv_env_micro_output(
        self,
        input_channel: Channel,
        micro_batch_id: int,
        mode: str = "train",
    ) -> dict[str, Any]:
        if self._vlm_ae_role == WorkerRole.VLM:
            specs = self._vlm_ae_env_micro_src_specs[mode][micro_batch_id]
        else:
            raise ValueError(
                f"Unsupported role for micro env recv: {self._vlm_ae_role}. "
                "In disagg mode only VLM ranks should receive env micro-batches."
            )

        obs_batches: list[tuple[int, dict[str, Any]]] = []
        for spec in specs:
            obs_batch = await input_channel.get(
                key=CommMapper.build_channel_key(
                    spec.peer_rank,
                    self._rank,
                    extra=f"{mode}_obs_mb{micro_batch_id}_ord{spec.order}",
                ),
                async_op=True,
            ).async_wait()
            actual_size = self._infer_env_batch_size(obs_batch)
            assert actual_size == spec.size, (
                f"Expected env micro batch size {spec.size} from env rank {spec.peer_rank}, "
                f"got {actual_size} for micro_batch_id={micro_batch_id}."
            )
            obs_batches.append((spec.dst_start, obs_batch))

        ordered_obs_batches = [batch for _, batch in sorted(obs_batches, key=lambda item: item[0])]
        return self._merge_obs_batches(ordered_obs_batches)

    def _slice_rollout_result(
        self, rollout_result: RolloutResult, start: int, end: int
    ) -> RolloutResult:
        return RolloutResult(
            actions=rollout_result.actions[start:end] if rollout_result.actions is not None else None,
            prev_logprobs=rollout_result.prev_logprobs[start:end]
            if rollout_result.prev_logprobs is not None
            else None,
            prev_values=rollout_result.prev_values[start:end]
            if rollout_result.prev_values is not None
            else None,
            bootstrap_values=rollout_result.bootstrap_values[start:end]
            if rollout_result.bootstrap_values is not None
            else None,
            save_flags=rollout_result.save_flags[start:end]
            if rollout_result.save_flags is not None
            else None,
            forward_inputs={
                key: value[start:end].contiguous()
                for key, value in rollout_result.forward_inputs.items()
            },
            versions=rollout_result.versions[start:end]
            if rollout_result.versions is not None
            else None,
        )

    def _send_rollout_result_micro(
        self,
        output_channel: Channel,
        rollout_result: RolloutResult,
        micro_batch_id: int,
        mode: str = "train",
    ) -> None:
        specs = self._ae_env_micro_dst_specs[mode][micro_batch_id]
        for spec in specs:
            rollout_result_i = self._slice_rollout_result(
                rollout_result, spec.src_start, spec.src_end
            )
            output_channel.put(
                rollout_result_i,
                key=CommMapper.build_channel_key(
                    self._rank,
                    spec.peer_rank,
                    extra=f"{mode}_rollout_results_mb{micro_batch_id}_ord{spec.order}",
                ),
                async_op=True,
            )

    async def _recv_baseline_micro_env_output(
        self,
        input_channel: Channel,
        micro_batch_id: int,
        mode: str = "train",
    ) -> dict[str, Any]:
        src_rank, _ = self.src_ranks[mode][0]
        return await input_channel.get(
            key=CommMapper.build_channel_key(
                src_rank, self._rank, extra=f"{mode}_obs_mb{micro_batch_id}"
            ),
            async_op=True,
        ).async_wait()

    def _send_baseline_micro_rollout_result(
        self,
        output_channel: Channel,
        rollout_result: RolloutResult,
        micro_batch_id: int,
        mode: str = "train",
    ) -> None:
        dst_rank, _ = self.dst_ranks[mode][0]
        output_channel.put(
            rollout_result,
            key=CommMapper.build_channel_key(
                self._rank, dst_rank, extra=f"{mode}_rollout_results_mb{micro_batch_id}"
            ),
            async_op=True,
        )

    async def _unified_micro_generate_one_epoch(
        self, input_channel: Channel, output_channel: Channel, rollout_epoch_idx: int
    ):
        self.update_dagger_beta()
        num_micro_batches = self._baseline_micro_num_micro_batches

        for chunk_step in range(self.n_train_chunk_steps):
            for stage_idx in range(self.num_pipeline_stages):
                for micro_id in range(num_micro_batches):
                    env_output = await self._recv_baseline_micro_env_output(
                        input_channel, micro_id, mode="train"
                    )
                    t0 = time.time()
                    actions, result = self.predict(env_output["obs"])
                    t1 = time.time()
                    _append_worker_timeline_event(
                        self,
                        component="rollout",
                        tag=f"predict_e{rollout_epoch_idx}_cs{chunk_step}_st{stage_idx}_mb{micro_id}",
                        t0=t0,
                        t1=t1,
                        extra={
                            "chunk_step": chunk_step,
                            "stage_idx": stage_idx,
                            "micro_batch_id": micro_id,
                            "role": "unified",
                        },
                    )
                    if hasattr(self, "append_baseline_model_timeline"):
                        self.append_baseline_model_timeline(
                            result.get("_baseline_timing"),
                            rollout_epoch_idx=rollout_epoch_idx,
                            chunk_step=chunk_step,
                            stage_idx=stage_idx,
                            micro_batch_id=micro_id,
                        )
                    save_flags = None
                    if result.get("expert_label_flag", False):
                        save_flags = torch.full(
                            (actions.shape[0], self.cfg.actor.model.num_action_chunks),
                            True,
                            dtype=torch.bool,
                            device=actions.device,
                        )
                    prev_logprobs = result.get("prev_logprobs")
                    version_ref = (
                        prev_logprobs
                        if prev_logprobs is not None
                        else result["prev_values"]
                    )
                    rollout_result = RolloutResult(
                        actions=actions,
                        prev_logprobs=prev_logprobs if self.collect_prev_infos else None,
                        prev_values=result["prev_values"]
                        if self.collect_prev_infos
                        else None,
                        bootstrap_values=self.get_bootstrap_values(
                            env_output.get("final_obs", None)
                        ),
                        save_flags=save_flags,
                        forward_inputs=result["forward_inputs"],
                        versions=torch.full(
                            (actions.shape[0], 1),
                            float(self.version),
                            dtype=torch.float32,
                            device=version_ref.device if version_ref is not None else actions.device,
                        ),
                    )
                    self._send_baseline_micro_rollout_result(
                        output_channel, rollout_result, micro_id, mode="train"
                    )

        for stage_idx in range(self.num_pipeline_stages):
            for micro_id in range(num_micro_batches):
                env_output = await self._recv_baseline_micro_env_output(
                    input_channel, micro_id, mode="train"
                )
                t0 = time.time()
                actions, result = self.predict(env_output["obs"])
                t1 = time.time()
                _append_worker_timeline_event(
                    self,
                    component="rollout",
                    tag=f"predict_tail_e{rollout_epoch_idx}_st{stage_idx}_mb{micro_id}",
                    t0=t0,
                    t1=t1,
                    extra={
                        "chunk_step": -1,
                        "stage_idx": stage_idx,
                        "micro_batch_id": micro_id,
                        "role": "unified",
                    },
                )
                if hasattr(self, "append_baseline_model_timeline"):
                    self.append_baseline_model_timeline(
                        result.get("_baseline_timing"),
                        rollout_epoch_idx=rollout_epoch_idx,
                        chunk_step=-1,
                        stage_idx=stage_idx,
                        micro_batch_id=micro_id,
                    )
                rollout_result = RolloutResult(
                    actions=actions,
                    prev_values=result["prev_values"] if self.collect_prev_infos else None,
                    bootstrap_values=self.get_bootstrap_values(
                        env_output.get("final_obs", None)
                    ),
                )
                self._send_baseline_micro_rollout_result(
                    output_channel, rollout_result, micro_id, mode="train"
                )

    async def patched_generate_one_epoch(self, input_channel: Channel, output_channel: Channel, rollout_epoch_idx: int):
        if BASELINE_MICROPIPE_ENABLED and not VLM_AE_DISAGG_ENABLED:
            return await self._unified_micro_generate_one_epoch(
                input_channel, output_channel, rollout_epoch_idx
            )
        if not VLM_AE_DISAGG_ENABLED:
            from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker
            return await MultiStepRolloutWorker.original_generate_one_epoch(
                self, input_channel, output_channel, rollout_epoch_idx
            )

        if self._vlm_ae_role == WorkerRole.VLM:
            return await self._vlm_generate_one_epoch(input_channel, output_channel, rollout_epoch_idx)
        elif self._vlm_ae_role == WorkerRole.AE:
            return await self._ae_generate_one_epoch(input_channel, output_channel, rollout_epoch_idx)
        else:
            from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker
            return await MultiStepRolloutWorker.original_generate_one_epoch(
                self, input_channel, output_channel, rollout_epoch_idx
            )

    async def _vlm_generate_one_epoch(self, input_channel: Channel, output_channel: Channel, rollout_epoch_idx: int):
        """
        VLM worker: Process all micro-batches sequentially, send KV to AE.

        The VLM worker:
        1. Receives env_output for its local shard
        2. Splits obs into micro-batches
        3. For each micro-batch:
           - Computes KV cache
           - Bundles AE-needed obs/final_obs metadata
           - Sends the unified bundle to AE worker via NCCL P2P
        4. Does NOT send rollout results (AE handles that)

        Pipeline Overlap:
        - VLM sends are synchronous NCCL operations
        - AE can start receiving as soon as first send completes
        - While AE denoises MB(N), VLM computes KV for MB(N+1)
        """
        num_micro_batches = self._vlm_ae_config.num_micro_batches

        self.update_dagger_beta()

        # Main chunk steps
        for chunk_step in range(self.n_train_chunk_steps):
            for stage_idx in range(self.num_pipeline_stages):
                stage_kv_total = 0.0
                stage_send_total = 0.0

                # Process each micro-batch sequentially
                for micro_id in range(num_micro_batches):
                    env_output = await self._recv_env_micro_output(
                        input_channel, micro_id, mode="train"
                    )
                    micro_obs = env_output["obs"]

                    with self._vlm_ae_lock:
                        self._vlm_ae_request_counter += 1
                        request_id = self._vlm_ae_request_counter

                    # Compute KV cache
                    t0 = time.time()
                    kv_data = self.hf_model._predict_vlm_stage(micro_obs)
                    t1 = time.time()
                    stage_kv_total += t1 - t0
                    _append_worker_timeline_event(
                        self,
                        component="rollout_vlm",
                        tag=f"kv_e{rollout_epoch_idx}_cs{chunk_step}_st{stage_idx}_mb{micro_id}",
                        t0=t0,
                        t1=t1,
                        extra={
                            "chunk_step": chunk_step,
                            "stage_idx": stage_idx,
                            "micro_batch_id": micro_id,
                            "role": "vlm",
                        },
                    )

                    send_data = _pack_kv_message(
                        request_id=request_id,
                        chunk_step=chunk_step,
                        micro_batch_id=micro_id,
                        vlm_rank=self._rank,
                        kv_data=kv_data,
                        env_output=env_output,
                    )

                    stage_send_total += self._dispatch_vlm_send(
                        send_data,
                        rollout_epoch_idx=rollout_epoch_idx,
                        chunk_step=chunk_step,
                        stage_idx=stage_idx,
                        micro_id=micro_id,
                        tail=False,
                    )

                if VLM_AE_DEBUG:
                    _get_logger().debug(
                        "[VLM-AE] rank=%s chunk=%s stage=%s kv_ms=%.1f send_ms=%.1f",
                        self._rank,
                        chunk_step,
                        stage_idx,
                        stage_kv_total * 1000.0,
                        stage_send_total * 1000.0,
                    )

        # Tail steps
        for stage_idx in range(self.num_pipeline_stages):
            stage_kv_total = 0.0
            stage_send_total = 0.0

            for micro_id in range(num_micro_batches):
                env_output = await self._recv_env_micro_output(
                    input_channel, micro_id, mode="train"
                )
                micro_obs = env_output["obs"]

                with self._vlm_ae_lock:
                    self._vlm_ae_request_counter += 1
                    request_id = self._vlm_ae_request_counter

                t0 = time.time()
                kv_data = self.hf_model._predict_vlm_stage(micro_obs)
                t1 = time.time()
                stage_kv_total += t1 - t0
                _append_worker_timeline_event(
                    self,
                    component="rollout_vlm",
                    tag=f"kv_tail_e{rollout_epoch_idx}_st{stage_idx}_mb{micro_id}",
                    t0=t0,
                    t1=t1,
                    extra={
                        "chunk_step": -1,
                        "stage_idx": stage_idx,
                        "micro_batch_id": micro_id,
                        "role": "vlm",
                    },
                )

                send_data = _pack_kv_message(
                    request_id=request_id,
                    chunk_step=-1,
                    micro_batch_id=micro_id,
                    vlm_rank=self._rank,
                    kv_data=kv_data,
                    env_output=env_output,
                )

                stage_send_total += self._dispatch_vlm_send(
                    send_data,
                    rollout_epoch_idx=rollout_epoch_idx,
                    chunk_step=-1,
                    stage_idx=stage_idx,
                    micro_id=micro_id,
                    tail=True,
                )

            if VLM_AE_DEBUG:
                _get_logger().debug(
                    "[VLM-AE] rank=%s tail_stage=%s kv_ms=%.1f send_ms=%.1f",
                    self._rank,
                    stage_idx,
                    stage_kv_total * 1000.0,
                    stage_send_total * 1000.0,
                )

        if getattr(self, "_vlm_ae_pending_sends", None):
            self._flush_pending_vlm_sends()

    async def _ae_generate_one_epoch(self, input_channel: Channel, output_channel: Channel, rollout_epoch_idx: int):
        """
        AE worker: Receive KV from VLM workers, run denoise, send rollout results.

        The AE worker:
        1. Issues async recv for all VLM workers
        2. As VLM bundles arrive, accumulates complete micro-batches
        3. Merges KV plus obs/final_obs metadata from the VLM bundles
        4. Runs denoise and sends rollout results

        Pipeline Overlap Strategy:
        - AE receives KV data in a pipelined manner
        - As soon as all 3 VLM workers have sent KV for a micro-batch,
          AE can start denoise
        - While AE denoises, VLM continues computing next micro-batch
        """
        num_micro_batches = self._vlm_ae_config.num_micro_batches
        num_vlm_workers = NUM_VLM_WORKERS

        # Main chunk steps
        for chunk_step in range(self.n_train_chunk_steps):
            for stage_idx in range(self.num_pipeline_stages):
                stage_t0 = time.time()
                recv_total = 0.0
                process_total = 0.0

                micro_batch_kv = {i: [] for i in range(num_micro_batches)}
                received_counts = {i: 0 for i in range(num_micro_batches)}
                recv_completion_queue, total_expected = self._issue_async_kv_recvs(
                    num_micro_batches, num_vlm_workers
                )
                received_total = 0
                next_micro_to_process = 0

                while received_total < total_expected:
                    recv_result = recv_completion_queue.get()
                    received_total += 1
                    t_unpack0 = time.time()
                    kv_payload = recv_result["payload"]
                    kv_data = _unpack_kv_message(kv_payload)
                    t_unpack1 = time.time()
                    micro_id = kv_data.get("micro_batch_id", recv_result["micro_batch_id"])
                    vlm_rank = kv_data.get("vlm_rank", recv_result["src_rank"])
                    micro_batch_kv[micro_id].append(kv_data)
                    received_counts[micro_id] += 1
                    recv_total += t_unpack1 - recv_result["t0"]
                    _append_worker_timeline_event(
                        self,
                        component="kv_wait",
                        tag=(
                            f"recv_e{rollout_epoch_idx}_cs{chunk_step}_st{stage_idx}"
                            f"_mb{micro_id}_src{vlm_rank}"
                        ),
                        t0=recv_result["t0"],
                        t1=recv_result["t_ready"],
                        extra={
                            "chunk_step": chunk_step,
                            "stage_idx": stage_idx,
                            "micro_batch_id": micro_id,
                            "src_rank": vlm_rank,
                            "dst_rank": self._rank,
                            "direction": "vlm_to_ae",
                        },
                    )
                    if t_unpack0 > recv_result["t_ready"]:
                        _append_worker_timeline_event(
                            self,
                            component="kv_queue",
                            tag=(
                                f"queue_e{rollout_epoch_idx}_cs{chunk_step}_st{stage_idx}"
                                f"_mb{micro_id}_src{vlm_rank}"
                            ),
                            t0=recv_result["t_ready"],
                            t1=t_unpack0,
                            extra={
                                "chunk_step": chunk_step,
                                "stage_idx": stage_idx,
                                "micro_batch_id": micro_id,
                                "src_rank": vlm_rank,
                                "dst_rank": self._rank,
                                "direction": "vlm_to_ae",
                            },
                        )
                    _append_worker_timeline_event(
                        self,
                        component="kv_unpack",
                        tag=(
                            f"unpack_e{rollout_epoch_idx}_cs{chunk_step}_st{stage_idx}"
                            f"_mb{micro_id}_src{vlm_rank}"
                        ),
                        t0=t_unpack0,
                        t1=t_unpack1,
                        extra={
                            "chunk_step": chunk_step,
                            "stage_idx": stage_idx,
                            "micro_batch_id": micro_id,
                            "src_rank": vlm_rank,
                            "dst_rank": self._rank,
                            "direction": "vlm_to_ae",
                        },
                    )

                    while (
                        next_micro_to_process < num_micro_batches
                        and received_counts[next_micro_to_process] == num_vlm_workers
                    ):
                        t_process_start = time.time()
                        merged_kv = self._merge_kv_from_vlm_workers(
                            micro_batch_kv[next_micro_to_process]
                        )
                        actions, result = self.hf_model._predict_ae_stage(
                            kv_data=merged_kv,
                            env_obs=merged_kv.get("env_obs"),
                            mode="train",
                            compute_values=True,
                        )
                        t_process_end = time.time()
                        process_total += t_process_end - t_process_start
                        _append_ae_model_phase_events(
                            self,
                            stage_start_time=t_process_start,
                            rollout_epoch_idx=rollout_epoch_idx,
                            chunk_step=chunk_step,
                            stage_idx=stage_idx,
                            micro_id=next_micro_to_process,
                        )
                        _append_worker_timeline_event(
                            self,
                            component="rollout_ae",
                            tag=(
                                f"denoise_e{rollout_epoch_idx}_cs{chunk_step}_st{stage_idx}"
                                f"_mb{next_micro_to_process}"
                            ),
                            t0=t_process_start,
                            t1=t_process_end,
                            extra={
                                "chunk_step": chunk_step,
                                "stage_idx": stage_idx,
                                "micro_batch_id": next_micro_to_process,
                                "role": "ae",
                            },
                        )
                        save_flags = None
                        if result.get("expert_label_flag", False):
                            save_flags = torch.full(
                                (actions.shape[0], self.cfg.actor.model.num_action_chunks),
                                True,
                                dtype=torch.bool,
                                device=actions.device,
                            )

                        version_ref = (
                            result["prev_logprobs"]
                            if result.get("prev_logprobs") is not None
                            else result["prev_values"]
                        )
                        rollout_result = RolloutResult(
                            actions=actions,
                            prev_logprobs=result["prev_logprobs"]
                            if self.collect_prev_infos
                            else None,
                            prev_values=result["prev_values"]
                            if self.collect_prev_infos
                            else None,
                            bootstrap_values=self.get_bootstrap_values(
                                merged_kv.get("final_obs")
                            ),
                            save_flags=save_flags,
                            forward_inputs=result["forward_inputs"],
                            versions=torch.full(
                                (actions.shape[0], 1),
                                float(self.version),
                                dtype=torch.float32,
                                device=version_ref.device if version_ref is not None else actions.device,
                            ),
                        )
                        self._send_rollout_result_micro(
                            output_channel,
                            rollout_result,
                            next_micro_to_process,
                            mode="train",
                        )
                        next_micro_to_process += 1

                stage_t1 = time.time()
                _append_worker_timeline_event(
                    self,
                    component="rollout",
                    tag=f"predict_e{rollout_epoch_idx}_cs{chunk_step}_st{stage_idx}",
                    t0=stage_t0,
                    t1=stage_t1,
                    extra={
                        "chunk_step": chunk_step,
                        "stage_idx": stage_idx,
                        "role": "ae",
                    },
                )
                if VLM_AE_DEBUG:
                    _get_logger().debug(
                        "[VLM-AE] AE rank=%s chunk=%s stage=%s recv_ms=%.1f process_ms=%.1f total_ms=%.1f",
                        self._rank,
                        chunk_step,
                        stage_idx,
                        recv_total * 1000.0,
                        process_total * 1000.0,
                        (stage_t1 - stage_t0) * 1000.0,
                    )

        # Tail steps
        for stage_idx in range(self.num_pipeline_stages):
            stage_t0 = time.time()
            recv_total = 0.0
            process_total = 0.0

            micro_batch_kv = {i: [] for i in range(num_micro_batches)}
            received_counts = {i: 0 for i in range(num_micro_batches)}
            recv_completion_queue, total_expected = self._issue_async_kv_recvs(
                num_micro_batches, num_vlm_workers
            )
            received_total = 0
            next_micro_to_process = 0

            while received_total < total_expected:
                recv_result = recv_completion_queue.get()
                received_total += 1
                t_unpack0 = time.time()
                kv_payload = recv_result["payload"]
                kv_data = _unpack_kv_message(kv_payload)
                t_unpack1 = time.time()
                micro_id = kv_data.get("micro_batch_id", recv_result["micro_batch_id"])
                vlm_rank = kv_data.get("vlm_rank", recv_result["src_rank"])
                micro_batch_kv[micro_id].append(kv_data)
                received_counts[micro_id] += 1
                recv_total += t_unpack1 - recv_result["t0"]
                _append_worker_timeline_event(
                    self,
                    component="kv_wait",
                    tag=(
                        f"recv_tail_e{rollout_epoch_idx}_st{stage_idx}"
                        f"_mb{micro_id}_src{vlm_rank}"
                    ),
                    t0=recv_result["t0"],
                    t1=recv_result["t_ready"],
                    extra={
                        "chunk_step": -1,
                        "stage_idx": stage_idx,
                        "micro_batch_id": micro_id,
                        "src_rank": vlm_rank,
                        "dst_rank": self._rank,
                        "direction": "vlm_to_ae",
                    },
                )
                if t_unpack0 > recv_result["t_ready"]:
                    _append_worker_timeline_event(
                        self,
                        component="kv_queue",
                        tag=(
                            f"queue_tail_e{rollout_epoch_idx}_st{stage_idx}"
                            f"_mb{micro_id}_src{vlm_rank}"
                        ),
                        t0=recv_result["t_ready"],
                        t1=t_unpack0,
                        extra={
                            "chunk_step": -1,
                            "stage_idx": stage_idx,
                            "micro_batch_id": micro_id,
                            "src_rank": vlm_rank,
                            "dst_rank": self._rank,
                            "direction": "vlm_to_ae",
                        },
                    )
                _append_worker_timeline_event(
                    self,
                    component="kv_unpack",
                    tag=(
                        f"unpack_tail_e{rollout_epoch_idx}_st{stage_idx}"
                        f"_mb{micro_id}_src{vlm_rank}"
                    ),
                    t0=t_unpack0,
                    t1=t_unpack1,
                    extra={
                        "chunk_step": -1,
                        "stage_idx": stage_idx,
                        "micro_batch_id": micro_id,
                        "src_rank": vlm_rank,
                        "dst_rank": self._rank,
                        "direction": "vlm_to_ae",
                        },
                    )

                while (
                    next_micro_to_process < num_micro_batches
                    and received_counts[next_micro_to_process] == num_vlm_workers
                ):
                    merged_kv = self._merge_kv_from_vlm_workers(
                        micro_batch_kv[next_micro_to_process]
                    )
                    t_process_start = time.time()
                    actions, result = self.hf_model._predict_ae_stage(
                        kv_data=merged_kv,
                        env_obs=merged_kv.get("env_obs"),
                        mode="train",
                        compute_values=True,
                    )
                    t_process_end = time.time()
                    process_total += t_process_end - t_process_start
                    _append_ae_model_phase_events(
                        self,
                        stage_start_time=t_process_start,
                        rollout_epoch_idx=rollout_epoch_idx,
                        chunk_step=-1,
                        stage_idx=stage_idx,
                        micro_id=next_micro_to_process,
                    )
                    _append_worker_timeline_event(
                        self,
                        component="rollout_ae",
                        tag=f"denoise_tail_e{rollout_epoch_idx}_st{stage_idx}_mb{next_micro_to_process}",
                        t0=t_process_start,
                        t1=t_process_end,
                        extra={
                            "chunk_step": -1,
                            "stage_idx": stage_idx,
                            "micro_batch_id": next_micro_to_process,
                            "role": "ae",
                        },
                    )
                    rollout_result = RolloutResult(
                        actions=actions,
                        prev_values=result["prev_values"]
                        if self.collect_prev_infos
                        else None,
                        bootstrap_values=self.get_bootstrap_values(
                            merged_kv.get("final_obs")
                        ),
                    )
                    self._send_rollout_result_micro(
                        output_channel,
                        rollout_result,
                        next_micro_to_process,
                        mode="train",
                    )
                    next_micro_to_process += 1

            if next_micro_to_process > 0:
                stage_t1 = time.time()
                _append_worker_timeline_event(
                    self,
                    component="rollout",
                    tag=f"predict_tail_e{rollout_epoch_idx}_st{stage_idx}",
                    t0=stage_t0,
                    t1=stage_t1,
                    extra={
                        "chunk_step": -1,
                        "stage_idx": stage_idx,
                        "role": "ae",
                    },
                )
                if VLM_AE_DEBUG:
                    _get_logger().debug(
                        "[VLM-AE] AE rank=%s tail_stage=%s recv_ms=%.1f process_ms=%.1f total_ms=%.1f",
                        self._rank,
                        stage_idx,
                        recv_total * 1000.0,
                        process_total * 1000.0,
                        (stage_t1 - stage_t0) * 1000.0,
                    )

    def _issue_async_kv_recvs(self, num_micro_batches: int, num_vlm_workers: int) -> tuple["queue.Queue[dict[str, Any]]", int]:
        if not hasattr(self, "_kv_recv_executor"):
            self._kv_recv_executor = ThreadPoolExecutor(
                max_workers=max(4, num_micro_batches * num_vlm_workers)
            )
        completion_queue: "queue.Queue[dict[str, Any]]" = queue.Queue()
        total_expected = num_micro_batches * num_vlm_workers
        for micro_batch_id in range(num_micro_batches):
            for vlm_rank in range(num_vlm_workers):
                t0 = time.time()
                recv_work = self.recv(
                    src_group_name=self._group_name,
                    src_rank=vlm_rank,
                    async_op=True,
                )
                self._kv_recv_executor.submit(
                    _wait_for_kv_recv,
                    recv_work,
                    completion_queue,
                    src_rank=vlm_rank,
                    micro_batch_id=micro_batch_id,
                    t0=t0,
                )
        return completion_queue, total_expected

    def _merge_kv_from_vlm_workers(self, kv_data_list: list[dict]) -> dict:
        """Merge KV cache from multiple VLM workers along batch dimension."""
        def rebuild_env_obs(merged_data: list[dict], prefix: str) -> dict[str, Any] | None:
            def merge_tensor(key: str) -> torch.Tensor | None:
                tensors = [d.get(key) for d in merged_data if d.get(key) is not None]
                return torch.cat(tensors, dim=0) if tensors else None

            def merge_list(key: str) -> list[str] | None:
                values = [d.get(key) for d in merged_data if d.get(key) is not None]
                if not values:
                    return None
                merged_values: list[str] = []
                for value in values:
                    merged_values.extend(value)
                return merged_values

            env_obs = {
                "main_images": merge_tensor(f"{prefix}_main_images"),
                "wrist_images": merge_tensor(f"{prefix}_wrist_images"),
                "extra_view_images": merge_tensor(f"{prefix}_extra_view_images"),
                "states": merge_tensor(f"{prefix}_states"),
                "task_descriptions": merge_list(f"{prefix}_task_descriptions"),
            }
            has_tensor_payload = any(
                env_obs[key] is not None
                for key in ("main_images", "wrist_images", "extra_view_images", "states")
            )
            if not has_tensor_payload:
                return None
            return env_obs

        if len(kv_data_list) == 1:
            kv_data = _unpack_kv_message(kv_data_list[0])
            return {
                "kv_cache": kv_data["kv_cache"],
                "prefix_output": kv_data.get("prefix_output"),
                "prefix_pad_masks": kv_data.get("prefix_pad_masks"),
                "state": kv_data.get("state"),
                "vlm_value": kv_data.get("vlm_value"),
                "env_obs": rebuild_env_obs([kv_data], "obs"),
                "final_obs": rebuild_env_obs([kv_data], "final"),
            }

        # Sort by vlm_rank to ensure consistent ordering
        kv_data_list = [_unpack_kv_message(kv_data) for kv_data in kv_data_list]
        kv_data_list.sort(key=lambda x: x.get("vlm_rank", 0))

        # Merge KV along batch dimension (dim=0)
        merged_kv = []
        num_layers = len(kv_data_list[0]["kv_cache"])

        for layer_idx in range(num_layers):
            k_shards = [kv["kv_cache"][layer_idx][0] for kv in kv_data_list]
            v_shards = [kv["kv_cache"][layer_idx][1] for kv in kv_data_list]
            merged_kv.append((torch.cat(k_shards, dim=0), torch.cat(v_shards, dim=0)))

        def merge_tensor(key):
            tensors = [d.get(key) for d in kv_data_list if d.get(key) is not None]
            return torch.cat(tensors, dim=0) if tensors else None

        return {
            "kv_cache": merged_kv,
            "prefix_output": merge_tensor("prefix_output"),
            "prefix_pad_masks": merge_tensor("prefix_pad_masks"),
            "state": merge_tensor("state"),
            "vlm_value": merge_tensor("vlm_value"),
            "env_obs": rebuild_env_obs(kv_data_list, "obs"),
            "final_obs": rebuild_env_obs(kv_data_list, "final"),
        }

    def _merge_results(self, results: list[dict]) -> dict:
        """Merge results from multiple micro-batches."""
        merged = {}
        for key in results[0].keys():
            values = [r.get(key) for r in results]
            if all(v is not None for v in values):
                if isinstance(values[0], torch.Tensor):
                    merged[key] = torch.cat(values, dim=0)
                elif isinstance(values[0], dict):
                    merged[key] = {k: torch.cat([v[k] for v in values if v.get(k) is not None], dim=0)
                                   for k in values[0].keys()}
                else:
                    merged[key] = values[0]
            else:
                merged[key] = None
        return merged

    # Store original for fallback
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker
    MultiStepRolloutWorker.original_generate_one_epoch = MultiStepRolloutWorker.generate_one_epoch

    # Apply patches
    MultiStepRolloutWorker.init_worker = patched_init_worker
    MultiStepRolloutWorker.generate_one_epoch = patched_generate_one_epoch
    MultiStepRolloutWorker._wait_one_pending_vlm_send = _wait_one_pending_vlm_send
    MultiStepRolloutWorker._flush_pending_vlm_sends = _flush_pending_vlm_sends
    MultiStepRolloutWorker._dispatch_vlm_send = _dispatch_vlm_send
    MultiStepRolloutWorker._recv_env_micro_output = _recv_env_micro_output
    MultiStepRolloutWorker._recv_baseline_micro_env_output = _recv_baseline_micro_env_output
    MultiStepRolloutWorker._slice_rollout_result = _slice_rollout_result
    MultiStepRolloutWorker._send_rollout_result_micro = _send_rollout_result_micro
    MultiStepRolloutWorker._send_baseline_micro_rollout_result = _send_baseline_micro_rollout_result
    MultiStepRolloutWorker._unified_micro_generate_one_epoch = _unified_micro_generate_one_epoch
    MultiStepRolloutWorker._vlm_generate_one_epoch = _vlm_generate_one_epoch
    MultiStepRolloutWorker._ae_generate_one_epoch = _ae_generate_one_epoch
    MultiStepRolloutWorker._issue_async_kv_recvs = _issue_async_kv_recvs
    MultiStepRolloutWorker._merge_kv_from_vlm_workers = _merge_kv_from_vlm_workers
    MultiStepRolloutWorker._merge_results = _merge_results
    MultiStepRolloutWorker._vlm_ae_patched = True
    _get_logger().info("[VLM-AE Patch] Applied 2-stage async pipeline patches")


if os.environ.get("VLM_AE_DISAGG", "0") == "1" or os.environ.get(
    "VLM_AE_BASELINE_MICROPIPE", "0"
) == "1":
    patch_huggingface_worker_for_pipeline()
