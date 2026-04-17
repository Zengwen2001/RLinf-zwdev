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
VLM-AE Disaggregation Module

This module provides the infrastructure for separating VLM (Vision-Language Model)
and AE (Action Expert) inference across different GPUs, enabling pipeline parallelism.

Architecture:
    GPU 0,1,2: VLM Workers (compute-bound, batch parallel)
    GPU 3:     AE Worker (memory-bound, denoise loop)

Usage:
    Set environment variable VLM_AE_DISAGG=1 to enable.

    export VLM_AE_DISAGG=1
    export VLM_AE_VLM_GPUS=0,1,2      # VLM worker GPUs (default: 0,1,2)
    export VLM_AE_AE_GPU=3             # AE worker GPU (default: 3)
    export VLM_AE_NUM_MICRO_BATCHES=2  # Number of micro-batches (default: 2)

Transfer Methods:
    - nccl: NCCL P2P for single-node (recommended for single-node)
    - direct: Direct GPU copy using CUDA IPC
    - rdma: RDMA via Mooncake/NIXL (future, for multi-node)
"""

import os
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.distributed as dist

from rlinf.utils.comm_mapping import CommMapper

# Configuration via environment variables
VLM_AE_DISAGG_ENABLED = os.environ.get("VLM_AE_DISAGG", "0") == "1"
VLM_AE_VLM_GPUS = [int(x) for x in os.environ.get("VLM_AE_VLM_GPUS", "0,1,2").split(",")]
VLM_AE_AE_GPU = int(os.environ.get("VLM_AE_AE_GPU", "3"))
VLM_AE_NUM_MICRO_BATCHES = int(os.environ.get("VLM_AE_NUM_MICRO_BATCHES", "2"))
VLM_AE_TRANSFER_BACKEND = os.environ.get("VLM_AE_TRANSFER_BACKEND", "nccl")  # nccl, direct, rdma
VLM_AE_DEBUG = os.environ.get("VLM_AE_DEBUG", "0") == "1"


class WorkerRole(Enum):
    """Role of the worker in disaggregated setup."""
    VLM = "vlm"  # Vision-Language Model worker
    AE = "ae"    # Action Expert worker
    UNIFIED = "unified"  # Default unified mode


class TransferStatus(Enum):
    """Status of KV transfer."""
    PENDING = "pending"
    TRANSFERRING = "transferring"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class KVTransferData:
    """Data structure for KV cache transfer between VLM and AE workers."""
    request_id: int
    kv_cache: Optional[list] = None  # List of (K, V) tuples per layer
    prefix_output: Optional[torch.Tensor] = None
    prefix_pad_masks: Optional[torch.Tensor] = None
    state: Optional[torch.Tensor] = None
    noise: Optional[torch.Tensor] = None
    metadata: dict = field(default_factory=dict)
    status: TransferStatus = TransferStatus.PENDING

    # For backward transfer (AE -> VLM, if needed)
    actions: Optional[torch.Tensor] = None


@dataclass
class PipelineBatch:
    """A batch in the pipeline."""
    batch_id: int
    micro_batch_id: int
    env_obs: dict
    kv_data: Optional[KVTransferData] = None
    actions: Optional[torch.Tensor] = None
    result: Optional[dict] = None


@dataclass(frozen=True)
class MicroSliceSpec:
    """One contiguous micro-batch slice transferred between two ranks."""

    peer_rank: int
    micro_batch_id: int
    src_start: int
    src_end: int
    dst_start: int
    dst_end: int
    order: int

    @property
    def size(self) -> int:
        return self.src_end - self.src_start


def _intersect_intervals(
    start0: int, end0: int, start1: int, end1: int
) -> tuple[int, int] | None:
    overlap_start = max(start0, start1)
    overlap_end = min(end0, end1)
    if overlap_start >= overlap_end:
        return None
    return overlap_start, overlap_end


def _validate_micro_layout(batch_size: int, num_micro_batches: int) -> tuple[int, int]:
    num_vlm_workers = get_num_vlm_workers()
    assert batch_size % num_vlm_workers == 0, (
        f"batch_size={batch_size} must be divisible by num_vlm_workers={num_vlm_workers}."
    )
    vlm_local_batch_size = batch_size // num_vlm_workers
    assert vlm_local_batch_size % num_micro_batches == 0, (
        f"vlm_local_batch_size={vlm_local_batch_size} must be divisible by "
        f"num_micro_batches={num_micro_batches}."
    )
    return vlm_local_batch_size, vlm_local_batch_size // num_micro_batches


def get_vlm_env_micro_src_slices(
    batch_size: int,
    env_world_size: int,
    rollout_rank: int,
    num_micro_batches: int,
) -> dict[int, list[MicroSliceSpec]]:
    """Expected env->VLM micro-batch slices for one VLM rollout rank."""
    vlm_worker_index = get_vlm_worker_index(rollout_rank)
    if vlm_worker_index is None:
        return {}

    env_local_batch_size = batch_size // env_world_size
    vlm_local_batch_size, vlm_micro_batch_size = _validate_micro_layout(
        batch_size, num_micro_batches
    )
    global_vlm_start = vlm_worker_index * vlm_local_batch_size

    slices: dict[int, list[MicroSliceSpec]] = defaultdict(list)
    for micro_batch_id in range(num_micro_batches):
        micro_global_start = global_vlm_start + micro_batch_id * vlm_micro_batch_size
        micro_global_end = micro_global_start + vlm_micro_batch_size
        for env_rank in range(env_world_size):
            env_global_start = env_rank * env_local_batch_size
            env_global_end = env_global_start + env_local_batch_size
            overlap = _intersect_intervals(
                micro_global_start, micro_global_end, env_global_start, env_global_end
            )
            if overlap is None:
                continue
            overlap_start, overlap_end = overlap
            slices[micro_batch_id].append(
                MicroSliceSpec(
                    peer_rank=env_rank,
                    micro_batch_id=micro_batch_id,
                    src_start=overlap_start - env_global_start,
                    src_end=overlap_end - env_global_start,
                    dst_start=overlap_start - micro_global_start,
                    dst_end=overlap_end - micro_global_start,
                    order=0,
                )
            )
        slices[micro_batch_id].sort(key=lambda item: item.dst_start)
        slices[micro_batch_id] = [
            MicroSliceSpec(
                peer_rank=item.peer_rank,
                micro_batch_id=item.micro_batch_id,
                src_start=item.src_start,
                src_end=item.src_end,
                dst_start=item.dst_start,
                dst_end=item.dst_end,
                order=order,
            )
            for order, item in enumerate(slices[micro_batch_id])
        ]
    return dict(slices)


def get_env_to_vlm_micro_dst_slices(
    batch_size: int,
    env_world_size: int,
    env_rank: int,
    num_micro_batches: int,
) -> dict[int, list[MicroSliceSpec]]:
    """Env->VLM micro-batch slices emitted by one env rank."""
    filtered: dict[int, list[MicroSliceSpec]] = {}
    for vlm_rank in range(get_num_vlm_workers()):
        recv_specs = get_vlm_env_micro_src_slices(
            batch_size, env_world_size, vlm_rank, num_micro_batches
        )
        for micro_batch_id, specs in recv_specs.items():
            env_specs = [
                MicroSliceSpec(
                    peer_rank=vlm_rank,
                    micro_batch_id=spec.micro_batch_id,
                    src_start=spec.src_start,
                    src_end=spec.src_end,
                    dst_start=spec.dst_start,
                    dst_end=spec.dst_end,
                    order=spec.order,
                )
                for spec in specs
                if spec.peer_rank == env_rank
            ]
            if env_specs:
                filtered.setdefault(micro_batch_id, []).extend(env_specs)
    return {
        micro_id: sorted(items, key=lambda item: (item.peer_rank, item.order))
        for micro_id, items in filtered.items()
    }


def _get_ae_micro_segments(
    batch_size: int, num_micro_batches: int
) -> dict[int, list[tuple[int, int, int, int]]]:
    """Return per-micro segments as (global_start, global_end, ae_start, ae_end)."""
    vlm_local_batch_size, vlm_micro_batch_size = _validate_micro_layout(
        batch_size, num_micro_batches
    )
    segments: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    for micro_batch_id in range(num_micro_batches):
        ae_offset = 0
        for vlm_rank in range(get_num_vlm_workers()):
            global_start = vlm_rank * vlm_local_batch_size + micro_batch_id * vlm_micro_batch_size
            global_end = global_start + vlm_micro_batch_size
            segments[micro_batch_id].append(
                (global_start, global_end, ae_offset, ae_offset + vlm_micro_batch_size)
            )
            ae_offset += vlm_micro_batch_size
    return dict(segments)


def get_ae_env_micro_src_slices(
    batch_size: int,
    env_world_size: int,
    num_micro_batches: int,
) -> dict[int, list[MicroSliceSpec]]:
    """Expected env->AE micro-batch slices for the AE rollout rank."""
    env_local_batch_size = batch_size // env_world_size
    slices: dict[int, list[MicroSliceSpec]] = defaultdict(list)
    segments = _get_ae_micro_segments(batch_size, num_micro_batches)

    for micro_batch_id, micro_segments in segments.items():
        order = 0
        for global_start, global_end, ae_start, _ in micro_segments:
            for env_rank in range(env_world_size):
                env_global_start = env_rank * env_local_batch_size
                env_global_end = env_global_start + env_local_batch_size
                overlap = _intersect_intervals(
                    global_start, global_end, env_global_start, env_global_end
                )
                if overlap is None:
                    continue
                overlap_start, overlap_end = overlap
                slices[micro_batch_id].append(
                    MicroSliceSpec(
                        peer_rank=env_rank,
                        micro_batch_id=micro_batch_id,
                        src_start=overlap_start - env_global_start,
                        src_end=overlap_end - env_global_start,
                        dst_start=ae_start + (overlap_start - global_start),
                        dst_end=ae_start + (overlap_end - global_start),
                        order=order,
                    )
                )
                order += 1
        slices[micro_batch_id].sort(key=lambda item: item.dst_start)
    return dict(slices)


def get_env_to_ae_micro_dst_slices(
    batch_size: int,
    env_world_size: int,
    env_rank: int,
    num_micro_batches: int,
) -> dict[int, list[MicroSliceSpec]]:
    """Env->AE micro-batch slices emitted by one env rank."""
    all_slices = get_ae_env_micro_src_slices(batch_size, env_world_size, num_micro_batches)
    ae_rank = get_ae_worker_rank()
    filtered: dict[int, list[MicroSliceSpec]] = {}
    for micro_batch_id, specs in all_slices.items():
        micro_specs = [
            MicroSliceSpec(
                peer_rank=ae_rank,
                micro_batch_id=spec.micro_batch_id,
                src_start=spec.src_start,
                src_end=spec.src_end,
                dst_start=spec.dst_start,
                dst_end=spec.dst_end,
                order=spec.order,
            )
            for spec in specs
            if spec.peer_rank == env_rank
        ]
        if micro_specs:
            filtered[micro_batch_id] = micro_specs
    return filtered


def get_ae_to_env_micro_dst_slices(
    batch_size: int,
    env_world_size: int,
    num_micro_batches: int,
) -> dict[int, list[MicroSliceSpec]]:
    """AE->env micro-batch slices emitted by the AE rollout rank."""
    env_local_batch_size = batch_size // env_world_size
    slices: dict[int, list[MicroSliceSpec]] = defaultdict(list)
    segments = _get_ae_micro_segments(batch_size, num_micro_batches)

    for micro_batch_id, micro_segments in segments.items():
        order = 0
        for global_start, global_end, ae_start, _ in micro_segments:
            for env_rank in range(env_world_size):
                env_global_start = env_rank * env_local_batch_size
                env_global_end = env_global_start + env_local_batch_size
                overlap = _intersect_intervals(
                    global_start, global_end, env_global_start, env_global_end
                )
                if overlap is None:
                    continue
                overlap_start, overlap_end = overlap
                slices[micro_batch_id].append(
                    MicroSliceSpec(
                        peer_rank=env_rank,
                        micro_batch_id=micro_batch_id,
                        src_start=ae_start + (overlap_start - global_start),
                        src_end=ae_start + (overlap_end - global_start),
                        dst_start=overlap_start - env_global_start,
                        dst_end=overlap_end - env_global_start,
                        order=order,
                    )
                )
                order += 1
        slices[micro_batch_id].sort(key=lambda item: item.dst_start)
    return dict(slices)


def get_env_from_ae_micro_src_slices(
    batch_size: int,
    env_world_size: int,
    env_rank: int,
    num_micro_batches: int,
) -> dict[int, list[MicroSliceSpec]]:
    """Expected AE->env micro-batch slices for one env rank."""
    all_slices = get_ae_to_env_micro_dst_slices(
        batch_size, env_world_size, num_micro_batches
    )
    ae_rank = get_ae_worker_rank()
    filtered: dict[int, list[MicroSliceSpec]] = {}
    for micro_batch_id, specs in all_slices.items():
        micro_specs = [
            MicroSliceSpec(
                peer_rank=ae_rank,
                micro_batch_id=spec.micro_batch_id,
                src_start=spec.src_start,
                src_end=spec.src_end,
                dst_start=spec.dst_start,
                dst_end=spec.dst_end,
                order=spec.order,
            )
            for spec in specs
            if spec.peer_rank == env_rank
        ]
        if micro_specs:
            filtered[micro_batch_id] = micro_specs
    return filtered


class KVTransferManager:
    """
    Manages KV cache transfer between VLM and AE workers.

    Supports multiple backends:
    - nccl: NCCL P2P communication (single node)
    - direct: Direct GPU copy using CUDA IPC
    - rdma: RDMA via Mooncake/NIXL (multi-node, future)

    For single-node case, we use NCCL P2P or direct GPU copy:
    1. VLM workers compute KV cache on their GPUs
    2. KV cache is transferred directly to AE GPU using P2P
    3. AE worker receives and merges KV cache from all VLM ranks
    """

    def __init__(
        self,
        role: WorkerRole,
        vlm_gpus: list[int] = None,
        ae_gpu: int = None,
        backend: str = "nccl",
    ):
        self.role = role
        self.vlm_gpus = vlm_gpus or VLM_AE_VLM_GPUS
        self.ae_gpu = ae_gpu if ae_gpu is not None else VLM_AE_AE_GPU
        self.backend = backend

        # Transfer buffers (in-memory for single-node)
        self._send_buffers: dict[int, KVTransferData] = {}
        self._recv_buffers: dict[int, KVTransferData] = {}
        self._status: dict[int, TransferStatus] = {}

        # For async transfer
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._lock = asyncio.Lock()

        # Initialize transfer backend
        self._transfer_impl = self._init_transfer_backend()

        if VLM_AE_DEBUG:
            print(f"[KVTransferManager] Initialized with role={role}, "
                  f"vlm_gpus={self.vlm_gpus}, ae_gpu={self.ae_gpu}, backend={backend}")

    def _init_transfer_backend(self):
        """Initialize the transfer backend implementation."""
        if self.backend == "nccl" or self.backend == "direct":
            from rlinf.models.embodiment.openpi.nccl_kv_transfer import DirectGpuTransfer
            return DirectGpuTransfer(
                role=self.role.value,
                vlm_gpus=self.vlm_gpus,
                ae_gpu=self.ae_gpu,
            )
        elif self.backend == "rdma":
            # Future: Mooncake/NIXL integration
            raise NotImplementedError("RDMA backend not yet implemented")
        else:
            raise ValueError(f"Unknown transfer backend: {self.backend}")

    async def send_kv(self, data: KVTransferData) -> asyncio.Future:
        """
        Send KV cache from VLM worker to AE worker.

        Args:
            data: KV transfer data containing cache and metadata

        Returns:
            Future that completes when transfer is done
        """
        if self.role != WorkerRole.VLM:
            raise RuntimeError("Only VLM worker can send KV cache")

        self._send_buffers[data.request_id] = data
        data.status = TransferStatus.TRANSFERRING

        if VLM_AE_DEBUG:
            print(f"[KVTransferManager] VLM sending KV cache for request {data.request_id}")

        # Use direct GPU transfer
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(
            self._executor,
            self._send_kv_impl,
            data
        )

    def _send_kv_impl(self, data: KVTransferData):
        """Implementation of KV cache send using direct GPU transfer."""
        try:
            if self._transfer_impl is not None:
                # Transfer KV cache to AE GPU
                if data.kv_cache:
                    data.kv_cache = self._transfer_impl.transfer_kv_cache(
                        data.kv_cache, self.ae_gpu
                    )

                # Transfer other tensors
                if data.prefix_output is not None:
                    data.prefix_output = self._transfer_impl.transfer_tensor(
                        data.prefix_output, self.ae_gpu
                    )
                if data.prefix_pad_masks is not None:
                    data.prefix_pad_masks = self._transfer_impl.transfer_tensor(
                        data.prefix_pad_masks, self.ae_gpu
                    )
                if data.state is not None:
                    data.state = self._transfer_impl.transfer_tensor(
                        data.state, self.ae_gpu
                    )

            data.status = TransferStatus.COMPLETED
            if VLM_AE_DEBUG:
                print(f"[KVTransferManager] VLM completed transfer for request {data.request_id}")

        except Exception as e:
            data.status = TransferStatus.FAILED
            print(f"[KVTransferManager] Transfer failed for request {data.request_id}: {e}")
            raise

    async def recv_kv(self, request_id: int, timeout: float = 30.0) -> KVTransferData:
        """
        Receive KV cache at AE worker from VLM worker.

        Args:
            request_id: Request ID to receive
            timeout: Timeout in seconds

        Returns:
            KV transfer data
        """
        if self.role != WorkerRole.AE:
            raise RuntimeError("Only AE worker can receive KV cache")

        if VLM_AE_DEBUG:
            print(f"[KVTransferManager] AE receiving KV cache for request {request_id}")

        # For direct GPU transfer, data is already on AE GPU
        # We just need to wait for all VLM ranks to complete
        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout:
            if request_id in self._recv_buffers:
                data = self._recv_buffers[request_id]
                if data.status == TransferStatus.COMPLETED:
                    return data
                elif data.status == TransferStatus.FAILED:
                    raise RuntimeError(f"Transfer failed for request {request_id}")

            await asyncio.sleep(0.001)

        raise TimeoutError(f"Timeout waiting for KV transfer for request {request_id}")

    def store_for_transfer(self, data: KVTransferData):
        """Store data for transfer (called by VLM worker)."""
        self._send_buffers[data.request_id] = data

    def retrieve_for_compute(self, request_id: int) -> KVTransferData:
        """Retrieve data for compute (called by AE worker)."""
        return self._recv_buffers.get(request_id, KVTransferData(request_id=request_id))

    def complete_transfer(self, request_id: int, data: KVTransferData):
        """Mark transfer as complete and store for AE retrieval."""
        data.status = TransferStatus.COMPLETED
        self._recv_buffers[request_id] = data

    def close(self):
        """Clean up resources."""
        self._executor.shutdown(wait=False)
        self._send_buffers.clear()
        self._recv_buffers.clear()


class VLMAEPipelineCoordinator:
    """
    Coordinates the pipeline execution between VLM and AE workers.

    This class manages:
    1. Micro-batch scheduling
    2. KV cache transfer coordination
    3. Pipeline stage synchronization
    """

    def __init__(
        self,
        num_micro_batches: int = None,
        batch_size: int = 24,
    ):
        self.num_micro_batches = num_micro_batches or VLM_AE_NUM_MICRO_BATCHES
        self.batch_size = batch_size
        self.micro_batch_size = batch_size // self.num_micro_batches

        # Pipeline queues
        self._vlm_queue: asyncio.Queue = None
        self._ae_queue: asyncio.Queue = None
        self._env_queue: asyncio.Queue = None

        # Request ID counter
        self._request_id_counter = 0

        if VLM_AE_DEBUG:
            print(f"[VLMAEPipelineCoordinator] Initialized with "
                  f"num_micro_batches={self.num_micro_batches}, "
                  f"batch_size={self.batch_size}, "
                  f"micro_batch_size={self.micro_batch_size}")

    def get_next_request_id(self) -> int:
        """Get next unique request ID."""
        self._request_id_counter += 1
        return self._request_id_counter

    def init_queues(self):
        """Initialize pipeline queues."""
        self._vlm_queue = asyncio.Queue(maxsize=self.num_micro_batches * 2)
        self._ae_queue = asyncio.Queue(maxsize=self.num_micro_batches * 2)
        self._env_queue = asyncio.Queue(maxsize=self.num_micro_batches * 2)

    def split_obs_for_micro_batch(
        self,
        env_obs: dict,
        micro_batch_id: int
    ) -> dict:
        """Split observations for a specific micro-batch."""
        start = micro_batch_id * self.micro_batch_size
        end = start + self.micro_batch_size

        result = {}
        for key, value in env_obs.items():
            if isinstance(value, torch.Tensor):
                result[key] = value[start:end]
            elif isinstance(value, list):
                result[key] = value[start:end]
            else:
                result[key] = value

        return result

    def merge_results_from_micro_batches(
        self,
        results: list[tuple[torch.Tensor, dict]]
    ) -> tuple[torch.Tensor, dict]:
        """Merge results from multiple micro-batches."""
        actions = torch.cat([r[0] for r in results], dim=0)

        # Merge result dicts
        merged_result = {}
        for key in results[0][1].keys():
            values = [r[1][key] for r in results]
            if all(v is not None for v in values):
                if isinstance(values[0], torch.Tensor):
                    merged_result[key] = torch.cat(values, dim=0)
                elif isinstance(values[0], dict):
                    merged_result[key] = values[0]  # Use first, they should be the same
                else:
                    merged_result[key] = values[0]
            else:
                merged_result[key] = None

        return actions, merged_result


def get_worker_role() -> WorkerRole:
    """
    Determine the role of current worker based on rank assignment.

    In Ray worker architecture, each worker sees only one GPU via CUDA_VISIBLE_DEVICES,
    so we use the worker's rank to determine its role.

    Returns:
        WorkerRole: VLM, AE, or UNIFIED
    """
    if not VLM_AE_DISAGG_ENABLED:
        return WorkerRole.UNIFIED

    # Get worker rank from environment (Ray sets this)
    # RANK is set by Ray, LOCAL_RANK is also available
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", 4))

    if VLM_AE_DEBUG:
        print(f"[get_worker_role] rank={rank}, world_size={world_size}, "
              f"vlm_gpus={VLM_AE_VLM_GPUS}, ae_gpu={VLM_AE_AE_GPU}")

    # Map rank to role:
    # - Ranks 0, 1, 2 are VLM workers (corresponding to VLM_GPUS)
    # - Rank 3 is AE worker (corresponding to AE_GPU)
    num_vlm_workers = len(VLM_AE_VLM_GPUS)

    if rank < num_vlm_workers:
        return WorkerRole.VLM
    elif rank == num_vlm_workers:
        return WorkerRole.AE
    else:
        # Additional ranks fallback to VLM or UNIFIED
        # For 4-GPU setup with 3 VLM + 1 AE, ranks >= 4 would be UNIFIED
        return WorkerRole.UNIFIED


def is_disagg_enabled() -> bool:
    """Check if disaggregated mode is enabled."""
    return VLM_AE_DISAGG_ENABLED


def get_num_vlm_workers() -> int:
    """Return the number of rollout ranks reserved for VLM workers."""
    return len(VLM_AE_VLM_GPUS)


def get_ae_worker_rank() -> int:
    """Return the rollout rank reserved for the AE worker."""
    return get_num_vlm_workers()


def get_vlm_worker_index(rank: int | None = None) -> int | None:
    """Map a rollout rank to its VLM-worker index."""
    if rank is None:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    return rank if 0 <= rank < get_num_vlm_workers() else None


def get_env_to_vlm_dst_ranks(
    batch_size: int,
    env_world_size: int,
    env_rank: int,
) -> list[tuple[int, int]]:
    """Map one env rank to the VLM rollout ranks that should receive obs shards."""
    return CommMapper.get_dst_ranks(
        batch_size=batch_size,
        src_world_size=env_world_size,
        dst_world_size=get_num_vlm_workers(),
        src_rank=env_rank,
    )


def get_vlm_env_src_ranks(
    batch_size: int,
    env_world_size: int,
    rollout_rank: int,
) -> list[tuple[int, int]]:
    """Map one VLM rollout rank to the env ranks that provide its obs shard."""
    vlm_worker_index = get_vlm_worker_index(rollout_rank)
    if vlm_worker_index is None:
        return []
    return CommMapper.get_src_ranks(
        batch_size=batch_size,
        src_world_size=env_world_size,
        dst_world_size=get_num_vlm_workers(),
        dst_rank=vlm_worker_index,
    )


def get_env_to_ae_dst_ranks(
    batch_size: int,
    env_world_size: int,
    env_rank: int,
) -> list[tuple[int, int]]:
    """Map one env rank to the AE rollout rank for metadata/full-stage obs."""
    dst_ranks = CommMapper.get_dst_ranks(
        batch_size=batch_size,
        src_world_size=env_world_size,
        dst_world_size=1,
        src_rank=env_rank,
    )
    ae_rank = get_ae_worker_rank()
    return [(ae_rank, size) for _, size in dst_ranks]


def get_ae_env_src_ranks(
    batch_size: int,
    env_world_size: int,
) -> list[tuple[int, int]]:
    """Map the AE rollout rank to all env ranks contributing to the full batch."""
    return CommMapper.get_src_ranks(
        batch_size=batch_size,
        src_world_size=env_world_size,
        dst_world_size=1,
        dst_rank=0,
    )


def get_ae_to_env_dst_ranks(
    batch_size: int,
    env_world_size: int,
) -> list[tuple[int, int]]:
    """Map the AE rollout rank to env ranks for rollout-result fan-out."""
    return CommMapper.get_dst_ranks(
        batch_size=batch_size,
        src_world_size=1,
        dst_world_size=env_world_size,
        src_rank=0,
    )


def get_env_from_ae_src_ranks(
    batch_size: int,
    env_world_size: int,
    env_rank: int,
) -> list[tuple[int, int]]:
    """Map one env rank to the AE rollout rank shards it should receive."""
    src_ranks = CommMapper.get_src_ranks(
        batch_size=batch_size,
        src_world_size=1,
        dst_world_size=env_world_size,
        dst_rank=env_rank,
    )
    ae_rank = get_ae_worker_rank()
    return [(ae_rank, size) for _, size in src_ranks]


# Global instances (initialized lazily)
_global_kv_manager: Optional[KVTransferManager] = None
_global_coordinator: Optional[VLMAEPipelineCoordinator] = None


def get_kv_manager() -> KVTransferManager:
    """Get or create global KV transfer manager."""
    global _global_kv_manager
    if _global_kv_manager is None:
        _global_kv_manager = KVTransferManager(
            role=get_worker_role(),
            backend=VLM_AE_TRANSFER_BACKEND,
        )
    return _global_kv_manager


def get_coordinator(batch_size: int = 24) -> VLMAEPipelineCoordinator:
    """Get or create global pipeline coordinator."""
    global _global_coordinator
    if _global_coordinator is None:
        _global_coordinator = VLMAEPipelineCoordinator(batch_size=batch_size)
    return _global_coordinator
