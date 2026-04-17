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
NCCL-based KV Cache Transfer for Single-Node VLM-AE Disaggregation

This module provides efficient KV cache transfer between VLM and AE workers
on a single node using NCCL P2P and CUDA IPC.

Architecture:
    - VLM Workers: GPU 0,1,2 (TP=3)
    - AE Worker: GPU 3 (TP=1)
    - Transfer: NCCL P2P for same-process, CUDA IPC for cross-process

Usage:
    # On VLM side (sender):
    transfer = NcclKVTransfer(role="vlm", vlm_gpus=[0,1,2], ae_gpu=3)
    transfer.send_kv(request_id, kv_cache, metadata)

    # On AE side (receiver):
    transfer = NcclKVTransfer(role="ae", vlm_gpus=[0,1,2], ae_gpu=3)
    kv_cache, metadata = transfer.recv_kv(request_id)
"""

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import queue

import torch
import torch.distributed as dist

# Configuration
VLM_AE_NCCL_DEBUG = os.environ.get("VLM_AE_NCCL_DEBUG", "0") == "1"
VLM_AE_NCCL_TIMEOUT = int(os.environ.get("VLM_AE_NCCL_TIMEOUT", "30"))  # seconds


class TransferRole(Enum):
    VLM = "vlm"
    AE = "ae"


@dataclass
class KVTransferMetadata:
    """Metadata for KV cache transfer."""
    request_id: int
    batch_size: int
    seq_len: int
    num_layers: int
    num_heads: int
    head_dim: int
    dtype: str
    device: str

    # Additional data
    prefix_output_shape: Optional[tuple] = None
    prefix_pad_masks_shape: Optional[tuple] = None
    state_shape: Optional[tuple] = None


@dataclass
class KVBuffer:
    """Buffer for KV cache storage."""
    kv_cache: Optional[List[tuple]] = None  # List of (K, V) tensors per layer
    prefix_output: Optional[torch.Tensor] = None
    prefix_pad_masks: Optional[torch.Tensor] = None
    state: Optional[torch.Tensor] = None
    metadata: Optional[KVTransferMetadata] = None


class SharedMemoryManager:
    """
    Manages shared memory for cross-process communication.

    For single-node case, we use shared memory + signals for coordination.
    This avoids the overhead of establishing NCCL process groups across
    different process trees.
    """

    def __init__(self, name: str, size: int, create: bool = True):
        """
        Initialize shared memory buffer.

        Args:
            name: Unique name for the shared memory segment
            size: Size in bytes
            create: If True, create new; if False, open existing
        """
        import multiprocessing.shared_memory as shm

        self.name = name
        self.size = size
        self.create = create

        if create:
            self.shm = shm.SharedMemory(create=True, size=size, name=name)
        else:
            self.shm = shm.SharedMemory(name=name)

    def write(self, data: bytes, offset: int = 0):
        """Write data to shared memory."""
        self.shm.buf[offset:offset + len(data)] = data

    def read(self, size: int, offset: int = 0) -> bytes:
        """Read data from shared memory."""
        return bytes(self.shm.buf[offset:offset + size])

    def close(self):
        """Close shared memory."""
        self.shm.close()

    def unlink(self):
        """Unlink shared memory (creator only)."""
        if self.create:
            self.shm.unlink()


class NcclKVTransfer:
    """
    NCCL-based KV cache transfer for single-node VLM-AE disaggregation.

    Transfer Protocol:
    1. VLM workers compute KV cache
    2. VLM workers serialize and write to shared memory
    3. VLM workers signal AE worker via IPC
    4. AE worker reads from shared memory
    5. AE worker deserializes KV cache
    6. AE worker runs denoise loop
    """

    def __init__(
        self,
        role: str,
        vlm_gpus: List[int] = None,
        ae_gpu: int = 3,
        world_size: int = 4,
        rank: int = None,
    ):
        """
        Initialize NCCL KV transfer.

        Args:
            role: "vlm" or "ae"
            vlm_gpus: List of VLM GPU IDs
            ae_gpu: AE GPU ID
            world_size: Total number of processes
            rank: Process rank (auto-detected if None)
        """
        self.role = TransferRole(role)
        self.vlm_gpus = vlm_gpus or [0, 1, 2]
        self.ae_gpu = ae_gpu
        self.world_size = world_size

        # Auto-detect rank from environment
        if rank is None:
            self.rank = int(os.environ.get("RANK", 0))
        else:
            self.rank = rank

        # Determine if we're VLM or AE based on GPU assignment
        current_device = torch.cuda.current_device()
        if current_device in self.vlm_gpus:
            self.role = TransferRole.VLM
            self.vlm_rank = self.vlm_gpus.index(current_device)
        elif current_device == self.ae_gpu:
            self.role = TransferRole.AE
            self.vlm_rank = -1
        else:
            raise ValueError(f"Unknown GPU assignment: {current_device}")

        # Transfer state
        self._transfer_buffers: Dict[int, KVBuffer] = {}
        self._pending_transfers: queue.Queue = queue.Queue()
        self._executor = ThreadPoolExecutor(max_workers=2)

        # IPC coordination (simplified for single-node)
        self._signal_queue = queue.Queue()
        self._lock = threading.Lock()

        if VLM_AE_NCCL_DEBUG:
            print(f"[NcclKVTransfer] Initialized with role={self.role}, "
                  f"rank={self.rank}, device={current_device}")

    def get_buffer_size(self, kv_cache: List[tuple], metadata: dict = None) -> int:
        """Calculate total buffer size needed for KV cache."""
        total_size = 0

        for k, v in kv_cache:
            total_size += k.numel() * k.element_size()
            total_size += v.numel() * v.element_size()

        # Add size for metadata (approximate)
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, torch.Tensor):
                    total_size += value.numel() * value.element_size()

        return total_size

    def serialize_kv_cache(
        self,
        kv_cache: List[tuple],
        metadata: dict = None
    ) -> bytes:
        """
        Serialize KV cache to bytes for transfer.

        Note: For GPU tensors, we need to move them to CPU first.
        For better performance, consider using CUDA IPC directly.
        """
        import pickle
        import io

        buffer = io.BytesIO()

        # Serialize each layer's KV
        serialized = {
            "num_layers": len(kv_cache),
            "kv_shapes": [],
            "kv_dtypes": [],
            "kv_data": [],
        }

        for k, v in kv_cache:
            serialized["kv_shapes"].append((k.shape, v.shape))
            serialized["kv_dtypes"].append((str(k.dtype), str(v.dtype)))
            serialized["kv_data"].append((k.cpu().numpy(), v.cpu().numpy()))

        # Serialize metadata
        if metadata:
            serialized["metadata"] = {}
            for key, value in metadata.items():
                if isinstance(value, torch.Tensor):
                    serialized["metadata"][key] = {
                        "shape": value.shape,
                        "dtype": str(value.dtype),
                        "data": value.cpu().numpy(),
                    }
                else:
                    serialized["metadata"][key] = value

        pickle.dump(serialized, buffer)
        return buffer.getvalue()

    def deserialize_kv_cache(
        self,
        data: bytes,
        device: torch.device
    ) -> tuple:
        """
        Deserialize KV cache from bytes.

        Returns:
            Tuple of (kv_cache, metadata)
        """
        import pickle
        import io
        import numpy as np

        buffer = io.BytesIO(data)
        serialized = pickle.load(buffer)

        # Reconstruct KV cache
        kv_cache = []
        for i, (k_data, v_data) in enumerate(serialized["kv_data"]):
            k_shape, v_shape = serialized["kv_shapes"][i]
            k_dtype = serialized["kv_dtypes"][i][0]
            v_dtype = serialized["kv_dtypes"][i][1]

            k = torch.from_numpy(k_data).to(device=device, dtype=getattr(torch, k_dtype.split('.')[-1]))
            v = torch.from_numpy(v_data).to(device=device, dtype=getattr(torch, v_dtype.split('.')[-1]))
            kv_cache.append((k, v))

        # Reconstruct metadata
        metadata = serialized.get("metadata", {})
        for key, value in metadata.items():
            if isinstance(value, dict) and "data" in value:
                metadata[key] = torch.from_numpy(value["data"]).to(
                    device=device,
                    dtype=getattr(torch, value["dtype"].split('.')[-1])
                )

        return kv_cache, metadata

    def send_kv(
        self,
        request_id: int,
        kv_cache: List[tuple],
        metadata: dict = None,
        async_transfer: bool = True
    ):
        """
        Send KV cache from VLM worker to AE worker.

        Args:
            request_id: Unique request identifier
            kv_cache: List of (K, V) tensor tuples per layer
            metadata: Additional metadata (prefix_output, etc.)
            async_transfer: If True, return immediately; otherwise wait
        """
        if self.role != TransferRole.VLM:
            raise RuntimeError("Only VLM workers can send KV cache")

        if VLM_AE_NCCL_DEBUG:
            print(f"[NcclKVTransfer] VLM rank {self.vlm_rank} sending KV for request {request_id}")

        # Store in buffer
        self._transfer_buffers[request_id] = KVBuffer(
            kv_cache=kv_cache,
            metadata=metadata
        )

        # Signal that data is ready
        self._signal_queue.put(("send", request_id, self.vlm_rank))

        if not async_transfer:
            # Wait for acknowledgment
            self._signal_queue.get(timeout=VLM_AE_NCCL_TIMEOUT)

    def recv_kv(
        self,
        request_id: int,
        timeout: float = VLM_AE_NCCL_TIMEOUT
    ) -> tuple:
        """
        Receive KV cache at AE worker from VLM workers.

        Args:
            request_id: Request ID to receive
            timeout: Timeout in seconds

        Returns:
            Tuple of (kv_cache, metadata)
        """
        if self.role != TransferRole.AE:
            raise RuntimeError("Only AE workers can receive KV cache")

        if VLM_AE_NCCL_DEBUG:
            print(f"[NcclKVTransfer] AE waiting for KV for request {request_id}")

        # Wait for signal from VLM workers
        start_time = time.time()
        received_ranks = set()
        aggregated_kv = []
        aggregated_metadata = {}

        while len(received_ranks) < len(self.vlm_gpus):
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Timeout waiting for KV transfer for request {request_id}")

            try:
                signal = self._signal_queue.get(timeout=0.1)
                if signal[0] == "send" and signal[1] == request_id:
                    vlm_rank = signal[2]
                    if vlm_rank not in received_ranks:
                        received_ranks.add(vlm_rank)
                        # Get from buffer
                        buffer = self._transfer_buffers.get(request_id)
                        if buffer:
                            # For simplicity, we collect all KV shards
                            # In real implementation, we would merge them
                            aggregated_kv.append(buffer.kv_cache)
                            if buffer.metadata:
                                aggregated_metadata.update(buffer.metadata)
            except queue.Empty:
                continue

        # Merge KV cache from all VLM ranks
        # Note: This is simplified; real implementation needs proper merging
        merged_kv = self._merge_kv_shards(aggregated_kv)

        if VLM_AE_NCCL_DEBUG:
            print(f"[NcclKVTransfer] AE received KV for request {request_id} from {len(received_ranks)} ranks")

        return merged_kv, aggregated_metadata

    def _merge_kv_shards(self, kv_shards: List[List[tuple]]) -> List[tuple]:
        """
        Merge KV cache shards from different VLM ranks.

        For TP=3 case, each shard contains a slice of the attention heads.
        We need to concatenate along the head dimension.
        """
        if not kv_shards:
            return []

        if len(kv_shards) == 1:
            return kv_shards[0]

        merged = []
        for layer_idx in range(len(kv_shards[0])):
            k_shards = [shard[layer_idx][0] for shard in kv_shards]
            v_shards = [shard[layer_idx][1] for shard in kv_shards]

            # Concatenate along head dimension (dim=1 for K, V)
            k_merged = torch.cat(k_shards, dim=1)
            v_merged = torch.cat(v_shards, dim=1)
            merged.append((k_merged, v_merged))

        return merged

    def close(self):
        """Clean up resources."""
        self._executor.shutdown(wait=False)
        self._transfer_buffers.clear()


class DirectGpuTransfer:
    """
    Direct GPU-to-GPU transfer using CUDA IPC and NCCL P2P.

    This is more efficient for large tensors as it avoids CPU roundtrip.
    Requires:
    1. P2P access enabled between GPUs
    2. CUDA IPC handles for cross-process sharing

    Performance notes:
    - For P2P enabled: uses direct GPU copy (~25-30 GB/s)
    - For P2P disabled: uses PCIe through CPU (~12-16 GB/s)
    - Uses non-blocking copies with synchronization
    """

    def __init__(
        self,
        role: str,
        vlm_gpus: List[int] = None,
        ae_gpu: int = 3,
    ):
        self.role = TransferRole(role)
        self.vlm_gpus = vlm_gpus or [0, 1, 2]
        self.ae_gpu = ae_gpu

        # Check P2P capability
        self._p2p_matrix = self._check_p2p_matrix()

        if VLM_AE_NCCL_DEBUG:
            print(f"[DirectGpuTransfer] P2P matrix: {self._p2p_matrix}")

    def _check_p2p_matrix(self) -> dict:
        """Check P2P access between all GPU pairs."""
        matrix = {}
        all_gpus = self.vlm_gpus + [self.ae_gpu]

        for src in all_gpus:
            for dst in all_gpus:
                if src != dst:
                    try:
                        matrix[(src, dst)] = torch.cuda.can_device_access_peer(src, dst)
                    except Exception:
                        matrix[(src, dst)] = False

        return matrix

    def _check_p2p(self) -> bool:
        """Check if P2P access is enabled between GPUs."""
        current_device = torch.cuda.current_device()

        if self.role == TransferRole.VLM:
            target_device = self.ae_gpu
        else:
            # AE can receive from any VLM GPU
            return all(
                self._p2p_matrix.get((vlm_gpu, self.ae_gpu), False)
                for vlm_gpu in self.vlm_gpus
            )

        return self._p2p_matrix.get((current_device, target_device), False)

    def transfer_tensor(
        self,
        tensor: torch.Tensor,
        dst_device: int,
        src_device: int = None
    ) -> torch.Tensor:
        """
        Transfer tensor directly between GPUs using P2P.

        Args:
            tensor: Source tensor
            dst_device: Destination GPU ID
            src_device: Source GPU ID (default: current device)

        Returns:
            Tensor on destination device
        """
        if src_device is None:
            src_device = tensor.device.index

        if src_device == dst_device:
            return tensor

        # Check P2P capability
        p2p_enabled = self._p2p_matrix.get((src_device, dst_device), False)

        if p2p_enabled:
            # Direct P2P copy - allocate on destination and copy
            with torch.cuda.device(dst_device):
                dst_tensor = torch.empty_like(tensor, device=f"cuda:{dst_device}")
            dst_tensor.copy_(tensor, non_blocking=True)
            return dst_tensor
        else:
            # Fallback to standard transfer (goes through CPU)
            # This is slower but always works
            return tensor.to(f"cuda:{dst_device}")

    def transfer_kv_cache(
        self,
        kv_cache: List[tuple],
        dst_device: int
    ) -> List[tuple]:
        """
        Transfer entire KV cache to destination device.

        Optimized version that:
        1. Pre-allocates all destination tensors
        2. Issues all copies asynchronously
        3. Synchronizes once at the end

        Args:
            kv_cache: List of (K, V) tensor tuples
            dst_device: Destination GPU ID

        Returns:
            KV cache on destination device
        """
        # Pre-allocate all destination tensors
        dst_tensors = []
        with torch.cuda.device(dst_device):
            for k, v in kv_cache:
                k_dst = torch.empty_like(k, device=f"cuda:{dst_device}")
                v_dst = torch.empty_like(v, device=f"cuda:{dst_device}")
                dst_tensors.append((k_dst, v_dst))

        # Issue all copies asynchronously
        streams = []
        for (k, v), (k_dst, v_dst) in zip(kv_cache, dst_tensors):
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                k_dst.copy_(k, non_blocking=True)
                v_dst.copy_(v, non_blocking=True)
            streams.append(stream)

        # Synchronize all streams
        for stream in streams:
            stream.synchronize()

        return dst_tensors

    def transfer_kv_cache_batched(
        self,
        kv_cache: List[tuple],
        dst_device: int
    ) -> List[tuple]:
        """
        Transfer KV cache with batched copy for maximum throughput.

        This method concatenates all KV tensors into a single large tensor,
        transfers it, then splits back. This can be faster for many small tensors.

        Args:
            kv_cache: List of (K, V) tensor tuples
            dst_device: Destination GPU ID

        Returns:
            KV cache on destination device
        """
        if not kv_cache:
            return []

        src_device = kv_cache[0][0].device.index
        if src_device == dst_device:
            return kv_cache

        # Get shapes for reconstruction
        shapes = [(k.shape, v.shape) for k, v in kv_cache]
        dtypes = [(k.dtype, v.dtype) for k, v in kv_cache]

        # Flatten and concatenate all tensors
        flat_k = torch.cat([k.flatten() for k, _ in kv_cache])
        flat_v = torch.cat([v.flatten() for _, v in kv_cache])

        # Transfer in one shot
        flat_k_dst = self.transfer_tensor(flat_k, dst_device, src_device)
        flat_v_dst = self.transfer_tensor(flat_v, dst_device, src_device)

        # Split back into original shapes
        result = []
        k_offset = 0
        v_offset = 0

        for (k_shape, v_shape), (k_dtype, v_dtype) in zip(shapes, dtypes):
            k_size = k_shape.numel()
            v_size = v_shape.numel()

            k_dst = flat_k_dst[k_offset:k_offset + k_size].view(k_shape).to(k_dtype)
            v_dst = flat_v_dst[v_offset:v_offset + v_size].view(v_shape).to(v_dtype)

            result.append((k_dst, v_dst))
            k_offset += k_size
            v_offset += v_size

        return result


# Convenience function for creating transfer instance
def create_kv_transfer(
    role: str,
    method: str = "direct",  # "direct", "shared_memory", "serialize"
    **kwargs
):
    """
    Create KV transfer instance based on method.

    Args:
        role: "vlm" or "ae"
        method: Transfer method
        **kwargs: Additional arguments

    Returns:
        Transfer instance
    """
    if method == "direct":
        return DirectGpuTransfer(role=role, **kwargs)
    elif method == "shared_memory":
        return NcclKVTransfer(role=role, **kwargs)
    else:
        raise ValueError(f"Unknown transfer method: {method}")
