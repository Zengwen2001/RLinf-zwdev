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

"""Opt-in, rank-local workload capture for overlap replay experiments."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

FIXTURE_SCHEMA_VERSION = 1


class ProfileFixtureKind(str, Enum):
    ACTOR_MICROBATCH = "actor_microbatch"
    ROLLOUT_OBSERVATION = "rollout_observation"


def _to_cpu_tree(value: Any, path: str = "payload") -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value.copy())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {key: _to_cpu_tree(item, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(
            _to_cpu_tree(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, list):
        return [
            _to_cpu_tree(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    raise TypeError(f"Unsupported fixture value at {path}: {type(value).__name__}")


def _describe_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "type": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, Mapping):
        return {key: _describe_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_describe_tree(item) for item in value]
    return {"type": type(value).__name__}


class ProfileFixtureCapture:
    """Capture a bounded number of fixed workloads for one worker rank."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        output_dir: str | os.PathLike[str] | None = None,
        kinds: Sequence[str] = (),
        rank: int = 0,
        role: str = "worker",
        target_version: int | None = None,
        max_per_kind_per_rank: int = 1,
    ):
        self.enabled = bool(enabled)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.kinds = frozenset(kinds)
        self.rank = int(rank)
        self.role = role.lower()
        self.target_version = target_version
        self.max_per_kind_per_rank = int(max_per_kind_per_rank)
        self._counts: dict[str, int] = {}

        if self.enabled and self.output_dir is None:
            raise ValueError("profile_capture.output_dir is required when enabled")
        if self.max_per_kind_per_rank < 1:
            raise ValueError("max_per_kind_per_rank must be at least 1")

    @classmethod
    def from_config(cls, cfg: Any, *, rank: int, role: str):
        capture_cfg = OmegaConf.select(cfg, "runner.profile_capture", default=None)
        if capture_cfg is None:
            return cls(rank=rank, role=role)
        return cls(
            enabled=capture_cfg.get("enabled", False),
            output_dir=capture_cfg.get("output_dir", None),
            kinds=capture_cfg.get("kinds", ()),
            rank=rank,
            role=role,
            target_version=capture_cfg.get("version", None),
            max_per_kind_per_rank=capture_cfg.get("max_per_kind_per_rank", 1),
        )

    def maybe_dump(
        self,
        kind: ProfileFixtureKind,
        *,
        version: int,
        payload: Any,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path | None:
        if not self.wants(kind, version=version):
            return None

        count = self._counts.get(kind.value, 0)
        cpu_payload = _to_cpu_tree(payload)
        cpu_metadata = _to_cpu_tree(dict(metadata or {}), path="metadata")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{self.role}-{kind.value}-rank{self.rank:04d}-{count:02d}"
        fixture_path = self.output_dir / f"{stem}.pt"
        manifest_path = self.output_dir / f"{stem}.json"

        record = {
            "schema_version": FIXTURE_SCHEMA_VERSION,
            "kind": kind.value,
            "role": self.role,
            "rank": self.rank,
            "version": int(version),
            "payload": cpu_payload,
            "metadata": cpu_metadata,
        }
        self._atomic_torch_save(record, fixture_path)
        stat = fixture_path.stat()
        manifest = {
            "schema_version": FIXTURE_SCHEMA_VERSION,
            "kind": kind.value,
            "role": self.role,
            "rank": self.rank,
            "version": int(version),
            "fixture": fixture_path.name,
            "file_size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "payload": _describe_tree(cpu_payload),
            "metadata": cpu_metadata,
        }
        self._atomic_json_dump(manifest, manifest_path)
        self._counts[kind.value] = count + 1
        return fixture_path

    def wants(self, kind: ProfileFixtureKind, *, version: int) -> bool:
        """Return whether payload construction and capture are currently needed."""
        if not self.enabled or kind.value not in self.kinds:
            return False
        if self.target_version is not None and int(version) != int(self.target_version):
            return False
        return self._counts.get(kind.value, 0) < self.max_per_kind_per_rank

    @staticmethod
    def _atomic_torch_save(record: dict[str, Any], destination: Path) -> None:
        fd, temporary = tempfile.mkstemp(
            dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
        )
        os.close(fd)
        try:
            torch.save(record, temporary)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def _atomic_json_dump(record: dict[str, Any], destination: Path) -> None:
        fd, temporary = tempfile.mkstemp(
            dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(record, stream, indent=2, sort_keys=True)
                stream.write("\n")
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def load_profile_fixture(path: str | os.PathLike[str]) -> dict[str, Any]:
    record = torch.load(path, map_location="cpu", weights_only=False)
    if record.get("schema_version") != FIXTURE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported fixture schema version: {record.get('schema_version')}"
        )
    return record
