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

import json

import numpy as np
import pytest
import torch

from rlinf.profiling.fixture import (
    FIXTURE_SCHEMA_VERSION,
    ProfileFixtureCapture,
    ProfileFixtureKind,
    load_profile_fixture,
)


def test_disabled_capture_does_not_create_output_directory(tmp_path):
    output_dir = tmp_path / "fixtures"
    capture = ProfileFixtureCapture(output_dir=output_dir)

    result = capture.maybe_dump(
        ProfileFixtureKind.ACTOR_MICROBATCH,
        version=2,
        payload={"tensor": torch.ones(2)},
    )

    assert result is None
    assert not capture.wants(ProfileFixtureKind.ACTOR_MICROBATCH, version=2)
    assert not output_dir.exists()


def test_capture_filters_version_and_limits_each_kind(tmp_path):
    capture = ProfileFixtureCapture(
        enabled=True,
        output_dir=tmp_path,
        kinds=[ProfileFixtureKind.ACTOR_MICROBATCH.value],
        rank=3,
        role="Actor",
        target_version=2,
        max_per_kind_per_rank=1,
    )

    assert (
        capture.maybe_dump(
            ProfileFixtureKind.ACTOR_MICROBATCH,
            version=1,
            payload={"tensor": torch.ones(2)},
        )
        is None
    )
    fixture = capture.maybe_dump(
        ProfileFixtureKind.ACTOR_MICROBATCH,
        version=2,
        payload={"tensor": torch.ones(2)},
    )
    assert fixture is not None
    assert (
        capture.maybe_dump(
            ProfileFixtureKind.ACTOR_MICROBATCH,
            version=2,
            payload={"tensor": torch.ones(2)},
        )
        is None
    )
    assert fixture.name == "actor-actor_microbatch-rank0003-00.pt"


def test_capture_moves_nested_tensors_to_cpu_and_writes_manifest(tmp_path):
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3).t()
    array = np.arange(4, dtype=np.int32).reshape(2, 2)
    capture = ProfileFixtureCapture(
        enabled=True,
        output_dir=tmp_path,
        kinds=[ProfileFixtureKind.ROLLOUT_OBSERVATION.value],
        rank=1,
        role="rollout",
    )

    fixture = capture.maybe_dump(
        ProfileFixtureKind.ROLLOUT_OBSERVATION,
        version=4,
        payload={"nested": [source], "array": array},
        metadata={"batch_size": 2},
    )

    record = load_profile_fixture(fixture)
    saved_tensor = record["payload"]["nested"][0]
    assert saved_tensor.device.type == "cpu"
    assert saved_tensor.is_contiguous()
    assert torch.equal(saved_tensor, source)
    assert np.array_equal(record["payload"]["array"], array)

    manifest = json.loads(fixture.with_suffix(".json").read_text())
    assert manifest["schema_version"] == FIXTURE_SCHEMA_VERSION
    assert manifest["file_size"] == fixture.stat().st_size
    assert manifest["mtime_ns"] == fixture.stat().st_mtime_ns
    assert manifest["payload"]["nested"][0] == {
        "type": "tensor",
        "shape": [3, 2],
        "dtype": "torch.float32",
    }
    assert "checksum" not in manifest
    assert not list(tmp_path.glob("*.tmp"))


def test_capture_rejects_unsupported_payload_before_creating_files(tmp_path):
    capture = ProfileFixtureCapture(
        enabled=True,
        output_dir=tmp_path,
        kinds=[ProfileFixtureKind.ACTOR_MICROBATCH.value],
    )

    with pytest.raises(TypeError, match="Unsupported fixture value"):
        capture.maybe_dump(
            ProfileFixtureKind.ACTOR_MICROBATCH,
            version=0,
            payload={"bad": object()},
        )

    assert not list(tmp_path.iterdir())


def test_load_rejects_unknown_schema(tmp_path):
    fixture = tmp_path / "fixture.pt"
    torch.save({"schema_version": FIXTURE_SCHEMA_VERSION + 1}, fixture)

    with pytest.raises(ValueError, match="Unsupported fixture schema"):
        load_profile_fixture(fixture)
