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

"""Tests for NvidiaGPUManager profiling methods in nvidia_gpu.py."""

from __future__ import annotations

import asyncio
from unittest.mock import call, patch

import pytest

import rlinf.scheduler.hardware.accelerators.nvidia_gpu as nv_module
from rlinf.scheduler.hardware.accelerators.nvidia_gpu import NvidiaGPUManager


@pytest.fixture(autouse=True)
def _reset_profiling_flag():
    """Reset module-level flag between tests for isolation."""
    nv_module._nv_profiling_active = False
    yield
    nv_module._nv_profiling_active = False


class TestStartStop:
    """NvidiaGPUManager.start_profiling / stop_profiling drive torch.cuda.profiler."""

    def test_start_sets_active_and_calls_cuda_profiler_start(self):
        with patch("torch.cuda.profiler.start") as mock_start:
            NvidiaGPUManager.start_profiling(step_idx=5)
        assert NvidiaGPUManager.is_profiling_active() is True
        mock_start.assert_called_once_with()

    def test_stop_clears_active_and_calls_cuda_profiler_stop(self):
        with patch("torch.cuda.profiler.start"):
            NvidiaGPUManager.start_profiling()
        with patch("torch.cuda.profiler.stop") as mock_stop:
            NvidiaGPUManager.stop_profiling()
        assert NvidiaGPUManager.is_profiling_active() is False
        mock_stop.assert_called_once_with()

    def test_double_start_is_idempotent(self):
        with patch("torch.cuda.profiler.start") as mock_start:
            NvidiaGPUManager.start_profiling()
            NvidiaGPUManager.start_profiling()
        assert mock_start.call_count == 1

    def test_stop_without_start_is_noop(self):
        with patch("torch.cuda.profiler.stop") as mock_stop:
            NvidiaGPUManager.stop_profiling()
        mock_stop.assert_not_called()


class TestProfilingRangeSync:
    """NvidiaGPUManager.profiling_range on sync code is transparent off / wraps on."""

    def test_passes_through_when_inactive(self):
        results = []
        with NvidiaGPUManager.profiling_range("test/op"):
            results.append(42)
        assert results == [42]
        assert NvidiaGPUManager.is_profiling_active() is False

    def test_emits_range_when_active(self):
        with patch("torch.cuda.profiler.start"):
            NvidiaGPUManager.start_profiling()

        with (
            patch("torch.cuda.nvtx.range_start", return_value=17) as mock_range_start,
            patch("torch.cuda.nvtx.range_end") as mock_range_end,
        ):
            with NvidiaGPUManager.profiling_range("test/op", color="green"):
                pass

        mock_range_start.assert_called_once_with("test/op")
        mock_range_end.assert_called_once_with(17)

    def test_ends_range_on_exception(self):
        with patch("torch.cuda.profiler.start"):
            NvidiaGPUManager.start_profiling()

        with (
            patch("torch.cuda.nvtx.range_start", return_value=23) as mock_range_start,
            patch("torch.cuda.nvtx.range_end") as mock_range_end,
        ):
            with pytest.raises(ValueError, match="oops"):
                with NvidiaGPUManager.profiling_range("test/op"):
                    raise ValueError("oops")

        mock_range_start.assert_called_once_with("test/op")
        mock_range_end.assert_called_once_with(23)


class TestProfilingRangeAsync:
    """profiling_range works correctly inside async coroutines."""

    def test_passes_through_when_inactive(self):
        async def coro():
            with NvidiaGPUManager.profiling_range("test/async_op"):
                return 99

        assert asyncio.run(coro()) == 99

    def test_emits_range_when_active(self):
        with patch("torch.cuda.profiler.start"):
            NvidiaGPUManager.start_profiling()

        async def coro():
            with NvidiaGPUManager.profiling_range("test/async_op"):
                return 7

        with (
            patch("torch.cuda.nvtx.range_start", return_value=31) as mock_range_start,
            patch("torch.cuda.nvtx.range_end") as mock_range_end,
        ):
            result = asyncio.run(coro())

        assert result == 7
        mock_range_start.assert_called_once_with("test/async_op")
        mock_range_end.assert_called_once_with(31)

    def test_interleaved_tasks_end_their_own_ranges(self):
        with patch("torch.cuda.profiler.start"):
            NvidiaGPUManager.start_profiling()

        async def run_interleaved():
            first_entered = asyncio.Event()
            second_entered = asyncio.Event()
            release_first = asyncio.Event()
            release_second = asyncio.Event()

            async def first():
                with NvidiaGPUManager.profiling_range("first"):
                    first_entered.set()
                    await release_first.wait()

            async def second():
                await first_entered.wait()
                with NvidiaGPUManager.profiling_range("second"):
                    second_entered.set()
                    await release_second.wait()

            first_task = asyncio.create_task(first())
            second_task = asyncio.create_task(second())
            await second_entered.wait()
            release_first.set()
            await first_task
            release_second.set()
            await second_task

        with (
            patch("torch.cuda.nvtx.range_start", side_effect=[41, 42]) as range_start,
            patch("torch.cuda.nvtx.range_end") as range_end,
        ):
            asyncio.run(run_interleaved())

        assert range_start.call_args_list == [call("first"), call("second")]
        assert range_end.call_args_list == [call(41), call(42)]

    def test_cancelled_task_ends_its_range(self):
        with patch("torch.cuda.profiler.start"):
            NvidiaGPUManager.start_profiling()

        async def run_cancelled():
            entered = asyncio.Event()

            async def pending():
                with NvidiaGPUManager.profiling_range("cancelled"):
                    entered.set()
                    await asyncio.Event().wait()

            task = asyncio.create_task(pending())
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        with (
            patch("torch.cuda.nvtx.range_start", return_value=51) as range_start,
            patch("torch.cuda.nvtx.range_end") as range_end,
        ):
            asyncio.run(run_cancelled())

        range_start.assert_called_once_with("cancelled")
        range_end.assert_called_once_with(51)
