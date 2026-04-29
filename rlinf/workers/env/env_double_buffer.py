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
import concurrent.futures
import time
from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass
class PendingTrainBootstrap:
    pool_idx: int
    bootstrap_plans: list[Any]
    future: concurrent.futures.Future


class EnvDoubleBufferCoordinator:
    def __init__(
        self,
        *,
        env_pools: list[list[Any]],
        bootstrap_envs: Callable[[list[Any], list[Any] | None], list[Any]],
        send_bootstrap: Callable[[Any, list[Any]], None],
        set_active_envs: Callable[[list[Any]], None],
        record_timer_duration: Callable[[str, float], None],
        log_warning: Callable[[str], None],
        on_disabled: Callable[[], None] | None = None,
        rank: int,
    ) -> None:
        if len(env_pools) != 2:
            raise ValueError("Env double buffer requires exactly two env pools.")
        self.env_pools = env_pools
        self._bootstrap_envs = bootstrap_envs
        self._send_bootstrap = send_bootstrap
        self._set_active_envs = set_active_envs
        self._record_timer_duration = record_timer_duration
        self._log_warning = log_warning
        self._on_disabled = on_disabled
        self.active_pool_idx = 0
        self.pending_bootstrap: PendingTrainBootstrap | None = None
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"env-double-buffer-rank{rank}",
        )
        self.prepare_total = 0
        self.prepare_hits = 0
        self.fallback_count = 0
        self.disabled = False
        self._set_active_pool_idx(0)

    @staticmethod
    def env_supports_double_buffer(env: Any) -> bool:
        return hasattr(env, "plan_next_bootstrap_reset") and hasattr(
            env, "reset_to_bootstrap_plan"
        )

    @classmethod
    def envs_support_double_buffer(cls, env_list: list[Any]) -> bool:
        return all(cls.env_supports_double_buffer(env) for env in env_list)

    def _set_active_pool_idx(self, pool_idx: int) -> None:
        self.active_pool_idx = pool_idx
        self._set_active_envs(self.env_pools[pool_idx])

    def reset_run_state(self) -> None:
        self.cancel_pending()
        self.prepare_total = 0
        self.prepare_hits = 0
        self.fallback_count = 0

    def disable(self, *, cancel_pending: bool = True) -> None:
        self.disabled = True
        if self._on_disabled is not None:
            self._on_disabled()
        if cancel_pending:
            self.cancel_pending()

    def cancel_pending(self, *, wait: bool = True) -> None:
        pending_bootstrap = self.pending_bootstrap
        if pending_bootstrap is None:
            return
        cancelled = pending_bootstrap.future.cancel()
        if wait and not cancelled:
            try:
                pending_bootstrap.future.result()
            except Exception as exc:
                self._log_warning(
                    "Env double buffer pending prepare finished with an error while "
                    f"waiting for cancellation/shutdown: {exc}"
                )
        self.pending_bootstrap = None

    def shutdown(self) -> None:
        self.cancel_pending(wait=True)
        self._executor.shutdown(wait=True, cancel_futures=True)

    def _plan_bootstrap_for_pool(self, pool_idx: int) -> list[Any]:
        return [env.plan_next_bootstrap_reset() for env in self.env_pools[pool_idx]]

    def _prepare_bootstrap_for_pool(
        self, pool_idx: int, bootstrap_plans: list[Any]
    ) -> tuple[int, list[Any], float]:
        start_time = time.perf_counter()
        env_outputs = self._bootstrap_envs(
            self.env_pools[pool_idx],
            bootstrap_plans,
        )
        return pool_idx, env_outputs, time.perf_counter() - start_time

    def schedule_next_bootstrap_prepare(self) -> None:
        if self.disabled:
            return
        if self.pending_bootstrap is not None:
            raise RuntimeError(
                "A double-buffer bootstrap prepare task is already active."
            )
        source_pool_idx = self.active_pool_idx
        target_pool_idx = 1 - source_pool_idx
        bootstrap_plans = self._plan_bootstrap_for_pool(source_pool_idx)
        future = self._executor.submit(
            self._prepare_bootstrap_for_pool,
            target_pool_idx,
            bootstrap_plans,
        )
        self.pending_bootstrap = PendingTrainBootstrap(
            pool_idx=target_pool_idx,
            bootstrap_plans=bootstrap_plans,
            future=future,
        )

    async def consume_bootstrap(self, rollout_channel: Any) -> list[Any]:
        if self.disabled:
            env_outputs = self._bootstrap_envs(
                self.env_pools[self.active_pool_idx],
                None,
            )
            self._send_bootstrap(rollout_channel, env_outputs)
            return env_outputs
        pending_bootstrap = self.pending_bootstrap
        if pending_bootstrap is None:
            self.fallback_count += 1
            env_outputs = self._bootstrap_envs(
                self.env_pools[self.active_pool_idx],
                None,
            )
            self._send_bootstrap(rollout_channel, env_outputs)
            return env_outputs

        task_ready = pending_bootstrap.future.done()
        wait_start = None if task_ready else time.perf_counter()
        try:
            pool_idx, env_outputs, prepare_duration = await asyncio.wrap_future(
                pending_bootstrap.future
            )
        except Exception as exc:
            self._log_warning(
                "Env double buffer background prepare failed; using synchronous bootstrap "
                f"for this boundary. Error: {exc}"
            )
            self.fallback_count += 1
            try:
                env_outputs = self._bootstrap_envs(
                    self.env_pools[self.active_pool_idx],
                    pending_bootstrap.bootstrap_plans,
                )
            except Exception as fallback_exc:
                self.disable(cancel_pending=False)
                self._log_warning(
                    "Fatal env double buffer fallback failure; disabling coordinator. "
                    "The active pool could not reset with the already planned "
                    f"bootstrap plan. Error: {fallback_exc}"
                )
                raise RuntimeError(
                    "Env double buffer fallback failed on the active pool."
                ) from fallback_exc
            self._send_bootstrap(rollout_channel, env_outputs)
            return env_outputs
        finally:
            if wait_start is not None:
                self._record_timer_duration(
                    "bootstrap_wait_ready", time.perf_counter() - wait_start
                )
            self.pending_bootstrap = None

        self.prepare_total += 1
        if task_ready:
            self.prepare_hits += 1
        self._record_timer_duration("bootstrap_prepare_bg", prepare_duration)
        if pool_idx == pending_bootstrap.pool_idx:
            self._set_active_pool_idx(pool_idx)
        self._send_bootstrap(rollout_channel, env_outputs)
        return env_outputs

    def append_metrics(self, env_metrics: dict[str, list]) -> None:
        total = max(self.prepare_total, 1)
        hit_ratio = self.prepare_hits / total
        env_metrics["double_buffer_hit_ratio"].append(
            torch.tensor([hit_ratio], dtype=torch.float32)
        )
        env_metrics["double_buffer_fallback_count"].append(
            torch.tensor([float(self.fallback_count)], dtype=torch.float32)
        )
