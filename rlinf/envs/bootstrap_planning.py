# Copyright 2026 The RLinf Authors.
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

import copy
from typing import Any

import numpy as np


class BootstrapResetPlannerMixin:
    """Mixin for environments that participate in env double-buffer bootstrap planning.

    Provides the plan/apply protocol so that the active pool can snapshot its
    planner state (reseeded reset IDs, start offset, generator state) and the
    standby pool can restore it for a deterministic hand-off at epoch boundaries.

    Set ``_manages_bootstrap_state = False`` on subclasses that do not own
    bootstrap state (e.g. stateless procedural envs). In that case
    ``plan_next_bootstrap_reset`` returns ``None`` and the standby pool resets
    independently, providing latency hiding without state continuity.
    """

    bootstrap_planner_array_fields = ("reset_state_ids",)
    bootstrap_planner_scalar_fields = ("start_idx",)
    bootstrap_planner_generator_fields = (("generator_state", "_generator"),)
    _manages_bootstrap_state: bool = True

    def _clone_reset_state_ids(self):
        return np.array(self.reset_state_ids, copy=True)

    def _snapshot_bootstrap_planner_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        for field_name in self.bootstrap_planner_array_fields:
            state[field_name] = np.array(getattr(self, field_name), copy=True)
        for field_name in self.bootstrap_planner_scalar_fields:
            state[field_name] = int(getattr(self, field_name))
        for state_key, attr_name in self.bootstrap_planner_generator_fields:
            generator = getattr(self, attr_name)
            state[state_key] = copy.deepcopy(generator.bit_generator.state)
        return state

    def _restore_bootstrap_planner_state(self, state: dict[str, Any]) -> None:
        for field_name in self.bootstrap_planner_array_fields:
            setattr(self, field_name, np.array(state[field_name], copy=True))
        for field_name in self.bootstrap_planner_scalar_fields:
            setattr(self, field_name, int(state[field_name]))
        for state_key, attr_name in self.bootstrap_planner_generator_fields:
            generator = getattr(self, attr_name)
            generator.bit_generator.state = copy.deepcopy(state[state_key])

    def plan_next_bootstrap_reset(self) -> dict[str, Any]:
        """Produce a plan dict with the next reset IDs and a snapshot of planner state."""
        if not self._manages_bootstrap_state:
            return None
        if self.use_fixed_reset_state_ids and self.is_start:
            plan_ids = self._clone_reset_state_ids()
        else:
            self.update_reset_state_ids()
            plan_ids = self._clone_reset_state_ids()
        return {
            "reset_state_ids": plan_ids,
            "planner_state": self._snapshot_bootstrap_planner_state(),
        }

    def apply_bootstrap_plan(self, plan):
        """Restore planner state and return reset_state_ids for subsequent reset."""
        planner_state = None if plan is None else plan.get("planner_state", None)
        if planner_state is not None:
            self._restore_bootstrap_planner_state(planner_state)
        return plan.get("reset_state_ids", None) if plan is not None else None
