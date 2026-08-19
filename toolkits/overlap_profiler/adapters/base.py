# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Common replay adapter protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ReplayAdapter(ABC):
    @abstractmethod
    def prepare(self, stage: str) -> None:
        """Prepare stage inputs and preceding work outside the timed interval."""

    def prepare_iteration(self) -> None:
        """Rebuild transient state such as an autograd graph before one trial."""

    @abstractmethod
    def run_once(self) -> Any:
        """Execute exactly one prepared stage workload."""

    @abstractmethod
    def work_signature(self) -> Any:
        """Return a stable structural description of the prepared workload."""

    def synchronize(self) -> None:
        """Synchronize the active accelerator after a trial."""
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def reset_between_trials(self) -> None:
        """Reset mutable state after a trial when the stage requires it."""
