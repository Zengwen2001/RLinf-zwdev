# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Low-overhead inter-process exclusion for collocated GPU stages."""

import fcntl
import os
import re
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from rlinf.profiling import ProfileStage, profile_range


class InterProcessStageGate:
    """Serialize selected stages running in separate workers on one GPU."""

    def __init__(
        self,
        gate_id: str,
        *,
        node_rank: int,
        accelerator_rank: int,
        wait_stage: ProfileStage,
        synchronize: Callable[[], None],
        record_wait: Callable[[float], None] | None = None,
        priority_waiter: bool = False,
    ) -> None:
        """Open the lock shared by workers assigned to one accelerator."""
        safe_gate_id = re.sub(r"[^A-Za-z0-9_.-]", "_", gate_id)
        gate_dir = Path(tempfile.gettempdir()) / "rlinf-stage-gates"
        gate_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path = gate_dir / (
            f"{safe_gate_id}-node{node_rank}-accelerator{accelerator_rank}.lock"
        )
        self.priority_path = self.path.with_suffix(".priority.lock")
        self._fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        self._priority_fd = os.open(
            self.priority_path,
            os.O_CREAT | os.O_RDWR,
            0o600,
        )
        self._wait_stage = wait_stage
        self._synchronize = synchronize
        self._record_wait = record_wait
        self._priority_waiter = priority_waiter

    def _acquire(self) -> None:
        if self._priority_waiter:
            # Announce intent before waiting for the resource. Once admitted,
            # the resource lock itself keeps later holders out.
            fcntl.flock(self._priority_fd, fcntl.LOCK_SH)
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX)
            finally:
                fcntl.flock(self._priority_fd, fcntl.LOCK_UN)
            return

        while True:
            fcntl.flock(self._fd, fcntl.LOCK_EX)
            try:
                fcntl.flock(
                    self._priority_fd,
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
            except BlockingIOError:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                time.sleep(0.001)
                continue
            fcntl.flock(self._priority_fd, fcntl.LOCK_UN)
            return

    @contextmanager
    def hold(self) -> Iterator[None]:
        """Wait for ownership and release only after queued GPU work completes."""
        wait_start = time.perf_counter()
        with profile_range(self._wait_stage):
            self._acquire()
        if self._record_wait is not None:
            self._record_wait(time.perf_counter() - wait_start)
        try:
            yield
        finally:
            try:
                self._synchronize()
            finally:
                fcntl.flock(self._fd, fcntl.LOCK_UN)

    def close(self) -> None:
        """Close this process's lock descriptor."""
        if self._fd < 0:
            return
        os.close(self._fd)
        self._fd = -1
        os.close(self._priority_fd)
        self._priority_fd = -1
