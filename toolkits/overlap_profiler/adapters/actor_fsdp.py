# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Replay fixed async-PPO micro-batches on an initialized FSDP actor worker."""

from __future__ import annotations

from enum import Enum
from typing import Any

from rlinf.profiling import ProfileFixtureKind, load_profile_fixture
from rlinf.utils.nested_dict_process import put_tensor_device

from ..work_signature import work_signature
from .base import ReplayAdapter


class ActorReplayStage(str, Enum):
    FORWARD = "actor.forward"
    BACKWARD_NO_SYNC = "actor.backward_no_sync"
    BACKWARD_SYNC = "actor.backward_sync"
    OPTIMIZER = "actor.optimizer"
    FULL_MICROBATCH = "actor.full_microbatch"


class ActorFSDPReplayAdapter(ReplayAdapter):
    """Use production actor methods while keeping orchestration outside RLinf."""

    def __init__(self, actor: Any, fixture_path: str):
        self.actor = actor
        self.fixture = load_profile_fixture(fixture_path)
        if self.fixture["kind"] != ProfileFixtureKind.ACTOR_MICROBATCH.value:
            raise ValueError("Actor replay requires an actor_microbatch fixture")
        self.stage: ActorReplayStage | None = None
        self.data: dict[str, Any] | None = None
        self._loss = None
        self._backward_context = None

    def prepare(self, stage: str) -> None:
        self.stage = ActorReplayStage(stage)
        device = getattr(self.actor, "device", "cuda")
        self.data = put_tensor_device(self.fixture["payload"]["data"], device)
        self.actor.model.train()
        gradient_accumulation = self.fixture["metadata"].get(
            "gradient_accumulation", self.actor.gradient_accumulation
        )
        self.actor.gradient_accumulation = int(gradient_accumulation)
        self.actor.optimizer.zero_grad()

    def prepare_iteration(self) -> None:
        self._require_prepared()
        self._loss = None
        self._backward_context = None
        if self.stage in {
            ActorReplayStage.BACKWARD_NO_SYNC,
            ActorReplayStage.BACKWARD_SYNC,
            ActorReplayStage.OPTIMIZER,
        }:
            sync = self.stage != ActorReplayStage.BACKWARD_NO_SYNC
            self._backward_context = self.actor.before_micro_batch(
                self.actor.model, is_last_micro_batch=sync
            )
            self._loss, _, _ = self.actor.forward_and_loss_micro_batch(self.data)
        if self.stage == ActorReplayStage.OPTIMIZER:
            with self._backward_context:
                self.actor.grad_scaler.scale(self._loss).backward()

    def run_once(self) -> Any:
        self._require_prepared()
        if self.stage == ActorReplayStage.FORWARD:
            return self.actor.forward_micro_batch(self.data)[0]
        if self.stage in {
            ActorReplayStage.BACKWARD_NO_SYNC,
            ActorReplayStage.BACKWARD_SYNC,
        }:
            if self._loss is None or self._backward_context is None:
                raise RuntimeError("prepare_iteration() must precede backward replay")
            with self._backward_context:
                return self.actor.grad_scaler.scale(self._loss).backward()
        if self.stage == ActorReplayStage.OPTIMIZER:
            if self._loss is None:
                raise RuntimeError("prepare_iteration() must precede optimizer replay")
            return self.actor.optimizer_step()

        self.actor.optimizer.zero_grad()
        backward_context = self.actor.before_micro_batch(
            self.actor.model, is_last_micro_batch=True
        )
        loss, metrics, _ = self.actor.forward_and_loss_micro_batch(self.data)
        with backward_context:
            self.actor.grad_scaler.scale(loss).backward()
        optimizer_result = self.actor.optimizer_step()
        return metrics, optimizer_result

    def reset_between_trials(self) -> None:
        self.actor.optimizer.zero_grad()
        self._loss = None
        self._backward_context = None

    def work_signature(self) -> Any:
        self._require_prepared()
        return {
            "stage": self.stage.value,
            "rank": self.fixture["rank"],
            "version": self.fixture["version"],
            "data": work_signature(self.data),
            "gradient_accumulation": self.actor.gradient_accumulation,
        }

    def _require_prepared(self) -> None:
        if self.stage is None or self.data is None:
            raise RuntimeError("prepare() must be called before replay")
