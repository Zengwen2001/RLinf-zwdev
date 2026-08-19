# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Replay fixed legacy OpenPI rollout stages on an initialized policy model."""

from __future__ import annotations

import random
from enum import Enum
from typing import Any

import torch

from rlinf.profiling import ProfileFixtureKind, load_profile_fixture

from ..work_signature import work_signature
from .base import ReplayAdapter


class OpenPIReplayStage(str, Enum):
    OBS_PREPROCESS = "rollout.obs_preprocess"
    VLM_PREFIX = "rollout.vlm_prefix"
    AE_DENOISE = "rollout.ae_denoise"
    FULL_POLICY = "rollout.full_policy"


class OpenPIReplayAdapter(ReplayAdapter):
    """Replay OpenPI boundaries already exposed by the production model."""

    def __init__(self, model: Any, fixture_path: str):
        self.model = model
        self.fixture = load_profile_fixture(fixture_path)
        if self.fixture["kind"] != ProfileFixtureKind.ROLLOUT_OBSERVATION.value:
            raise ValueError("OpenPI replay requires a rollout_observation fixture")
        self.stage: OpenPIReplayStage | None = None
        self.env_obs = self.fixture["payload"]["env_obs"]
        self.mode = self.fixture["metadata"].get("mode", "train")
        self.observation = None
        self.preprocessed = None
        self.prefix = None
        self.noise = None
        self._denoise_rng_state = None
        self._denoise_cuda_rng_state = None

    def prepare(self, stage: str) -> None:
        self.stage = OpenPIReplayStage(stage)
        self.model.eval()
        self._restore_rng()
        if self.stage in {
            OpenPIReplayStage.VLM_PREFIX,
            OpenPIReplayStage.AE_DENOISE,
        }:
            self.observation = self._build_observation()
            self.preprocessed = self.model._preprocess_observation(
                self.observation, train=False
            )
        if self.stage == OpenPIReplayStage.AE_DENOISE:
            self._prepare_ae_inputs()

    def prepare_iteration(self) -> None:
        self._require_prepared()
        if self.stage == OpenPIReplayStage.AE_DENOISE:
            self._restore_rng()
            self._prepare_ae_inputs()

    def run_once(self) -> Any:
        self._require_prepared()
        self._restore_rng()
        if self.stage == OpenPIReplayStage.OBS_PREPROCESS:
            observation = self._build_observation()
            return self.model._preprocess_observation(observation, train=False)
        if self.stage == OpenPIReplayStage.VLM_PREFIX:
            return self.model._build_prefix_cache(*self.preprocessed[:4])
        if self.stage == OpenPIReplayStage.AE_DENOISE:
            torch.set_rng_state(self._denoise_rng_state)
            if self._denoise_cuda_rng_state is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(self._denoise_cuda_rng_state)
            state = self.preprocessed[4]
            prefix_output, prefix_pad_masks, past_key_values = self.prefix
            return self.model._sample_actions_with_prefix_cache(
                state,
                prefix_output,
                prefix_pad_masks,
                past_key_values,
                noise=self.noise.clone(),
                mode=self.mode,
                compute_values=True,
            )
        return self.model.predict_action_batch(self.env_obs, mode=self.mode)

    def work_signature(self) -> Any:
        self._require_prepared()
        return {
            "stage": self.stage.value,
            "rank": self.fixture["rank"],
            "version": self.fixture["version"],
            "observation": work_signature(self.env_obs),
            "num_steps": int(self.model.config.num_steps),
            "action_horizon": int(self.model.config.action_horizon),
            "action_dim": int(self.model.config.action_dim),
        }

    def _build_observation(self):
        from openpi.models import model as openpi_model

        processed = self.model.obs_processor(self.env_obs)
        processed = self.model.input_transform(processed, transpose=False)
        processed = self.model.precision_processor(processed)
        return openpi_model.Observation.from_dict(processed)

    def _restore_rng(self) -> None:
        payload = self.fixture["payload"]
        torch.set_rng_state(payload["torch_rng_state"])
        if "python_rng_state" in payload:
            random.setstate(payload["python_rng_state"])
        cuda_state = payload.get("cuda_rng_state")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_state)

    def _prepare_ae_inputs(self) -> None:
        images, img_masks, lang_tokens, lang_masks, state = self.preprocessed
        self.prefix = self.model._build_prefix_cache(
            images, img_masks, lang_tokens, lang_masks
        )
        shape = (
            state.shape[0],
            self.model.config.action_horizon,
            self.model.config.action_dim,
        )
        self.noise = self.model.sample_noise(shape, state.device)
        self._denoise_rng_state = torch.get_rng_state()
        self._denoise_cuda_rng_state = (
            torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        )

    def _require_prepared(self) -> None:
        if self.stage is None:
            raise RuntimeError("prepare() must be called before replay")
