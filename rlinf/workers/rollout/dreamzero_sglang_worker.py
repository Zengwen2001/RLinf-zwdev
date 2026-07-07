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
import os
import socket
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, open_dict
from tianshou.data import Batch
from tqdm import tqdm

from rlinf.data.datasets.dreamzero.data_transforms import (
    build_dreamzero_composed_transform,
    collect_dreamzero_dataset_keys,
    convert_rollout_env_obs,
    format_training_prompt,
    load_dreamzero_dataset_metadata,
    normalize_instruction_text,
    rollout_obs_layout_for_embodiment,
)
from rlinf.models.embodiment.dreamzero.dreamzero_policy import DreamZeroPolicy
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


_DREAMZERO_ROLLOUT_METADATA_KEYS = (
    "dreamzero_env_rank",
    "dreamzero_local_env_id",
    "dreamzero_episode_id",
    "dreamzero_reset_mask",
)


class _DreamZeroActionAdapter:
    """Reuse RLinf DreamZero transforms without constructing the HF model."""

    def __init__(self, cfg: DictConfig):
        tokenizer_path = cfg.get("tokenizer_path", "google/umt5-xxl")
        self.embodiment_tag = str(cfg.embodiment_tag)
        self._rollout_obs_layout = rollout_obs_layout_for_embodiment(
            self.embodiment_tag
        )
        self.data_transforms = build_dreamzero_composed_transform(cfg, tokenizer_path)
        self.data_transforms.set_metadata(load_dreamzero_dataset_metadata(cfg))
        self.data_transforms.eval()
        _, _, action_keys, _ = collect_dreamzero_dataset_keys(
            self.data_transforms, self.embodiment_tag
        )
        self._action_keys = tuple(action_keys)
        self._dream_transform = self.data_transforms.transforms[-1]
        self.config = SimpleNamespace(
            data_transforms=self.data_transforms,
            relative_action=bool(cfg.get("relative_action", False)),
            relative_action_per_horizon=bool(
                cfg.get("relative_action_per_horizon", False)
            ),
            relative_action_keys=list(cfg.get("relative_action_keys") or []),
        )

    def observation_convert(self, env_obs: dict[str, Any]) -> dict[str, Any]:
        return convert_rollout_env_obs(self.embodiment_tag, env_obs)

    def normalize_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        normalized_input = self.data_transforms(obs)
        if isinstance(normalized_input, Batch):
            normalized_input = normalized_input.__getstate__()
        normalized_input = dict(normalized_input)
        self._align_rollout_prompt_tokens(obs, normalized_input)
        return normalized_input

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if torch.is_tensor(value):
            value = value.detach().cpu()
            if value.ndim == 0:
                return [value.item()]
            return value.flatten().tolist()
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return [value.item()]
            return value.reshape(-1).tolist()
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]

    def _align_rollout_prompt_tokens(
        self, obs: dict[str, Any], normalized_input: dict[str, Any]
    ) -> None:
        tasks = obs.get("annotation.task")
        if tasks is None or "text" not in normalized_input:
            return

        embodiment_ids = self._as_list(normalized_input["embodiment_id"])
        task_list = self._as_list(tasks)
        if len(task_list) == 1 and len(embodiment_ids) > 1:
            task_list = task_list * len(embodiment_ids)

        texts = [
            format_training_prompt(
                normalize_instruction_text(task),
                int(embodiment_id),
                self._dream_transform.embodiment_tag_mapping,
            )
            for task, embodiment_id in zip(task_list, embodiment_ids)
        ]
        ids, mask = self._dream_transform.tokenizer(
            texts,
            return_mask=True,
            add_special_tokens=True,
        )
        normalized_input["text"] = ids
        normalized_input["text_attention_mask"] = mask

    def unapply(self, batch: Batch, obs: dict[str, Any] | None = None, **kwargs):
        return DreamZeroPolicy.unapply(self, batch, obs=obs, **kwargs)

    def actions_from_unapply(self, act_dict: dict[str, Any]) -> np.ndarray:
        actions = DreamZeroPolicy._actions_from_unapply(self, act_dict)
        if self._rollout_obs_layout.binarize_gripper:
            actions[..., -1] = np.where(actions[..., -1] > 0, 1.0, -1.0).astype(
                actions.dtype
            )
        return actions


class DreamZeroSGLangRolloutWorker(MultiStepRolloutWorker):
    """Evaluation-only DreamZero rollout worker backed by sglang's in-process pipeline."""

    def init_worker(self):
        if not self.only_eval:
            raise NotImplementedError(
                "DreamZero sglang rollout worker currently supports eval only."
            )

        rollout_model_config = copy.deepcopy(self.model_cfg)
        with open_dict(rollout_model_config):
            rollout_model_config.precision = self.cfg.rollout.model.precision
            rollout_model_config.model_path = self.cfg.rollout.model.model_path

        self.action_adapter = _DreamZeroActionAdapter(rollout_model_config)
        self._init_sglang_pipeline(rollout_model_config)
        self._init_eval_session_state()

        self.setup_sample_params()

    def _init_sglang_pipeline(self, model_cfg: DictConfig) -> None:
        from sglang.multimodal_gen.configs.pipeline_configs.dreamzero import (
            DreamZeroPipelineConfig,
        )
        from sglang.multimodal_gen.configs.sample.dreamzero import (
            DreamZeroSamplingParams,
        )
        from sglang.multimodal_gen.runtime.distributed.parallel_state import (
            maybe_init_distributed_environment_and_model_parallel,
            model_parallel_is_initialized,
        )
        from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
            DiffGenerator,
        )
        from sglang.multimodal_gen.runtime.pipelines.dreamzero_pipeline import (
            DreamZeroPipeline,
        )
        from sglang.multimodal_gen.runtime.server_args import (
            Backend,
            ServerArgs,
            set_global_server_args,
        )

        action_head_cfg = model_cfg.action_head_cfg.config
        sglang_cfg = self.cfg.rollout.get("sglang", {})
        tp_size = int(sglang_cfg.get("tp_size", 1))
        sp_size = int(sglang_cfg.get("sp_size", 1))
        cfg_parallel_degree = int(sglang_cfg.get("cfg_parallel_degree", 1))
        sglang_world_size = tp_size * sp_size * cfg_parallel_degree
        max_sessions = int(sglang_cfg.get("max_sessions", self.total_num_eval_envs))
        num_inference_steps = int(sglang_cfg.get("num_inference_steps", 16))
        port_offset = int(getattr(self, "_rank", 0)) * int(
            sglang_cfg.get("port_stride", 100)
        )
        http_port = int(sglang_cfg.get("port", 30000)) + port_offset
        scheduler_port = int(sglang_cfg.get("scheduler_port", 5555)) + port_offset
        master_port = int(sglang_cfg.get("master_port", 30005)) + port_offset
        default_pipeline_config = DreamZeroPipelineConfig()
        pipeline_config = DreamZeroPipelineConfig(
            dreamzero_compile_components=bool(
                sglang_cfg.get("compile_components", True)
            ),
            dreamzero_sequence_parallel_size=sp_size,
            dreamzero_max_sessions=max_sessions,
            cfg_scale=float(sglang_cfg.get("cfg_scale", 5.0)),
            embodiment_tag=str(model_cfg.embodiment_tag),
            action_horizon=int(model_cfg.action_horizon),
            num_inference_steps=num_inference_steps,
            dynamic_cache_schedule=bool(
                sglang_cfg.get(
                    "dynamic_cache_schedule",
                    default_pipeline_config.dynamic_cache_schedule,
                )
            ),
            num_frames=int(action_head_cfg.num_frames),
            synthetic_height=int(model_cfg.target_video_height),
            synthetic_width=int(model_cfg.target_video_width),
            tile_size_height=int(action_head_cfg.get("tile_size_height", 34)),
            tile_size_width=int(action_head_cfg.get("tile_size_width", 34)),
            tile_stride_height=int(action_head_cfg.get("tile_stride_height", 18)),
            tile_stride_width=int(action_head_cfg.get("tile_stride_width", 16)),
            tiled=bool(action_head_cfg.get("tiled", False)),
            relative_action_per_horizon=bool(
                model_cfg.get("relative_action_per_horizon", False)
            ),
        )
        server_args = ServerArgs(
            model_path=str(model_cfg.model_path),
            backend=Backend.SGLANG,
            pipeline_class_name="DreamZeroPipeline",
            pipeline_config=pipeline_config,
            attention_backend=sglang_cfg.get("attention_backend", "TORCH_SDPA"),
            num_gpus=sglang_world_size,
            tp_size=tp_size,
            sp_degree=sp_size,
            cfg_parallel_degree=cfg_parallel_degree,
            component_paths={
                "dreamzero_dit": str(model_cfg.model_path),
                # The VAE loader reads runtime config from config.json before it
                # resolves .pth weights, so pass the full DreamZero checkpoint dir.
                "dreamzero_vae": str(model_cfg.model_path),
                "dreamzero_text_encoder": str(model_cfg.text_encoder_pretrained_path),
                "dreamzero_image_encoder": str(model_cfg.image_encoder_pretrained_path),
            },
            log_level=str(sglang_cfg.get("log_level", "info")),
            port=http_port,
            scheduler_port=scheduler_port,
            master_port=master_port,
        )
        self._dreamzero_sampling_params_cls = DreamZeroSamplingParams
        self._sglang_server_args = server_args
        set_global_server_args(server_args)
        self._sglang_use_scheduler = sglang_world_size > 1
        self._sglang_generator = None
        self.sglang_pipeline = None
        if self._sglang_use_scheduler:
            self._validate_visible_sglang_devices(sglang_world_size)
            self._sglang_generator = DiffGenerator.from_server_args(
                server_args,
                local_mode=True,
            )
            self._check_local_scheduler_ready()
        else:
            self._init_sglang_single_rank_parallel_state(
                maybe_init_distributed_environment_and_model_parallel,
                model_parallel_is_initialized,
                server_args,
                sglang_cfg,
            )
            self.sglang_pipeline = DreamZeroPipeline(
                str(model_cfg.model_path), server_args
            )

    def _validate_visible_sglang_devices(self, sglang_world_size: int) -> None:
        # WorkerGroup sets VISIBLE_DEVICES before accelerator-specific helpers
        # translate it to CUDA_VISIBLE_DEVICES.
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get(
            "VISIBLE_DEVICES"
        )
        if not visible_devices:
            return
        visible_count = len([item for item in visible_devices.split(",") if item])
        if visible_count < sglang_world_size:
            raise RuntimeError(
                "DreamZero sglang parallel eval requires at least "
                f"{sglang_world_size} visible device(s), but only sees "
                f"{visible_count}: {visible_devices!r}. Set "
                "`rollout.tensor_parallel_size` and placement so one rollout "
                "worker owns the full sglang parallel group."
            )

    def _init_sglang_single_rank_parallel_state(
        self,
        maybe_init_distributed_environment_and_model_parallel,
        model_parallel_is_initialized,
        server_args,
        sglang_cfg,
    ) -> None:
        if model_parallel_is_initialized():
            return
        if (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() != 1
        ):
            raise RuntimeError(
                "sglang in-process DreamZero eval expects a per-rollout-rank "
                "single-process torch distributed group, but torch.distributed is "
                f"already initialized with world_size={torch.distributed.get_world_size()}."
            )

        host = "127.0.0.1"
        port = self._find_free_port()
        local_rank = int(getattr(self, "_local_rank", os.environ.get("LOCAL_RANK", 0)))
        env_keys = (
            "MASTER_ADDR",
            "MASTER_PORT",
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
        )
        saved_env = {key: os.environ.get(key) for key in env_keys}
        try:
            os.environ.update(
                {
                    "MASTER_ADDR": host,
                    "MASTER_PORT": str(port),
                    "RANK": "0",
                    "WORLD_SIZE": "1",
                    "LOCAL_RANK": str(local_rank),
                    "LOCAL_WORLD_SIZE": "1",
                }
            )
            maybe_init_distributed_environment_and_model_parallel(
                tp_size=int(sglang_cfg.get("tp_size", 1)),
                sp_size=int(sglang_cfg.get("sp_size", 1)),
                cfg_degree=int(sglang_cfg.get("cfg_parallel_degree", 1)),
                dp_size=1,
                distributed_init_method=f"tcp://{host}:{port}",
                dist_timeout=getattr(server_args, "dist_timeout", None),
            )
        finally:
            for key, value in saved_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    @staticmethod
    def _check_local_scheduler_ready() -> None:
        from sglang.multimodal_gen.runtime.scheduler_client import (
            sync_scheduler_client,
        )

        last_error = None
        for _ in range(60):
            try:
                if sync_scheduler_client.ping():
                    return
            except Exception as exc:
                last_error = exc
            time.sleep(1)
        raise ConnectionError(
            "DreamZero local sglang scheduler did not become ready within 60s."
        ) from last_error

    def _init_eval_session_state(self) -> None:
        sglang_cfg = self.cfg.rollout.get("sglang", {})
        self._eval_predict_calls = 0
        self._debug_dump_count = 0
        self._debug_sessions = bool(sglang_cfg.get("debug_sessions", False))
        self._debug_batch_print = bool(
            sglang_cfg.get("debug_batch", False)
        ) or bool(int(os.environ.get("DREAMZERO_SGLANG_WORKER_DEBUG_BATCH", "0") or 0))
        self._seed = int(sglang_cfg.get("seed", 1140))
        self._chunks_per_episode = max(
            1,
            int(self.cfg.env.eval.max_episode_steps)
            // int(self.model_cfg.num_action_chunks),
        )

    @staticmethod
    def _debug_shape(value: Any) -> Any:
        if torch.is_tensor(value):
            return tuple(value.shape)
        if isinstance(value, np.ndarray):
            return value.shape
        if isinstance(value, Mapping):
            return {
                key: DreamZeroSGLangRolloutWorker._debug_shape(item)
                for key, item in value.items()
            }
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return f"{type(value).__name__}[{len(value)}]"
        return type(value).__name__

    def _debug_batch_log(self, message: str) -> None:
        if self._debug_batch_print:
            print(
                f"[DreamZeroSGLangWorker batch-debug rank={self._rank} "
                f"pid={os.getpid()} call={self._eval_predict_calls}] {message}",
                flush=True,
            )

    @staticmethod
    def _debug_to_cpu(value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, Mapping):
            return {
                key: DreamZeroSGLangRolloutWorker._debug_to_cpu(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return type(value)(
                DreamZeroSGLangRolloutWorker._debug_to_cpu(item) for item in value
            )
        return value

    def _maybe_dump_rollout_debug(
        self,
        *,
        slot: int,
        logical_session_id: str,
        reset_mask: bool,
        normalized_input: dict[str, Any],
        converted_obs: dict[str, Any],
        normalized_action: Any,
        actions: np.ndarray,
    ) -> None:
        dump_dir = os.environ.get("DREAMZERO_ROLLOUT_DEBUG_DUMP_DIR")
        limit = int(os.environ.get("DREAMZERO_ROLLOUT_DEBUG_DUMP_LIMIT", "0") or 0)
        if not dump_dir or limit <= 0 or self._debug_dump_count >= limit:
            return

        prefix = os.environ.get("DREAMZERO_ROLLOUT_DEBUG_DUMP_PREFIX", "sglang")
        path = (
            Path(dump_dir)
            / f"{prefix}_rank{self._rank}_pid{os.getpid()}_call{self._debug_dump_count:04d}_slot{slot}.pt"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "backend": "sglang",
                "rank": self._rank,
                "pid": os.getpid(),
                "call": self._debug_dump_count,
                "slot": slot,
                "logical_session_id": logical_session_id,
                "reset_mask": reset_mask,
                "converted_obs": self._debug_to_cpu(converted_obs),
                "normalized_input": self._debug_to_cpu(normalized_input),
                "normalized_action": self._debug_to_cpu(normalized_action),
                "action": np.asarray(actions).copy(),
            },
            path,
        )
        self._debug_dump_count += 1

    @staticmethod
    def _as_metadata_list(value: Any, batch_size: int, name: str) -> list[Any]:
        if torch.is_tensor(value):
            value = value.detach().cpu()
            if value.ndim == 0:
                values = [value.item()]
            else:
                values = value.flatten().tolist()
        elif isinstance(value, np.ndarray):
            if value.ndim == 0:
                values = [value.item()]
            else:
                values = value.reshape(-1).tolist()
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            values = list(value)
        else:
            values = [value]

        if len(values) == 1 and batch_size > 1:
            values = values * batch_size
        if len(values) != batch_size:
            raise ValueError(
                f"DreamZero rollout metadata {name!r} has {len(values)} values, "
                f"expected batch_size={batch_size}."
            )
        return values

    def _split_rollout_metadata(
        self, env_obs: Any
    ) -> tuple[Any, dict[str, Any]]:
        if not isinstance(env_obs, Mapping):
            return env_obs, {}
        metadata = {
            key: env_obs[key]
            for key in _DREAMZERO_ROLLOUT_METADATA_KEYS
            if key in env_obs
        }
        if not metadata:
            return env_obs, {}
        cleaned = {
            key: value
            for key, value in env_obs.items()
            if key not in _DREAMZERO_ROLLOUT_METADATA_KEYS
        }
        return cleaned, metadata

    @staticmethod
    def _merge_dreamzero_metadata_values(values: list[Any]) -> Any:
        first_non_none = next((value for value in values if value is not None), None)
        if first_non_none is None:
            return None
        if torch.is_tensor(first_non_none):
            return torch.cat(values, dim=0)
        if isinstance(first_non_none, np.ndarray):
            return np.concatenate(values, axis=0)
        if isinstance(first_non_none, list):
            return [item for value in values for item in value]
        return values

    @staticmethod
    def _merge_obs_batches(obs_batches: list[dict[str, Any]]) -> dict[str, Any]:
        merged = MultiStepRolloutWorker._merge_obs_batches(obs_batches)
        for key in _DREAMZERO_ROLLOUT_METADATA_KEYS:
            values = [obs_batch.get(key, None) for obs_batch in obs_batches]
            if any(value is not None for value in values):
                merged[key] = (
                    DreamZeroSGLangRolloutWorker._merge_dreamzero_metadata_values(
                        values
                    )
                )
        return merged

    @staticmethod
    def _obs_with_payload_metadata(env_output: dict[str, Any]) -> dict[str, Any]:
        obs = env_output["obs"]
        metadata = {
            key: env_output[key]
            for key in _DREAMZERO_ROLLOUT_METADATA_KEYS
            if key in env_output
        }
        if not metadata:
            return obs
        if not isinstance(obs, Mapping):
            raise TypeError(
                "DreamZero sglang eval metadata requires dict observations, "
                f"got {type(obs)!r}."
            )
        return {**obs, **metadata}

    def _session_ids_for_batch(
        self, batch_size: int, metadata: dict[str, Any]
    ) -> list[str]:
        missing = [
            key
            for key in ("dreamzero_env_rank", "dreamzero_local_env_id")
            if key not in metadata
        ]
        if missing:
            raise ValueError(
                "DreamZero sglang eval requires env metadata for session cache: "
                f"missing {missing}"
            )
        env_ranks = self._as_metadata_list(
            metadata["dreamzero_env_rank"], batch_size, "dreamzero_env_rank"
        )
        local_env_ids = self._as_metadata_list(
            metadata["dreamzero_local_env_id"],
            batch_size,
            "dreamzero_local_env_id",
        )
        session_ids = [
            f"rlinf-eval-env{int(env_rank)}-slot{int(local_env_id)}"
            for env_rank, local_env_id in zip(env_ranks, local_env_ids)
        ]

        if len(set(session_ids)) != len(session_ids):
            raise ValueError(
                "DreamZero sglang eval received duplicate session ids in one "
                f"batch: {session_ids}"
            )
        return session_ids

    def _reset_mask_for_batch(
        self, batch_size: int, metadata: dict[str, Any]
    ) -> list[bool]:
        if "dreamzero_reset_mask" not in metadata:
            raise ValueError(
                "DreamZero sglang eval requires dreamzero_reset_mask for "
                "session cache"
            )
        reset_values = self._as_metadata_list(
            metadata["dreamzero_reset_mask"], batch_size, "dreamzero_reset_mask"
        )
        return [bool(value) for value in reset_values]

    def _active_session_count(self) -> int | None:
        cache_manager = getattr(self.sglang_pipeline, "cache_manager", None)
        if cache_manager is None:
            return None
        return cache_manager.get_active_count()

    def _build_request(
        self,
        normalized_input,
        original_obs,
        *,
        session_ids: list[str],
        reset_mask: list[bool],
    ):
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

        sampling_params = self._dreamzero_sampling_params_cls(
            num_inference_steps=int(
                self._sglang_server_args.pipeline_config.num_inference_steps
            ),
            guidance_scale=float(self._sglang_server_args.pipeline_config.cfg_scale),
            seed=self._seed,
            embodiment_tag=self.action_adapter.embodiment_tag,
            action_horizon=int(self.model_cfg.action_horizon),
            relative_action_per_horizon=bool(
                self.model_cfg.get("relative_action_per_horizon", False)
            ),
        )
        extra = {
            "dreamzero_normalized_input": normalized_input,
            "dreamzero_original_obs": original_obs,
            "dreamzero_session_ids": session_ids,
            "dreamzero_reset_mask": reset_mask,
        }
        return Req(
            sampling_params=sampling_params,
            extra=extra,
            suppress_logs=True,
        )

    def _forward_sglang_request(self, req):
        self._debug_batch_log("enter sglang forward")
        if self._sglang_use_scheduler:
            # DiffGenerator.generate() performs image/video prompt parsing and
            # file output handling. DreamZero rollout already builds a Req with
            # normalized tensors, so we call the scheduler client boundary
            # directly and keep this contract local to the worker.
            response = self._sglang_generator._send_to_scheduler_and_wait_for_response(
                [req]
            )
            error = getattr(response, "error", None)
            if error is not None:
                raise RuntimeError(
                    f"DreamZero sglang scheduler request failed: {error}"
                )
            self._debug_batch_log(
                "exit scheduler forward output_shape="
                f"{self._debug_shape(getattr(response, 'output', response))}"
            )
            return response
        response = self.sglang_pipeline.forward(req, self._sglang_server_args)
        self._debug_batch_log(
            "exit in-process forward output_shape="
            f"{self._debug_shape(getattr(response, 'output', response))}"
        )
        return response

    def _extract_action_output(self, response, converted_obs):
        action_output = getattr(response, "output", response)
        if isinstance(action_output, list) and len(action_output) == 1:
            action_output = action_output[0]
        if isinstance(action_output, Batch) and hasattr(action_output, "action"):
            action_output = action_output.action
        if torch.is_tensor(action_output):
            action_output = self.action_adapter.unapply(
                Batch(normalized_action=action_output.detach().cpu().float()),
                obs=converted_obs,
            )
        return self._as_numeric_action_array(action_output)

    def _as_numeric_action_array(self, action_output: Any) -> np.ndarray:
        if isinstance(action_output, Batch):
            if hasattr(action_output, "action"):
                action_output = action_output.action
            elif hasattr(action_output, "act"):
                action_output = self.action_adapter.actions_from_unapply(
                    action_output.act
                )
        if isinstance(action_output, dict):
            action_output = self.action_adapter.actions_from_unapply(action_output)
        if torch.is_tensor(action_output):
            action_output = action_output.detach().cpu().float().numpy()
        if isinstance(action_output, list):
            action_output = [
                item.detach().cpu().float().numpy() if torch.is_tensor(item) else item
                for item in action_output
            ]
        actions = np.asarray(action_output)
        if actions.dtype == np.dtype("O") and actions.size == 1:
            actions = np.asarray(actions.item())
        if actions.dtype == np.dtype("O"):
            raise TypeError(
                "DreamZero sglang action output is not numeric. "
                f"type={type(action_output)!r}, dtype={actions.dtype}, "
                f"shape={actions.shape}"
            )
        return actions.astype(np.float32, copy=False)

    @staticmethod
    def _infer_batch_size(value: Any) -> int:
        if torch.is_tensor(value) or isinstance(value, np.ndarray):
            return int(value.shape[0])
        if isinstance(value, Mapping):
            for item in value.values():
                try:
                    return DreamZeroSGLangRolloutWorker._infer_batch_size(item)
                except (TypeError, IndexError):
                    continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return len(value)
        raise TypeError("Unable to infer DreamZero rollout batch size.")

    @staticmethod
    def _slice_batch_value(value: Any, index: int, batch_size: int) -> Any:
        if torch.is_tensor(value):
            return (
                value[index : index + 1] if value.shape[:1] == (batch_size,) else value
            )
        if isinstance(value, np.ndarray):
            return (
                value[index : index + 1] if value.shape[:1] == (batch_size,) else value
            )
        if isinstance(value, Mapping):
            return {
                key: DreamZeroSGLangRolloutWorker._slice_batch_value(
                    item, index, batch_size
                )
                for key, item in value.items()
            }
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) == batch_size:
                return [value[index]]
        return value

    def _run_batch_request(
        self,
        normalized_input: dict[str, Any],
        converted_obs: dict[str, Any],
        session_metadata: dict[str, Any],
    ) -> np.ndarray:
        batch_size = self._infer_batch_size(normalized_input)
        session_ids = self._session_ids_for_batch(batch_size, session_metadata)
        reset_mask = self._reset_mask_for_batch(batch_size, session_metadata)
        self._debug_batch_log(
            f"build req batch_size={batch_size} reset_count={sum(reset_mask)} "
            f"session_ids={session_ids[:4]} "
            f"input_shapes={self._debug_shape(normalized_input)}"
        )
        req = self._build_request(
            normalized_input,
            converted_obs,
            session_ids=session_ids,
            reset_mask=reset_mask,
        )
        response = self._forward_sglang_request(req)
        self._eval_predict_calls += 1

        actions = self._extract_action_output(response, converted_obs)
        self._debug_batch_log(f"actions_shape={self._debug_shape(actions)}")
        normalized_action = getattr(response, "dreamzero_action_pred", None)
        if normalized_action is None and self._sglang_use_scheduler:
            normalized_action = getattr(response, "output", None)
        for slot in range(batch_size):
            self._maybe_dump_rollout_debug(
                slot=slot,
                logical_session_id=session_ids[slot],
                reset_mask=reset_mask[slot],
                normalized_input=self._slice_batch_value(
                    normalized_input, slot, batch_size
                ),
                converted_obs=self._slice_batch_value(converted_obs, slot, batch_size),
                normalized_action=self._slice_batch_value(
                    normalized_action, slot, batch_size
                ),
                actions=self._slice_batch_value(actions, slot, batch_size),
            )
        return actions

    def predict(self, env_obs, mode="eval", **kwargs):
        if mode != "eval":
            raise NotImplementedError(
                "DreamZero sglang rollout worker currently supports eval only."
            )

        rollout_env_obs, session_metadata = self._split_rollout_metadata(env_obs)
        converted_obs = self.action_adapter.observation_convert(rollout_env_obs)
        self._debug_batch_log(
            f"predict received env_obs_shape={self._debug_shape(env_obs)} "
            f"metadata_shape={self._debug_shape(session_metadata)} "
            f"converted_shape={self._debug_shape(converted_obs)}"
        )
        normalized_input = self.action_adapter.normalize_obs(converted_obs)
        active_before = self._active_session_count()
        actions = self._run_batch_request(
            normalized_input, converted_obs, session_metadata
        )
        if self._debug_sessions:
            self.log_info(
                "DreamZero sglang session "
                f"rank={self._rank} calls={self._eval_predict_calls} "
                f"active_before={active_before} "
                f"active_after={self._active_session_count()}"
            )

        flat = torch.as_tensor(actions, dtype=torch.float32).reshape(
            actions.shape[0], -1
        )
        result = {
            "prev_logprobs": torch.zeros_like(flat, dtype=torch.float32),
            "prev_values": torch.zeros((flat.shape[0], 1), dtype=torch.float32),
            "forward_inputs": {"action": flat.cpu()},
        }
        return actions, result

    async def evaluate(self, input_channel, output_channel):
        if self.enable_offload:
            self.reload_model()
        if self.env_decoupled_mode:
            while True:
                if self._use_delayed_per_env_receive():
                    env_output, split_sizes, routes = (
                        await self.recv_delayed_env_shards_with_timeout(
                            group_name=self.cfg.env.group_name,
                            channel=input_channel,
                            tag="rollout_results",
                            batch_size=self.eval_batch_size,
                            merge_fn=self._merge_obs_batches,
                            infer_batch_size_fn=self._infer_env_batch_size,
                            timeout_time=self.env_decoupled_recv_timeout_s,
                            recv_queue_size=self.rollout_queue_size,
                        )
                    )
                else:
                    env_output, split_sizes = (
                        await self.recv_from_and_record_batch_routes_with_timeout(
                            group_name=self.cfg.env.group_name,
                            channel=input_channel,
                            tag="rollout_results",
                            batch_size=self.eval_batch_size,
                            merge_fn=self._merge_obs_batches,
                            infer_batch_size_fn=self._infer_env_batch_size,
                            timeout_time=self.env_decoupled_recv_timeout_s,
                            recv_queue_size=self.rollout_queue_size,
                        )
                    )
                    routes = None
                actions, _ = self.predict(
                    self._obs_with_payload_metadata(env_output), mode="eval"
                )
                if isinstance(actions, torch.Tensor):
                    actions = actions.detach().cpu().contiguous()
                if routes is None:
                    send_work = self.send_to_recorded_batch_routes(
                        group_name=self.cfg.env.group_name,
                        channel=output_channel,
                        data=actions,
                        tag="rollout_results",
                        split_sizes=split_sizes,
                    )
                else:
                    send_work = self.send_to_delayed_env_shard_routes(
                        group_name=self.cfg.env.group_name,
                        channel=output_channel,
                        data=actions,
                        routes=routes,
                        tag="rollout_results",
                        split_sizes=split_sizes,
                    )
                await send_work.async_wait()
        else:
            for _ in tqdm(
                range(self.eval_rollout_epoch),
                desc="Evaluating Rollout Epochs",
                disable=(self._rank != 0),
            ):
                for _ in range(self.n_eval_chunk_steps):
                    for _ in range(self.num_pipeline_stages):
                        env_output = await self.recv_env_output(
                            input_channel=input_channel,
                            tag="eval_rollout_results",
                            batch_size=self.eval_batch_size,
                        )
                        actions, _ = self.predict(
                            self._obs_with_payload_metadata(env_output), mode="eval"
                        )
                        if isinstance(actions, torch.Tensor):
                            actions = actions.detach().cpu().contiguous()
                        self.send_rollout_result(
                            output_channel=output_channel,
                            rollout_result=actions,
                            tag="eval_rollout_results",
                            batch_size=self.eval_batch_size,
                        )

            if self.enable_offload:
                self.offload_model()

    def offload_model(self):
        raise NotImplementedError("DreamZero sglang rollout worker does not offload.")

    def reload_model(self):
        raise NotImplementedError("DreamZero sglang rollout worker does not offload.")

    def shutdown_sglang_backend(self):
        generator = getattr(self, "_sglang_generator", None)
        if generator is not None:
            generator.shutdown()
            self._sglang_generator = None
