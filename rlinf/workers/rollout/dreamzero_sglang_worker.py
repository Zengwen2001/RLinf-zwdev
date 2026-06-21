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
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, open_dict
from tianshou.data import Batch

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

    @staticmethod
    def _disable_groot_scheduler_compile() -> None:
        from groot.vla.model.dreamzero.modules.flow_unipc_multistep_scheduler import (
            FlowUniPCMultistepScheduler,
        )

        for name in ("multistep_uni_p_bh_update", "multistep_uni_c_bh_update"):
            method = getattr(FlowUniPCMultistepScheduler, name)
            original = getattr(method, "__wrapped__", None)
            if original is not None:
                setattr(FlowUniPCMultistepScheduler, name, original)

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

        self.dst_ranks = {"eval": self._setup_dst_ranks(self.total_num_eval_envs)}
        self.src_ranks = {"eval": self._setup_src_ranks(self.total_num_eval_envs)}
        self.log_info(f"Rollout worker initialized with dst_ranks: {self.dst_ranks}")
        self.log_info(f"Rollout worker initialized with src_ranks: {self.src_ranks}")
        self.setup_sample_params()

    def _init_sglang_pipeline(self, model_cfg: DictConfig) -> None:
        self._disable_groot_scheduler_compile()

        from sglang.multimodal_gen.runtime.distributed.parallel_state import (
            maybe_init_distributed_environment_and_model_parallel,
            model_parallel_is_initialized,
        )
        from sglang.multimodal_gen.configs.pipeline_configs.dreamzero import (
            DreamZeroPipelineConfig,
        )
        from sglang.multimodal_gen.configs.sample.dreamzero import (
            DreamZeroSamplingParams,
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
        max_sessions = int(sglang_cfg.get("max_sessions", self.total_num_eval_envs))
        num_inference_steps = int(sglang_cfg.get("num_inference_steps", 16))
        pipeline_config = DreamZeroPipelineConfig(
            dreamzero_compile_components=bool(
                sglang_cfg.get("compile_components", True)
            ),
            dreamzero_sequence_parallel_size=int(sglang_cfg.get("sp_size", 1)),
            dreamzero_max_sessions=max_sessions,
            cfg_scale=float(sglang_cfg.get("cfg_scale", 5.0)),
            embodiment_tag=str(model_cfg.embodiment_tag),
            action_horizon=int(model_cfg.action_horizon),
            num_inference_steps=num_inference_steps,
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
            tp_size=int(sglang_cfg.get("tp_size", 1)),
            cfg_parallel_degree=int(sglang_cfg.get("cfg_parallel_degree", 1)),
            component_paths={
                "dreamzero_dit": str(model_cfg.model_path),
                # The VAE loader reads runtime config from config.json before it
                # resolves .pth weights, so pass the full DreamZero checkpoint dir.
                "dreamzero_vae": str(model_cfg.model_path),
                "dreamzero_text_encoder": str(model_cfg.text_encoder_pretrained_path),
                "dreamzero_image_encoder": str(model_cfg.image_encoder_pretrained_path),
            },
            log_level=str(sglang_cfg.get("log_level", "info")),
        )
        self._dreamzero_sampling_params_cls = DreamZeroSamplingParams
        self._sglang_server_args = server_args
        set_global_server_args(server_args)
        self._init_sglang_single_rank_parallel_state(
            maybe_init_distributed_environment_and_model_parallel,
            model_parallel_is_initialized,
            server_args,
            sglang_cfg,
        )
        self.sglang_pipeline = DreamZeroPipeline(str(model_cfg.model_path), server_args)

    def _init_sglang_single_rank_parallel_state(
        self,
        maybe_init_distributed_environment_and_model_parallel,
        model_parallel_is_initialized,
        server_args,
        sglang_cfg,
    ) -> None:
        if model_parallel_is_initialized():
            return
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
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

    def _init_eval_session_state(self) -> None:
        sglang_cfg = self.cfg.rollout.get("sglang", {})
        self._eval_predict_calls = 0
        self._debug_dump_count = 0
        self._debug_sessions = bool(sglang_cfg.get("debug_sessions", False))
        self._seed = int(sglang_cfg.get("seed", 1140))
        self._chunks_per_episode = max(
            1,
            int(self.cfg.env.eval.max_episode_steps)
            // int(self.model_cfg.num_action_chunks),
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
        session_id: str,
        reset_session: bool,
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
                "session_id": session_id,
                "reset_session": reset_session,
                "converted_obs": self._debug_to_cpu(converted_obs),
                "normalized_input": self._debug_to_cpu(normalized_input),
                "normalized_action": self._debug_to_cpu(normalized_action),
                "action": np.asarray(actions).copy(),
            },
            path,
        )
        self._debug_dump_count += 1

    def _session_id_for_batch(self) -> str:
        return f"rlinf-eval-rank{self._rank}-batch"

    def _should_reset_session(self) -> bool:
        return self._eval_predict_calls % self._chunks_per_episode == 0

    def _active_session_count(self) -> int | None:
        session_store = getattr(self.sglang_pipeline, "session_store", None)
        if session_store is None:
            return None
        return session_store.get_active_count()

    def _build_request(
        self,
        normalized_input,
        original_obs,
        *,
        session_id: str,
        reset_session: bool,
    ):
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

        sampling_params = self._dreamzero_sampling_params_cls(
            num_inference_steps=int(
                self._sglang_server_args.pipeline_config.num_inference_steps
            ),
            guidance_scale=float(self._sglang_server_args.pipeline_config.cfg_scale),
            seed=self._seed,
            session_id=session_id,
            reset_session=reset_session,
            embodiment_tag=self.action_adapter.embodiment_tag,
            action_horizon=int(self.model_cfg.action_horizon),
            relative_action_per_horizon=bool(
                self.model_cfg.get("relative_action_per_horizon", False)
            ),
        )
        return Req(
            sampling_params=sampling_params,
            extra={
                "dreamzero_normalized_input": normalized_input,
                "dreamzero_original_obs": original_obs,
                "dreamzero_unapply": self.action_adapter.unapply,
                "dreamzero_session_id": session_id,
                "dreamzero_reset_session": reset_session,
            },
            suppress_logs=True,
        )

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
                value[index : index + 1]
                if value.shape[:1] == (batch_size,)
                else value
            )
        if isinstance(value, np.ndarray):
            return (
                value[index : index + 1]
                if value.shape[:1] == (batch_size,)
                else value
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
    ) -> np.ndarray:
        session_id = self._session_id_for_batch()
        reset_session = self._should_reset_session()
        req = self._build_request(
            normalized_input,
            converted_obs,
            session_id=session_id,
            reset_session=reset_session,
        )
        req = self.sglang_pipeline.forward(req, self._sglang_server_args)
        self._eval_predict_calls += 1

        action_output = req.output
        if isinstance(action_output, Batch) and hasattr(action_output, "action"):
            action_output = action_output.action
        elif isinstance(action_output, Batch) and hasattr(action_output, "act"):
            action_output = self.action_adapter.actions_from_unapply(action_output.act)
        if torch.is_tensor(action_output):
            action_output = action_output.detach().cpu().float().numpy()
        elif isinstance(action_output, dict):
            action_output = self.action_adapter.actions_from_unapply(action_output)
        actions = np.asarray(action_output)
        batch_size = self._infer_batch_size(normalized_input)
        normalized_action = getattr(req, "dreamzero_action_pred", None)
        for slot in range(batch_size):
            self._maybe_dump_rollout_debug(
                slot=slot,
                session_id=session_id,
                reset_session=reset_session,
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

        converted_obs = self.action_adapter.observation_convert(env_obs)
        normalized_input = self.action_adapter.normalize_obs(converted_obs)
        active_before = self._active_session_count()
        actions = self._run_batch_request(normalized_input, converted_obs)
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

    def offload_model(self):
        raise NotImplementedError("DreamZero sglang rollout worker does not offload.")

    def reload_model(self):
        raise NotImplementedError("DreamZero sglang rollout worker does not offload.")
