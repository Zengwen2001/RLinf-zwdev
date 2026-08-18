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

"""Structural contracts for production profiling stage coverage."""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _find_function(
    path: str, function_name: str, class_name: str | None = None
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse((REPO_ROOT / path).read_text())
    body = tree.body
    if class_name is not None:
        target_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        body = target_class.body
    return next(
        node
        for node in body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )


def _function_stages(path: str, class_name: str, *function_names: str) -> set[str]:
    tree = ast.parse((REPO_ROOT / path).read_text())
    target_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    functions = [
        node
        for node in target_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in function_names
    ]
    assert {node.name for node in functions} == set(function_names)
    return {
        node.attr
        for function in functions
        for node in ast.walk(function)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "ProfileStage"
    }


def _profile_range_bodies(
    path: str, function_name: str, stage_name: str, class_name: str | None = None
) -> list[list[ast.stmt]]:
    function = _find_function(path, function_name, class_name)
    bodies = []
    for node in ast.walk(function):
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        for item in node.items:
            expression = item.context_expr
            if not (
                isinstance(expression, ast.Call)
                and isinstance(expression.func, ast.Name)
                and expression.func.id == "profile_range"
                and expression.args
            ):
                continue
            stage = expression.args[0]
            if (
                isinstance(stage, ast.Attribute)
                and isinstance(stage.value, ast.Name)
                and stage.value.id == "ProfileStage"
                and stage.attr == stage_name
            ):
                bodies.append(node.body)
    assert bodies, f"{function_name} has no range for {stage_name}"
    return bodies


def _range_call_names(
    path: str, function_name: str, stage_name: str, class_name: str | None = None
) -> set[str]:
    calls = set()
    for body in _profile_range_bodies(path, function_name, stage_name, class_name):
        for statement in body:
            for node in ast.walk(statement):
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)
    return calls


def _range_contains_loop(
    path: str, function_name: str, stage_name: str, class_name: str | None = None
) -> bool:
    return any(
        isinstance(node, (ast.For, ast.AsyncFor, ast.While))
        for body in _profile_range_bodies(path, function_name, stage_name, class_name)
        for statement in body
        for node in ast.walk(statement)
    )


def test_async_ppo_actor_has_training_stage_ranges():
    stages = _function_stages(
        "rlinf/workers/actor/async_ppo_fsdp_worker.py",
        "AsyncPPOEmbodiedFSDPActor",
        "run_training",
    )
    assert stages == {
        "ACTOR_BATCH_PREP",
        "ACTOR_H2D",
        "ACTOR_FORWARD",
        "ACTOR_LOSS",
        "ACTOR_BACKWARD_NO_SYNC",
        "ACTOR_BACKWARD_SYNC",
        "ACTOR_EMPTY_CACHE",
        "ACTOR_OPTIMIZER",
        "ACTOR_METRICS_REDUCE",
    }


def test_sync_ppo_actor_uses_matching_training_and_weight_sync_ranges():
    training_stages = _function_stages(
        "rlinf/workers/actor/fsdp_actor_worker.py",
        "EmbodiedFSDPActor",
        "run_training",
        "train_micro_batch",
        "_train_sft_epoch",
    )
    assert training_stages == {
        "ACTOR_BATCH_PREP",
        "ACTOR_H2D",
        "ACTOR_FORWARD",
        "ACTOR_LOSS",
        "ACTOR_BACKWARD_NO_SYNC",
        "ACTOR_BACKWARD_SYNC",
        "ACTOR_EMPTY_CACHE",
        "ACTOR_OPTIMIZER",
        "ACTOR_METRICS_REDUCE",
    }
    assert _function_stages(
        "rlinf/workers/actor/fsdp_actor_worker.py",
        "EmbodiedFSDPActor",
        "sync_model_to_rollout",
    ) == {"WEIGHT_SYNC_ACTOR_SEND"}
    actor_path = "rlinf/workers/actor/fsdp_actor_worker.py"
    assert "model" in _range_call_names(
        actor_path, "train_micro_batch", "ACTOR_FORWARD", "EmbodiedFSDPActor"
    )
    assert "policy_loss" in _range_call_names(
        actor_path, "train_micro_batch", "ACTOR_LOSS", "EmbodiedFSDPActor"
    )
    assert "model" in _range_call_names(
        actor_path, "_train_sft_epoch", "ACTOR_FORWARD", "EmbodiedFSDPActor"
    )


def test_rollout_worker_has_pipeline_and_weight_apply_ranges():
    stages = _function_stages(
        "rlinf/workers/rollout/hf/huggingface_worker.py",
        "MultiStepRolloutWorker",
        "generate_one_epoch",
    )
    assert stages == {
        "ROLLOUT_RECV_OBS",
        "ROLLOUT_POSTPROCESS",
        "ROLLOUT_SEND_ACTION",
    }
    assert _function_stages(
        "rlinf/workers/rollout/hf/huggingface_worker.py",
        "MultiStepRolloutWorker",
        "sync_model_from_actor",
    ) == {"WEIGHT_SYNC_ROLLOUT_APPLY"}
    assert _function_stages(
        "rlinf/workers/rollout/hf/huggingface_worker.py",
        "MultiStepRolloutWorker",
        "_build_policy_output",
    ) == {"ROLLOUT_POSTPROCESS"}
    assert _function_stages(
        "rlinf/workers/rollout/hf/huggingface_worker.py",
        "MultiStepRolloutWorker",
        "get_bootstrap_values",
    ) == {"ROLLOUT_BOOTSTRAP_VALUE"}
    assert _function_stages(
        "rlinf/workers/rollout/hf/async_huggingface_worker.py",
        "AsyncMultiStepRolloutWorker",
        "decoupled_generate_one_epoch",
    ) == {
        "ROLLOUT_RECV_OBS",
        "ROLLOUT_SEND_ACTION",
    }


def test_legacy_openpi_has_model_phase_ranges():
    stages = _function_stages(
        "rlinf/models/embodiment/openpi/openpi_action_model.py",
        "OpenPi0ForRLActionPrediction",
        "predict_action_batch",
        "sample_actions",
        "_sample_actions_with_prefix_cache",
    )
    assert stages == {
        "ROLLOUT_OBS_PREPROCESS",
        "ROLLOUT_VLM_PREFIX",
        "ROLLOUT_AE_DENOISE",
        "ROLLOUT_POSTPROCESS",
    }
    openpi_path = "rlinf/models/embodiment/openpi/openpi_action_model.py"
    assert _range_contains_loop(
        openpi_path,
        "_sample_actions_with_prefix_cache",
        "ROLLOUT_AE_DENOISE",
        "OpenPi0ForRLActionPrediction",
    )


def test_env_worker_has_interaction_stage_ranges():
    stages = _function_stages(
        "rlinf/workers/env/env_worker.py",
        "EnvWorker",
        "_send_train_bootstrap",
        "_run_interact_once",
    )
    assert stages == {
        "ENV_RECV_ACTION",
        "ENV_BOOTSTRAP_REWARD",
        "ENV_TRAJECTORY_APPEND",
        "ENV_STEP",
        "ENV_SEND_OBS",
    }
