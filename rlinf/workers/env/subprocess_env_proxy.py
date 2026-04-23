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

import traceback
from multiprocessing import get_context
from typing import Any

from omegaconf import OmegaConf

from rlinf.envs import get_env_cls
from rlinf.envs.wrappers import CollectEpisode, RecordVideo


def _build_env_with_wrappers(
    *,
    env_cfg_container: dict[str, Any],
    num_envs: int,
    seed_offset: int,
    total_num_processes: int,
    worker_info: Any,
):
    env_cfg = OmegaConf.create(env_cfg_container)
    env_cls = get_env_cls(env_cfg.env_type, env_cfg)
    env = env_cls(
        cfg=env_cfg,
        num_envs=num_envs,
        seed_offset=seed_offset,
        total_num_processes=total_num_processes,
        worker_info=worker_info,
    )
    if env_cfg.video_cfg.save_video:
        env = RecordVideo(env, env_cfg.video_cfg)
    if env_cfg.get("data_collection", None) and getattr(
        env_cfg.data_collection, "enabled", False
    ):
        env = CollectEpisode(
            env,
            save_dir=env_cfg.data_collection.save_dir,
            rank=worker_info.rank if worker_info is not None else 0,
            num_envs=num_envs,
            export_format=getattr(env_cfg.data_collection, "export_format", "pickle"),
            robot_type=getattr(env_cfg.data_collection, "robot_type", "panda"),
            fps=getattr(env_cfg.data_collection, "fps", 10),
            only_success=getattr(env_cfg.data_collection, "only_success", False),
            finalize_interval=getattr(
                env_cfg.data_collection, "finalize_interval", 100
            ),
        )
    return env


def _subprocess_env_worker(conn, init_payload: dict[str, Any]) -> None:
    env = None
    try:
        env = _build_env_with_wrappers(**init_payload)
        conn.send({"type": "ready"})

        while True:
            cmd, payload = conn.recv()
            if cmd == "call":
                method = getattr(env, payload["name"])
                result = method(*payload.get("args", ()), **payload.get("kwargs", {}))
                conn.send({"type": "ok", "result": result})
            elif cmd == "get_attr":
                conn.send({"type": "ok", "result": getattr(env, payload["name"])})
            elif cmd == "set_attr":
                setattr(env, payload["name"], payload["value"])
                conn.send({"type": "ok", "result": None})
            elif cmd == "close":
                if env is not None and hasattr(env, "close"):
                    env.close()
                conn.send({"type": "ok", "result": None})
                break
            else:
                raise NotImplementedError(f"Unknown subprocess env command: {cmd}")
    except EOFError:
        pass
    except Exception:
        try:
            conn.send({"type": "error", "traceback": traceback.format_exc()})
        except Exception:
            pass
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        conn.close()


class SubprocessEnvProxy:
    def __init__(
        self,
        *,
        env_cfg_container: dict[str, Any],
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: Any,
    ) -> None:
        self._ctx = get_context("spawn")
        self._parent_conn, child_conn = self._ctx.Pipe()
        self._process = self._ctx.Process(
            target=_subprocess_env_worker,
            args=(
                child_conn,
                {
                    "env_cfg_container": env_cfg_container,
                    "num_envs": num_envs,
                    "seed_offset": seed_offset,
                    "total_num_processes": total_num_processes,
                    "worker_info": worker_info,
                },
            ),
            daemon=False,
        )
        self._process.start()
        child_conn.close()
        msg = self._parent_conn.recv()
        if msg.get("type") != "ready":
            raise RuntimeError(
                "Failed to initialize subprocess env proxy:\n"
                f"{msg.get('traceback', msg)}"
            )

    def _rpc(self, cmd: str, payload: dict[str, Any] | None = None):
        self._parent_conn.send((cmd, payload))
        msg = self._parent_conn.recv()
        if msg.get("type") == "error":
            raise RuntimeError(msg["traceback"])
        return msg.get("result")

    @property
    def is_start(self):
        return self._rpc("get_attr", {"name": "is_start"})

    @is_start.setter
    def is_start(self, value):
        self._rpc("set_attr", {"name": "is_start", "value": value})

    def reset(self, *args, **kwargs):
        return self._rpc("call", {"name": "reset", "args": args, "kwargs": kwargs})

    def chunk_step(self, *args, **kwargs):
        return self._rpc(
            "call", {"name": "chunk_step", "args": args, "kwargs": kwargs}
        )

    def update_reset_state_ids(self):
        return self._rpc("call", {"name": "update_reset_state_ids"})

    def plan_next_bootstrap_reset(self):
        return self._rpc("call", {"name": "plan_next_bootstrap_reset"})

    def reset_to_bootstrap_plan(self, plan):
        return self._rpc(
            "call",
            {
                "name": "reset_to_bootstrap_plan",
                "args": (plan,),
                "kwargs": {},
            },
        )

    def flush_video(self):
        return self._rpc("call", {"name": "flush_video"})

    def offload(self):
        return self._rpc("call", {"name": "offload"})

    def close(self):
        if self._parent_conn is None:
            return
        try:
            self._rpc("close")
        except Exception:
            pass
        try:
            self._parent_conn.close()
        except Exception:
            pass
        if self._process.is_alive():
            self._process.join(timeout=5)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        self._parent_conn = None
