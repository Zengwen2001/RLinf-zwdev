# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Single-node A/B/AB replay controller."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Protocol

from omegaconf import OmegaConf

from .metrics import compute_pair_sample


class PendingCall(Protocol):
    def wait(self) -> list[dict[str, Any]]: ...


class ReplayEndpoint(Protocol):
    role: str

    def prepare(self, fixture_dir: str, stage: str) -> list[dict[str, Any]]: ...

    def prepare_iteration(self) -> list[dict[str, Any]]: ...

    def dispatch_run(self, start_ns: int, trial_id: str) -> PendingCall: ...

    def reset(self) -> list[dict[str, Any]]: ...

    def close(self) -> list[dict[str, Any]]: ...


class WorkerGroupReplayEndpoint:
    def __init__(self, role: str, worker_group: Any):
        self.role = role
        self.worker_group = worker_group

    def prepare(self, fixture_dir: str, stage: str) -> list[dict[str, Any]]:
        return self.worker_group.prepare_overlap_replay(fixture_dir, stage).wait()

    def prepare_iteration(self) -> list[dict[str, Any]]:
        return self.worker_group.prepare_overlap_replay_iteration().wait()

    def dispatch_run(self, start_ns: int, trial_id: str) -> PendingCall:
        return self.worker_group.run_overlap_replay(start_ns, trial_id)

    def reset(self) -> list[dict[str, Any]]:
        return self.worker_group.reset_overlap_replay().wait()

    def close(self) -> list[dict[str, Any]]:
        return self.worker_group.close_overlap_replay().wait()


class PairwiseController:
    def __init__(
        self,
        endpoints: dict[str, ReplayEndpoint],
        *,
        fixture_dir: str,
        output_dir: str,
        warmup: int = 5,
        repetitions: int = 20,
        seed: int = 42,
        start_lead_ms: float = 100.0,
        max_start_skew_ms: float = 2.0,
        min_overlap_coverage: float = 0.8,
    ):
        self.endpoints = endpoints
        self.fixture_dir = str(Path(fixture_dir).resolve())
        self.output_dir = Path(output_dir)
        self.warmup = int(warmup)
        self.repetitions = int(repetitions)
        self.seed = int(seed)
        self.start_lead_ns = int(float(start_lead_ms) * 1e6)
        self.max_start_skew_ms = float(max_start_skew_ms)
        self.min_overlap_coverage = float(min_overlap_coverage)
        if self.warmup < 0 or self.repetitions < 1:
            raise ValueError("warmup must be non-negative and repetitions positive")

    def run(self, pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        raw_path = self.output_dir / "raw_trials.jsonl"
        sample_path = self.output_dir / "pair_samples.jsonl"
        all_samples: list[dict[str, Any]] = []
        try:
            with (
                raw_path.open("w", encoding="utf-8") as raw_stream,
                sample_path.open("w", encoding="utf-8") as sample_stream,
            ):
                for pair_index, pair in enumerate(pairs):
                    samples = self._run_pair(pair, pair_index, raw_stream)
                    for sample in samples:
                        sample_stream.write(json.dumps(sample, sort_keys=True) + "\n")
                    all_samples.extend(samples)
        finally:
            for endpoint in self.endpoints.values():
                endpoint.close()
        experiment = {
            "schema_version": 2,
            "fixture_dir": self.fixture_dir,
            "warmup": self.warmup,
            "repetitions": self.repetitions,
            "seed": self.seed,
            "start_lead_ms": self.start_lead_ns / 1e6,
            "max_start_skew_ms": self.max_start_skew_ms,
            "min_overlap_coverage": self.min_overlap_coverage,
            "pairs": pairs,
        }
        (self.output_dir / "experiment.json").write_text(
            json.dumps(experiment, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return all_samples

    def _run_pair(
        self, pair: dict[str, Any], pair_index: int, raw_stream
    ) -> list[dict[str, Any]]:
        pair_id = pair.get("id", f"pair_{pair_index:02d}")
        spec_a, spec_b = pair["a"], pair["b"]
        endpoint_a = self.endpoints[spec_a["role"]]
        endpoint_b = self.endpoints[spec_b["role"]]
        prepared_a = endpoint_a.prepare(self.fixture_dir, spec_a["stage"])
        prepared_b = endpoint_b.prepare(self.fixture_dir, spec_b["stage"])
        self._require_single_node(prepared_a + prepared_b)
        start_delays_ms = self._start_delays_ms(spec_a, spec_b)

        rng = random.Random(self.seed + pair_index)
        for warmup_index in range(self.warmup):
            order = ["A_alone", "B_alone", "concurrent"]
            rng.shuffle(order)
            for trial_kind in order:
                self._run_trial(
                    pair_id,
                    -warmup_index - 1,
                    trial_kind,
                    endpoint_a,
                    endpoint_b,
                    start_delays_ms,
                )

        samples = []
        for repetition in range(self.repetitions):
            order = ["A_alone", "B_alone", "concurrent"]
            rng.shuffle(order)
            trials = {}
            for trial_kind in order:
                trial = self._run_trial(
                    pair_id,
                    repetition,
                    trial_kind,
                    endpoint_a,
                    endpoint_b,
                    start_delays_ms,
                )
                trials[trial_kind] = trial
                raw_stream.write(json.dumps(trial, sort_keys=True) + "\n")
                raw_stream.flush()
            samples.append(
                compute_pair_sample(
                    pair_id=pair_id,
                    repetition=repetition,
                    stage_a=spec_a["stage"],
                    stage_b=spec_b["stage"],
                    a_alone=trials["A_alone"]["results"],
                    b_alone=trials["B_alone"]["results"],
                    concurrent=trials["concurrent"]["results"],
                    alignment_mode=pair.get("alignment_mode", "stage_entry"),
                    concurrent_start_delays_ms=start_delays_ms,
                    max_start_skew_ms=self.max_start_skew_ms,
                    min_overlap_coverage=self.min_overlap_coverage,
                )
            )
        return samples

    def _run_trial(
        self,
        pair_id: str,
        repetition: int,
        trial_kind: str,
        endpoint_a: ReplayEndpoint,
        endpoint_b: ReplayEndpoint,
        start_delays_ms: dict[str, float],
    ) -> dict[str, Any]:
        active = []
        if trial_kind in {"A_alone", "concurrent"}:
            active.append(endpoint_a)
        if trial_kind in {"B_alone", "concurrent"}:
            active.append(endpoint_b)
        prepare_results = [
            item for endpoint in active for item in endpoint.prepare_iteration()
        ]
        trial_id = f"{pair_id}:{repetition}:{trial_kind}"
        if any(not item.get("ok", False) for item in prepare_results):
            for endpoint in active:
                endpoint.reset()
            return {
                "pair_id": pair_id,
                "repetition": repetition,
                "trial_kind": trial_kind,
                "trial_id": trial_id,
                "results": prepare_results,
            }

        start_ns = time.monotonic_ns() + self.start_lead_ns
        pending = []
        requested_starts_ns = {}
        for endpoint in active:
            delay_ms = (
                start_delays_ms[endpoint.role] if trial_kind == "concurrent" else 0.0
            )
            endpoint_start_ns = start_ns + int(delay_ms * 1e6)
            requested_starts_ns[endpoint.role] = endpoint_start_ns
            pending.append(endpoint.dispatch_run(endpoint_start_ns, trial_id))
        results = [item for call in pending for item in call.wait()]
        reset_results = [item for endpoint in active for item in endpoint.reset()]
        for item in reset_results:
            if not item.get("ok", False):
                results.append(item)
        return {
            "pair_id": pair_id,
            "repetition": repetition,
            "trial_kind": trial_kind,
            "trial_id": trial_id,
            "requested_start_ns": start_ns,
            "requested_starts_ns": requested_starts_ns,
            "results": results,
        }

    @staticmethod
    def _start_delays_ms(
        spec_a: dict[str, Any], spec_b: dict[str, Any]
    ) -> dict[str, float]:
        delays = {
            spec_a["role"]: float(spec_a.get("start_delay_ms", 0.0)),
            spec_b["role"]: float(spec_b.get("start_delay_ms", 0.0)),
        }
        if any(value < 0 for value in delays.values()):
            raise ValueError("pair start_delay_ms values must be non-negative")
        minimum = min(delays.values())
        return {role: value - minimum for role, value in delays.items()}

    @staticmethod
    def _require_single_node(prepared: list[dict[str, Any]]) -> None:
        failures = [item for item in prepared if not item.get("ok", False)]
        if failures:
            raise RuntimeError(f"Failed to prepare replay workers: {failures}")
        node_ranks = {int(item["node_rank"]) for item in prepared}
        if len(node_ranks) != 1:
            raise ValueError(
                "PairwiseController currently supports one physical node; "
                f"observed node ranks {sorted(node_ranks)}"
            )


def run_configured_matrix(cfg: Any, actor_group: Any, rollout_group: Any) -> bool:
    profile_cfg = cfg.runner.get("overlap_profile", None)
    if profile_cfg is None or not bool(profile_cfg.get("enabled", False)):
        return False
    pairs_cfg = OmegaConf.load(profile_cfg.pairs_file)
    pairs = OmegaConf.to_container(pairs_cfg.get("pairs", []), resolve=True)
    controller = PairwiseController(
        {
            "actor": WorkerGroupReplayEndpoint("actor", actor_group),
            "rollout": WorkerGroupReplayEndpoint("rollout", rollout_group),
        },
        fixture_dir=profile_cfg.fixture_dir,
        output_dir=profile_cfg.output_dir,
        warmup=profile_cfg.get("warmup", 5),
        repetitions=profile_cfg.get("repetitions", 20),
        seed=profile_cfg.get("seed", 42),
        start_lead_ms=profile_cfg.get("start_lead_ms", 100.0),
        max_start_skew_ms=profile_cfg.get("max_start_skew_ms", 2.0),
        min_overlap_coverage=profile_cfg.get("min_overlap_coverage", 0.8),
    )
    profile_workers = bool(profile_cfg.get("profile_workers", False))
    if profile_workers:
        _validate_matrix_profiling(cfg)
    groups = (actor_group, rollout_group)
    profile_started = False
    try:
        if profile_workers:
            handles = [group.start_profile(0) for group in groups]
            profile_started = True
            for handle in handles:
                handle.wait()
        controller.run(pairs)
    finally:
        if profile_started:
            handles = [group.stop_profile() for group in groups]
            for handle in handles:
                handle.wait()
    return True


def _validate_matrix_profiling(cfg: Any) -> None:
    profiling = cfg.cluster.get("profiling", None)
    if profiling is None or not bool(profiling.get("enabled", False)):
        raise ValueError(
            "runner.overlap_profile.profile_workers=true requires enabled "
            "cluster.profiling"
        )
    steps = profiling.get("steps", None)
    if steps is None or 0 not in [int(step) for step in steps]:
        raise ValueError(
            "Matrix profiling requires cluster.profiling.steps to contain 0 so "
            "Nsight uses a cudaProfilerApi capture window"
        )
