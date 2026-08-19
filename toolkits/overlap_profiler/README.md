# RLinf Overlap Profiler

This toolkit replays fixed production workloads for isolated and pairwise overlap
experiments. The current implementation covers Async PPO FSDP Actor micro-batches
and the legacy OpenPI rollout path.

## Capture fixtures

Run fixture capture separately from Nsight profiling:

```bash
bash examples/embodiment/run_async.sh \
  libero_spatial_async_ppo_openpi \
  '+profile_capture@runner.profile_capture=openpi' \
  runner.max_steps=3
```

The preset captures one Actor micro-batch and one Rollout observation per rank at
model version 2. Output is written under:

```text
<log-dir>/libero_spatial_async_ppo_openpi/overlap_profile/fixtures/
```

Override the target version or per-rank limit with:

```text
runner.profile_capture.version=1
runner.profile_capture.max_per_kind_per_rank=2
```

Fixture capture and `cluster.profiling` are intentionally mutually exclusive.
Fixtures use atomic writes and per-file manifests. The first schema version does
not compute checksums.

## Replay adapters

`ActorFSDPReplayAdapter` wraps an initialized production Actor worker and supports:

```text
actor.forward
actor.backward_no_sync
actor.backward_sync
actor.optimizer
actor.full_microbatch
```

`OpenPIReplayAdapter` wraps an initialized legacy OpenPI policy and supports:

```text
rollout.obs_preprocess
rollout.vlm_prefix
rollout.ae_denoise
rollout.full_policy
```

The adapters do not create an alternative model-loading path. The phase 6
controller will instantiate them inside workers initialized with the production
topology, coordinate A/B/AB execution, and collect timing samples.
