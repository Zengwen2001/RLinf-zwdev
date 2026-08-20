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

The adapters do not create an alternative model-loading path. The controller
instantiates them inside workers initialized with the production topology,
coordinates A/B/AB execution, and collects timing samples.

## Run the pairwise matrix

Use a fixture directory produced by a previous capture run:

```bash
export OVERLAP_FIXTURE_DIR=/absolute/path/to/overlap_profile/fixtures
bash examples/embodiment/run_async.sh \
  libero_spatial_async_ppo_openpi \
  '+overlap_profile@runner.overlap_profile=openpi'
```

The production Actor, Rollout, and Env groups are initialized normally. Env stays
idle while the controller runs the configured Actor/OpenPI A/B/AB pairs. Results
are written to `overlap_profile/matrix/{raw_trials,pair_samples}.jsonl`.
Each worker result includes its structural work signature and fixed-work
numerators. Pair samples report isolated and concurrent throughput such as
samples/s, observations/s, action chunks/s, and denoise iterations/s alongside
duration, slowdown, benefit, interference, start skew, and overlap coverage.

`examples/embodiment/run_async.sh` automatically generates the CSV, JSON, and
HTML report after a successful matrix run. If worker profiling is enabled, it
waits until the training process and profiled workers exit, then exports and
parses every finalized `.nsys-rep` under the run's `profiling` directory. A
report failure is appended to `run_embodiment.log` without replacing the
training command's exit status.

For a lightweight Nsight Systems trace around the matrix, add both overrides:

```bash
bash examples/embodiment/run_async.sh \
  libero_spatial_async_ppo_openpi \
  '+overlap_profile@runner.overlap_profile=openpi' \
  '+profile@cluster.profiling=matrix_trace' \
  runner.overlap_profile.profile_workers=true
```

Keep timing-only and Nsight runs separate when measuring profiler overhead. A
full eight-pair trace is large; use a reduced pairs file or fewer repetitions for
the first trace validation.

## Regenerate reports

The run script generates reports automatically. Use the commands below to
regenerate them after changing the parser or report code.

```bash
python -m toolkits.overlap_profiler report \
  --run-dir /path/to/overlap_profile/matrix
```

To export and parse all `.nsys-rep` files in the run's profiling directory in
the same command:

```bash
python -m toolkits.overlap_profiler report \
  --run-dir /path/to/overlap_profile/matrix \
  --nsys-dir /path/to/run/profiling
```

Nsight Systems supplies kernel timing, concurrency, launch gaps, synchronization,
memcpy, and NCCL kernel intervals. SM/Tensor/DRAM activity remains explicitly
`unavailable` in this timeline parser and must come from an Nsight GPU Metrics
resource run. L2 hit rate, achieved occupancy, register use, and shared-memory use
must come from isolated NCU kernel profiling.

NCCL duration is derived from NCCL kernel intervals. Collective type, payload
bytes, algorithm bandwidth, and bus bandwidth stay `unavailable` unless a run
also records NCCL trace data or explicit collective metadata. NVLink/PCIe fields
likewise require Nsight GPU Metrics, DCGM, or hardware counters.

### Align rollout with Actor Forward GPU compute

The default matrix aligns worker stage entry. OpenPI Actor Forward performs host
work before its first GPU kernel, so this mode measures end-to-end pipeline
overlap. Nsight parsing also reports the complete CUDA work launched by each
stage, including kernels that finish after the CPU NVTX range:

```text
gpu_first_activity_offset_ms
gpu_last_activity_offset_ms
gpu_compute_envelope_ms
gpu_active_union_ms
```

To measure GPU interference, first obtain the Actor Forward first-activity
offset from a stage-entry Nsight run. Use the maximum across actor ranks for a
conservative common offset:

```bash
python -m toolkits.overlap_profiler report \
  --run-dir /path/to/overlap_profile/matrix

python -c "import pandas as pd; d=pd.read_csv('/path/to/overlap_profile/matrix/stage_samples.csv'); q=d[(d.role=='actor') & (d.repetition==0) & (d.trial_kind=='A_alone') & d.pair_id.str.startswith('actor_forward__')]; print(q.groupby('rank').gpu_first_activity_offset_ms.median()); print('use_ms=', q.gpu_first_activity_offset_ms.max())"
```

Then delay rollout stage entry to that Actor GPU offset. For example, if the
calibration reports 582.3 ms:

```bash
export ACTOR_FORWARD_GPU_OFFSET_MS=582.3
bash examples/embodiment/run_async.sh \
  libero_spatial_async_ppo_openpi \
  '+overlap_profile@runner.overlap_profile=openpi_forward_gpu'
```

For an Nsight trace of the two GPU-aligned Forward pairs:

```bash
export ACTOR_FORWARD_GPU_OFFSET_MS=582.3
bash examples/embodiment/run_async.sh \
  libero_spatial_async_ppo_openpi \
  '+overlap_profile@runner.overlap_profile=openpi_forward_gpu' \
  '+profile@cluster.profiling=matrix_trace' \
  runner.overlap_profile.profile_workers=true \
  runner.overlap_profile.warmup=1 \
  runner.overlap_profile.repetitions=1
```

`start_skew_ms` remains a launch-quality metric in this mode: it measures each
worker's lateness relative to its own requested start, rather than rejecting the
intentional rollout delay. `scheduled_start_offset_ms` records the observed
Actor-to-Rollout stage-entry offset.
