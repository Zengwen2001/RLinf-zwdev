# Hardware Utilization Monitor

Record CPU/GPU utilization while a command runs, then generate:

- `samples.csv`: raw time-series data.
- `report.html`: interactive Plotly charts with zoom, pan, legend filtering, and range sliders.
- `metadata.json`: command and run metadata.

The report contains two charts only:

- CPU utilization.
- GPU utilization.

When `metrics.log` is available and more than one training step has completed,
the monitor excludes the first step as warmup and draws dashed horizontal average
utilization lines for samples collected after step 1. With one or zero completed
steps, the average-line feature is disabled automatically.

## Wrap a training command

Embodiment run scripts have a built-in `profile-util` switch. The monitor output
is written to the current run log directory under `hardware_util/`.

```bash
bash examples/embodiment/run_embodiment.sh libero_spatial_ppo_openpi_pi05 profile-util
bash examples/embodiment/run_async.sh libero_spatial_async_ppo_openpi profile-util
```

Set the sampling interval with `UTIL_PROFILE_INTERVAL`:

```bash
UTIL_PROFILE_INTERVAL=0.5 bash examples/embodiment/run_async.sh libero_spatial_async_ppo_openpi profile-util
```

You can also call the monitor directly:

```bash
python3 toolkits/hardware_monitor/record_utilization.py \
  -o logs/hw_async_ppo \
  -i 1.0 \
  --title "async PPO hardware util" \
  -- \
  python3 examples/embodiment/train_async.py --config-name libero_spatial_async_ppo_openpi
```

## Monitor an existing run

```bash
python3 toolkits/hardware_monitor/record_utilization.py \
  -o logs/hw_manual \
  -i 1.0
```

Stop with `Ctrl-C`; the CSV and HTML report are still written.

## Common options

```bash
--gpu-ids 0,1      # only record selected GPUs
--no-gpu           # CPU only
--duration 600     # stop after 10 minutes
```
