# VLM-AE Pipeline Design Document

## 1. 文档目的

这份文档记录 `OpenPI` 在 `RLinf` 中当前的两条增量流水线实现：

1. `baseline micro-pipeline`
   只在 baseline 统一推理路径上增加 `2 micro-batch` 的异步重叠，不做 VLM/AE 拆分。
2. `disagg pipeline`
   将 rollout worker 拆成 `3 VLM + 1 AE`，VLM 负责 prefix/KV，AE 负责 denoise/value/forward_inputs，并通过 KV 传输连接。

文档重点说明：

- 当前代码已经实现了什么
- 各个代码文件承担什么职责
- 每个关键参数、函数、patch 的作用
- 运行时数据是怎么流动的
- 当前限制和已知行为

---

## 2. 当前实现状态

### 2.1 已实现的三条运行路径

1. baseline 原始路径
   不开任何环境变量，沿用 RLinf/OpenPI baseline 逻辑。

2. baseline micro-pipeline
   开启 `VLM_AE_BASELINE_MICROPIPE=1` 后：
   - 不拆分 VLM / AE worker
   - 仍然是每个 rollout worker 运行完整 `predict`
   - 但 env 和 rollout 之间按 `micro-batch` 通信
   - `mb0` 结果出来后可先发回 env 执行，env 再尽快返回下一轮 `mb0` obs

3. disagg pipeline
   开启 `VLM_AE_DISAGG=1` 后：
   - rollout world size 固定为 `4`
   - `rank 0,1,2` 为 VLM worker
   - `rank 3` 为 AE worker
   - env 只向 VLM 发送 micro-batch 级数据
   - VLM 向 AE 发送 KV cache / prefix_output / vlm_value / obs metadata bundle
   - AE 合并 KV 后完成 denoise，并把 micro 级 rollout result 发回 env

### 2.2 当前结论

当前实现已经不再是最早文档里描述的“VLM 和 AE 同时收完整 env_output”的状态，已经演进为：

- env 和 rollout 之间支持 `micro-batch` 级通信
- disagg 路径下当前热路径是 `env -> VLM -> AE -> env`
- env 可以按 `chunk_step_subset()` 执行局部 micro-batch
- timeline 已经可以记录 `env_micro`、`rollout_vlm`、`rollout_ae`、`kv_wait`、`kv_queue`、`kv_unpack`
- AE 侧已经支持 Gemma cache 兼容、AE phase 细分统计、VLM value 前移复用
- VLM 侧已经改为“首个 send 同步建链，后续 send 异步挂起并延迟回收”，以减少 `mb0 -> send -> mb1` 和 `mb1 -> send -> next chunk` 的硬等待

需要特别说明的是：历史版本里确实存在一条 `env -> AE` metadata 通路，当时它主要用于给 AE 侧补 `forward_inputs` 和 `bootstrap` 所需的 obs/final_obs。当前实现已经把这部分 metadata 合并进 `VLM -> AE` bundle，因此不应再把 AE 理解成“直接依赖 env 输入”的计算节点。

---

## 3. 两条流水线的目标时序

### 3.1 baseline micro-pipeline

目标是只在 baseline 基础上增加 micro-batch overlap：

```text
rollout(N, mb0) -> env(N, mb0) -> rollout(N+1, mb0)
rollout(N, mb1) ----------------^
```

特点：

- 不拆模型，不传 KV
- 只把 env 和 rollout 的交互拆成 `mb0 / mb1`
- 主要用于评估“仅靠 micro-pipeline 能拿到多少收益”

### 3.2 disagg pipeline

目标是实现：

```text
VLM N mb0 -> VLM N mb1 -> VLM N+1 mb0
AE  N mb0 -> AE  N mb1 -> AE  N+1 mb0
Env N mb0 -> Env N mb1 -> Env N+1 mb0
```

更具体地说：

- VLM `mb0` 计算完后，不应立即被同步 send 阻塞
- VLM `mb1` 计算完后，尽量不要等传输全部结束才进入下一个 chunk
- AE 在 `mb0` 计算时，`mb1` 的 KV 应尽早完成 recv，不要再假性卡在“等待传输”
- env 在 `mb0` action ready 后，尽早执行 `mb0`，并尽早把下一轮 `mb0` obs 回送

---

## 4. 运行开关与参数

### 4.1 disagg 模式环境变量

```bash
export VLM_AE_DISAGG=1
export VLM_AE_VLM_GPUS=0,1,2
export VLM_AE_AE_GPU=3
export VLM_AE_NUM_MICRO_BATCHES=2
export VLM_AE_MAX_PENDING_SENDS=3
```

参数说明：

- `VLM_AE_DISAGG`
  开启 VLM/AE 拆分路径。

- `VLM_AE_VLM_GPUS`
  指定哪些 rollout rank 视为 VLM worker，默认 `0,1,2`。

- `VLM_AE_AE_GPU`
  指定 AE worker 对应 rank 的逻辑 GPU 编号，默认 `3`。

- `VLM_AE_NUM_MICRO_BATCHES`
  每个 stage 切成多少个 micro-batch，目前主要按 `2` 使用。

- `VLM_AE_MAX_PENDING_SENDS`
  VLM 侧允许挂起多少个异步 KV send 再开始回收。值越大，VLM 越不容易被 send 同步卡住，但会增加挂起中的 KV 显存占用。

### 4.2 baseline micro-pipeline 模式环境变量

```bash
unset VLM_AE_DISAGG
export VLM_AE_BASELINE_MICROPIPE=1
export VLM_AE_NUM_MICRO_BATCHES=2
```

参数说明：

- `VLM_AE_BASELINE_MICROPIPE`
  启用“仅 baseline + micro-pipeline”路径，不启用 disagg。

- `VLM_AE_NUM_MICRO_BATCHES`
  当前要求本地 batch size 可被它整除。

### 4.3 compile 相关环境变量

```bash
export RLINF_TORCH_COMPILE_CACHE_ROOT=/path/to/cache_root
```

作用：

- rollout worker 初始化时会基于 rank 派生出独立的 `TORCHINDUCTOR_CACHE_DIR` 和 `TRITON_CACHE_DIR`
- 用来避免多 rank 同时 `max-autotune` 时互相踩 Triton/Inductor benchmark cache

---

## 5. 当前代码结构总览

### 5.1 `rlinf/models/embodiment/openpi/vlm_ae_disagg.py`

职责：定义 disagg 模式的角色、通信切片规则、rank 映射规则。

关键内容：

- 全局配置读取
  - `VLM_AE_DISAGG_ENABLED`
  - `VLM_AE_VLM_GPUS`
  - `VLM_AE_AE_GPU`
  - `VLM_AE_NUM_MICRO_BATCHES`
  - `VLM_AE_TRANSFER_BACKEND`
  - `VLM_AE_DEBUG`

- 角色定义
  - `WorkerRole`
    - `VLM`
    - `AE`
    - `UNIFIED`

- 切片结构
  - `MicroSliceSpec`
    描述一个 micro-batch 在两个 rank 之间如何切片和重组：
    - `peer_rank`
    - `micro_batch_id`
    - `src_start / src_end`
    - `dst_start / dst_end`
    - `order`

- 关键映射函数
  - `get_worker_role()`
    根据 `RANK/LOCAL_RANK` 推导当前 rollout worker 是 VLM 还是 AE。
  - `get_num_vlm_workers()`
    返回 VLM worker 数量。
  - `get_ae_worker_rank()`
    返回 AE rollout rank。
  - `get_vlm_worker_index()`
    把 rollout rank 映射为 VLM worker 序号。
  - `get_env_to_vlm_dst_ranks()`
    env -> VLM 的 coarse rank 映射。
  - `get_vlm_env_src_ranks()`
    VLM <- env 的 coarse rank 映射。
  - `get_ae_to_env_dst_ranks()`
    AE -> env 的 coarse rank 映射。
  - `get_env_from_ae_src_ranks()`
    env <- AE 的 coarse rank 映射。

- micro 级切片函数
  - `get_vlm_env_micro_src_slices()`
    VLM 预期从哪些 env rank 收到哪些 micro shard。
  - `get_env_to_vlm_micro_dst_slices()`
    env 实际往哪些 VLM rank 发哪些 micro shard。
  - `get_ae_to_env_micro_dst_slices()`
    AE -> env 的 rollout result micro fan-out 布局。
  - `get_env_from_ae_micro_src_slices()`
    env <- AE 的 micro result 接收布局。

这些函数共同决定了：

- 一个 micro-batch 在 env、VLM、AE 三侧如何切分
- 多个 env shard 如何在 VLM 侧重组成局部 batch
- AE 结果如何再扇出回各个 env rank

### 5.2 `rlinf/workers/rollout/hf/vlm_ae_pipeline_worker.py`

职责：patch `MultiStepRolloutWorker`，实现 baseline micro-pipeline 和 disagg pipeline 两条路径。

#### 5.2.1 顶层数据结构和工具函数

- `PipelineConfig`
  描述 disagg pipeline 的局部 batch 布局：
  - `enabled`
  - `num_micro_batches`
  - `batch_size`
  - `micro_batch_size`
  - `vlm_gpus`
  - `ae_gpu`

- `KVPipelineMessage`
  VLM -> AE 的张量化 bundle。为了走 `DATACLASS_WITH_TENSORS` 快路径，除了 KV 外，还把 AE 需要的 env metadata 合并进同一个 payload：
  - `kv_keys`
  - `kv_values`
  - `prefix_output`
  - `prefix_pad_masks`
  - `state`
  - `vlm_value`
  - `obs_*`
  - `final_*`

  这里的 `obs_* / final_*` 不是额外的 env->AE 通道，而是 VLM 在收到 env micro obs 后，顺手把 AE 需要的 metadata 一起打进 `VLM -> AE` bundle。当前实现里 AE 不再直接从 env 收单独的 micro obs。

- `_pack_kv_message() / _unpack_kv_message()`
  在 send/recv 两侧把 KV、当前 obs metadata、final obs metadata 打包、解包。

- `_append_worker_timeline_event()`
  给 timeline 写事件。

- `_append_ae_model_phase_events()`
  把模型内部记录的 AE phase timeline 写到 `rollout_ae_detail`。

#### 5.2.2 `patched_init_worker(self)`

这是整套 patch 的入口。

它的作用：

1. compile cache 隔离
   为每个 rollout rank 设置独立的 `TORCHINDUCTOR_CACHE_DIR` / `TRITON_CACHE_DIR`。

2. disagg 初始化
   - 计算当前 rank 的角色
   - 校验 rollout world size 和 batch divisibility
   - 重写 `src_ranks["train"]` / `dst_ranks["train"]`
   - 建立 env micro slice spec
   - 调用 `hf_model.setup_vlm_ae_disagg(batch_size=...)`

3. VLM 发送窗口初始化
   - `self._vlm_ae_pending_sends`
   - `self._vlm_ae_send_warmup_done`
   - `self._vlm_ae_max_pending_sends`

4. baseline micro-pipeline 初始化
   在 `VLM_AE_BASELINE_MICROPIPE=1` 时，只校验 batch 和 1:1 映射，不做 VLM/AE 拆分。

#### 5.2.3 baseline micro-pipeline 相关函数

- `_recv_baseline_micro_env_output()`
  rollout 从单个 env peer 接收 `mode_obs_mb{micro_id}`。

- `_send_baseline_micro_rollout_result()`
  rollout 立即把当前 micro 的 `RolloutResult` 发回 env。

- `_unified_micro_generate_one_epoch()`
  baseline micro-pipeline 的主循环：
  - 收到 `mb0` obs
  - 运行完整 `predict`
  - 立刻把 `mb0` result 发回 env
  - env 可以先 step `mb0`
  - rollout 再处理 `mb1`

#### 5.2.4 disagg VLM 相关函数

- `_recv_env_micro_output()`
  仅 VLM 从 env 侧接收一个 micro-batch 的 obs shard，并按切片规则 merge。

- `_dispatch_vlm_send()`
  当前最新的 VLM send 调度函数：
  - 首个 send 用 `async_op=False` 同步建链
  - 后续 send 用 `async_op=True`
  - 超过 `self._vlm_ae_max_pending_sends` 时，回收最老的 pending send

- `_wait_one_pending_vlm_send()`
  等待一个挂起的 send 完成，并记录 `kv_transfer` timeline。

- `_flush_pending_vlm_sends()`
  在尾部或窗口超限时统一回收 pending sends。

- `_vlm_generate_one_epoch()`
  VLM worker 主循环：
  - 按 micro 接收 env obs
  - 调 `hf_model._predict_vlm_stage()`
  - 把结果打包成 `KVPipelineMessage`
  - 调 `_dispatch_vlm_send()`

当前这个函数已经不再是“每个 micro 强同步 send”，而是“计算优先 + 小窗口异步 send”。

#### 5.2.5 disagg AE 相关函数

- `_issue_async_kv_recvs()`
  为每个 `micro_id x vlm_rank` 提前挂起 `recv(async_op=True)`，并用后台线程等待完成。

- `_merge_kv_from_vlm_workers()`
  把 3 张 VLM 卡传来的 shard 按 batch dim 合并：
  - `kv_cache`
  - `prefix_output`
  - `prefix_pad_masks`
  - `state`
  - `vlm_value`
  - `env_obs`
  - `final_obs`

- `_send_rollout_result_micro()`
  把 AE 生成的 micro rollout result 按 `MicroSliceSpec` fan-out 回 env。

- `_ae_generate_one_epoch()`
  AE worker 主循环：
  - 先把所有 recv 挂出去
  - 后台线程等 recv 完成，把完成消息放入队列
  - 主线程消费完成队列，记录：
    - `kv_wait`
  - `kv_queue`
  - `kv_unpack`
  - 某个 micro 收齐 3 份 VLM shard 后，先在本地合并出 `kv + env_obs + final_obs`
  - 再调用 `hf_model._predict_ae_stage()`
  - 构造 `RolloutResult`
  - 通过 `_send_rollout_result_micro()` 发回 env

#### 5.2.6 patch 注册

在文件末尾把上述函数挂到 `MultiStepRolloutWorker` 上：

- `init_worker`
- `generate_one_epoch`
- `_unified_micro_generate_one_epoch`
- `_vlm_generate_one_epoch`
- `_ae_generate_one_epoch`
- `_dispatch_vlm_send`
- `_flush_pending_vlm_sends`
- `_issue_async_kv_recvs`
- `_merge_kv_from_vlm_workers`

### 5.3 `rlinf/workers/env/env_worker.py`

职责：把 env 侧通信和 env step 扩展为支持 micro-batch 的 baseline/disagg 路径。

#### 5.3.1 初始化阶段

关键状态：

- `self._vlm_ae_disagg_enabled`
- `self._baseline_micro_pipeline_enabled`
- `self._vlm_ae_num_micro_batches`
- `self._vlm_ae_env_to_vlm_micro_specs`
- `self._vlm_ae_from_ae_micro_specs`

在 `init_worker()` 中：

- baseline 路径使用 RLinf 原始 `src_ranks/dst_ranks`
- disagg 路径会把 `dst_ranks["train"]` 改成发往 VLM 的 rank
- 同时建立所有 micro slice spec

#### 5.3.2 通用切片与回填函数

这些函数是 env micro-pipeline 的基础：

- `_slice_nested_batch()`
  切片任意嵌套张量/列表结构。

- `_slice_env_batch()`
  从局部 env batch 切出一个 micro。

- `_slice_env_output()`
  从 `EnvOutput` 里切出一个 micro。

- `_infer_local_env_batch_size()`
  推断本地 env batch 大小。

- `_initialize_missing_tensor_field()`
  给原来为 `None` 的字段创建完整 batch shape 的 tensor。

- `_replace_nested_slice()`
  把 micro 结果写回整批结构。

- `_update_env_output_slice()`
  将某个 micro step 后的奖励、done、截断、final_obs 等写回整批 `EnvOutput`。
  这是后来修正过的关键函数，用来避免 `mb0` 把整批缓存“缩成一小块”。

- `_slice_rollout_result()`
  从整批 `RolloutResult` 中取一个局部切片。

- `_merge_chunk_step_result_shards()`
  把多个 micro 的 `ChunkStepResult` 合并为一个 chunk-step 结果。

#### 5.3.3 baseline micro-pipeline 相关函数

- `_get_local_micro_bounds()`
  返回本地 batch 上某个 micro 的 `[start, end)`。

- `_send_baseline_micro_batch()`
  env 向唯一的 rollout peer 发送 `mode_obs_mb{micro_id}`。
  支持两种模式：
  - 从完整 batch 里现切
  - `already_sliced=True` 时，直接发送已经是 micro 大小的局部 obs

- `_recv_baseline_rollout_result_micro()`
  env 从 rollout 收当前 micro 的 `RolloutResult`。

- `_run_interact_once_baseline_micro()`
  baseline micro-pipeline 主循环：
  - bootstrap 时先发所有 micro obs
  - 每个 chunk step 逐个收 `mb0/mb1` result
  - 按 micro 执行 env subset step
  - 每个 micro step 后立即把下一轮该 micro obs 回送 rollout

#### 5.3.4 disagg 相关函数

- `_send_env_micro_batches()`
  env 只向 VLM 发局部 shard。
  AE 需要的 obs/final_obs metadata 由 VLM 在 `_pack_kv_message()` 时并入 `VLM -> AE` bundle。

- `_recv_rollout_result_micro()`
  env 从 AE 收 micro rollout result。

- `_run_interact_once_disagg()`
  disagg 主循环：
  - bootstrap 发所有 micro obs
  - chunk 内逐个收 AE 的 micro result
  - 调 `env_interact_micro_step()`
  - 更新整批 `EnvOutput`
  - 把下一轮该 micro 的新 obs 立即发给 VLM

#### 5.3.5 路径分发

`_run_interact_once()` 现在会根据环境变量分三路：

1. disagg
2. baseline micro-pipeline
3. 原始 baseline

### 5.4 `rlinf/models/embodiment/openpi/openpi_action_model.py`

职责：模型层对 disagg 做适配，拆出 VLM 阶段和 AE 阶段，并补齐 rollout 需要的数值。

关键函数：

- `setup_vlm_ae_disagg(batch_size)`
  初始化 disagg 所需的模型内部状态。

- `_predict_vlm_stage(env_obs)`
  VLM 阶段：
  - 做 prefix 前向
  - 产出 `kv_cache`
  - 产出 `prefix_output`
  - 产出 `prefix_pad_masks`
  - 产出 `state`
  - 预先计算 `vlm_value`

- `_predict_ae_stage(kv_data, env_obs, mode, compute_values)`
  AE 阶段：
  - 把 VLM 发来的 legacy KV 转成 Gemma 可用 cache
  - 运行 denoise loop
  - 生成 actions
  - 生成 `prev_logprobs` / `prev_values`
  - 用 `KVPipelineMessage` 里合并来的 obs metadata 组装 rollout 训练所需的 `forward_inputs`
  - 记录 `_last_ae_stage_timing`

- `_normalize_past_key_values_for_gemma()`
  将跨 rank 传输后的 KV 规范成 Gemma/Transformers 当前版本可接受的 cache 对象。

- `_build_rollout_forward_inputs()`
  按 baseline 语义重建 rollout 训练侧需要保存的 forward inputs。

- `get_value_from_vlm(prefix_output)`
  原本在 AE 侧重复计算，后来已经前移到 VLM 侧，AE 优先复用 VLM 传来的 `vlm_value`。

- `predict_action_batch()`
  原始统一入口。现在在 baseline 模式仍走这里；在 disagg 模式下会走 `_predict_vlm_stage + _predict_ae_stage` 的拆分逻辑。

### 5.5 `rlinf/envs/libero/libero_env.py`

职责：给 Libero 增加 subset step 能力，使 env 可按 micro-batch 推进。

关键函数：

- `step_subset(actions, env_idx, auto_reset=True)`
  只对给定 `env_idx` 子集执行一步。

- `chunk_step_subset(chunk_actions, env_idx)`
  对给定子集执行一整个 action chunk。

这是 env micro-pipeline 能成立的基础，否则 env 只能整批 step。

### 5.6 `rlinf/envs/wrappers/record_video.py`

职责：在 wrapper 层透传 subset step。

关键函数：

- `chunk_step_subset(*args, **kwargs)`
  直接转发到底层 env 的 `chunk_step_subset()`。

已知限制：

- 训练视频在 micro-step 模式下可能报
  `All images in a movie should have same size`
- 原因是不同 micro 的 env 数量不同，tiled frame 尺寸不一致
- 这只影响视频保存，不影响训练/rollout 正确性

### 5.7 `rlinf/utils/plot_timeline.py`

职责：把新加的 timeline 事件可视化。

已支持的关键 component：

- `env_micro`
- `rollout_vlm`
- `rollout_ae`
- `rollout_ae_detail`
- `kv_transfer`
- `kv_wait`
- `kv_queue`
- `kv_unpack`

这些事件用于分析：

- env 是否提前推进了 `mb0`
- VLM 是否被 send 卡住
- AE 的真实瓶颈是在收包、排队还是 denoise
- 模型内部 phase 的具体热点在哪里

---

## 6. 两条新路径的运行时数据流

### 6.1 baseline micro-pipeline

```text
EnvWorker
  -> send mode_obs_mb0
  -> send mode_obs_mb1

RolloutWorker(unified)
  -> recv mb0 obs
  -> predict(mb0)
  -> send rollout_results_mb0
  -> recv mb1 obs
  -> predict(mb1)
  -> send rollout_results_mb1

EnvWorker
  -> recv mb0 result
  -> chunk_step_subset(mb0)
  -> 立即 send 下一轮 mb0 obs
  -> recv mb1 result
  -> chunk_step_subset(mb1)
  -> 立即 send 下一轮 mb1 obs
```

### 6.2 disagg pipeline

```text
EnvWorker
  -> send VLM micro shard to VLM ranks

VLMWorker(rank 0/1/2)
  -> recv local micro obs
  -> _predict_vlm_stage()
  -> bundle current obs/final_obs metadata into KVPipelineMessage
  -> _dispatch_vlm_send() to AE

AEWorker(rank 3)
  -> _issue_async_kv_recvs()
  -> consume completed recvs
  -> merge 3-way KV shards + obs/final_obs metadata
  -> _predict_ae_stage()
  -> send micro rollout result back to env

EnvWorker
  -> recv AE micro result
  -> chunk_step_subset(micro)
  -> update full EnvOutput cache
  -> 立即 send next micro obs to VLM
```

---

## 7. 当前已做过的关键增量修改

### 7.1 通信与切片

- disagg 不再复用 baseline 的 rollout-world-size 映射
- 增加 env/VLM/AE 三侧的 micro slice spec
- 当前热路径为 `env -> VLM -> AE -> env`
- env -> AE 单独通道已经移除；AE 所需 metadata 合并进 `VLM -> AE` payload

### 7.2 KV 传输

- VLM -> AE 负载改为 `KVPipelineMessage`
- KV 展平成 dataclass tensor fields，避免落入 CPU object/pickle 慢路径
- AE 需要的 `obs/final_obs` metadata 也通过同一个 `KVPipelineMessage` 传递
- AE 提前挂起 recv，并用后台线程等待完成
- timeline 拆成：
  - `kv_transfer`
  - `kv_wait`
  - `kv_queue`
  - `kv_unpack`

### 7.3 模型兼容性

- AE 侧把 legacy KV 转成 Transformers/Gemma 可接受 cache
- AE compile 路径适配 `DynamicCache`
- VLM value 前移到 VLM worker 计算并复用
- AE 内部 phase timeline 已接入

### 7.4 compile 与稳定性

- 为每个 rollout rank 隔离 Triton/Inductor cache
- 避免多 rank `max-autotune` benchmark 文件互相覆盖
- disagg 路径不支持 RLinf 的 `enable_cuda_graph`

### 7.5 env 侧 micro-step

- Libero 支持 `step_subset` / `chunk_step_subset`
- env worker 支持微批回填整批 `EnvOutput`
- baseline micro-pipeline 和 disagg 都能使用同一套 env subset 执行基础设施

---

## 8. 当前限制和注意事项

1. disagg 目前假设 rollout world size 为 `3 VLM + 1 AE`
2. baseline micro-pipeline 目前要求 `env <-> rollout` 是 `1:1` 映射
3. `VLM_AE_MAX_PENDING_SENDS` 过大可能提高显存占用
4. 训练视频在 micro-step 模式下可能因 frame size 不一致而保存失败
5. timeline 中：
   - `kv_transfer/r0-r2` 更接近 sender 侧 send 完成时间
   - `kv_wait/r3` 是 AE posted recv 到 ready 的时间
   - `kv_queue/r3` 是消息已经 ready，但 AE 主线程还没来得及消费的时间
6. 当前 VLM send 已经明显去掉了“每个 micro 强制同步”的等待，但仍然可能在：
   - 首次建链
   - pending send 窗口打满
   - AE/env 背压较强
   时出现回收等待

---

## 9. 推荐测试方式

### 9.1 baseline micro-pipeline

```bash
ray stop --force
unset VLM_AE_DISAGG
export VLM_AE_BASELINE_MICROPIPE=1
export VLM_AE_NUM_MICRO_BATCHES=2
bash examples/embodiment/run_embodiment.sh libero_goal_ppo_openpi_pi05
```

### 9.2 disagg

```bash
ray stop --force
unset VLM_AE_BASELINE_MICROPIPE
export VLM_AE_DISAGG=1
export VLM_AE_NUM_MICRO_BATCHES=2
export VLM_AE_VLM_GPUS=0,1,2
export VLM_AE_AE_GPU=3
export VLM_AE_MAX_PENDING_SENDS=3
bash examples/embodiment/run_embodiment.sh libero_goal_ppo_openpi_pi05
```

### 9.3 timeline 绘图

```bash
python rlinf/utils/plot_timeline.py logs/<exp_dir>/timeline --format html
```

重点关注：

- `env_micro`
- `rollout_vlm`
- `kv_transfer`
- `kv_wait`
- `kv_queue`
- `rollout_ae`
- `rollout_ae_detail`

---

## 10. 当前推荐阅读顺序

如果要从代码角度理解整个实现，建议按这个顺序看：

1. `vlm_ae_disagg.py`
   先理解角色、rank 映射、micro slice spec
2. `vlm_ae_pipeline_worker.py`
   再看 rollout 侧 patch 如何驱动两条路径
3. `env_worker.py`
   看 env 如何按 micro 发送和执行
4. `openpi_action_model.py`
   看模型层如何拆成 VLM 阶段和 AE 阶段
5. `libero_env.py`
   看 subset step 的 env 基础能力
6. `plot_timeline.py`
   最后结合 timeline 理解实际重叠效果

---

## 11. 总结

当前代码已经从“概念设计”进入“可运行的增量实现”阶段：

- baseline micro-pipeline 可用于评估纯 micro overlap
- disagg pipeline 已经具备完整的数据通路
- timeline 已能帮助定位 VLM send、AE queue、AE 内部 phase 的热点
- env 侧 subset step 和 worker patch 已经形成稳定的代码结构

后续如果继续优化，优先级通常是：

1. 进一步减少 VLM send 回收造成的尾部等待
2. 继续提升 env `mb0 -> next mb0` 的启动及时性
3. 解决视频录制与 subset step 的兼容问题
4. 视需要扩展到更多 micro-batch 或更多 GPU 拓扑
