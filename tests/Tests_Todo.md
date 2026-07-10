# Tests Todo：PCL + PointWorld（本地回放验收 → Franka 联合测试）

## 目标
把调试与验收收口到同一条主链路（PCL schema_version=100，端口 9011）：

- 本地单机：server 启动稳定、client 单次请求稳定
- 本地回放：原生 `episode_dir`（back+side 双视角）回放稳定，验收脚本输出 `FINAL: PASS`
- Franka 侧数据测试：只采集并上行，不闭环执行动作
- Franka↔云端联合测试：从 dummy_hold → mppi_joint → PointWorld 三档递进

## 标准入口（唯一推荐）
- 验收脚本：`/home/wangyuhan/MPPI/tests/run_pw_replay_acceptance.sh`
- 回放 client（episode_dir → ObsPCL.cameras）：`/home/wangyuhan/MPPI/tests/pw_replay_acceptance.py`
- 单次请求 client（双视角）：`python3 -m mppi.comm.ws_client_sync_pcl --rgb-back ... --depth-back ... --rgb-side ... --depth-side ...`
- server：`python3 -m mppi.comm.ws_server_async_pcl`

## Profile（同一入口三档切换）
- `no_pw`：不使用 task term（用于对照）
- `obs_only`：只启用 `I_obs`
- `obs_infl`：启用 `I_obs + I_infl`

## 最小前置条件
- `POINTWORLD_ROOT` 可用（server 侧 import PointWorld 相关模块）
- `MPPI_PW_COTRACKER_CKPT` 可用
- `MPPI_PW_MODEL_PATH` 可用（如果启用 cost model）
- `configs/pointworld_static_aabbs.json` 已固定
- `ws_server_async_pcl.py` 已接受验收落盘改动（`MPPI_PW_ACCEPTANCE_DUMP_DIR` 生效）

---

## Stage 0：本地环境 sanity

### 0.1 依赖可 import
调试：Python 环境与关键依赖

```bash
cd /home/wangyuhan/MPPI
PYTHONPATH=/home/wangyuhan/MPPI/src python3 -c "import mppi; print('mppi OK')"
python3 -c "import websockets; print('websockets OK')"
python3 -c "import yaml; print('pyyaml OK')"
python3 -c "import cv2; print('opencv OK')"
python3 -c "import torch; print('torch OK')"
```

预期：全部打印 `... OK`。

---

## Stage 1：本地单机（通信 + 解码）

### 1.1 起 PCL server（dummy_hold，双视角标定就绪）
调试：websocket listen + PCL 解码链路不崩溃 + back/side 标定路径能被读取

```bash
cd /home/wangyuhan/MPPI
MPPI_PCL_CAM_INFO_BACK_PATH=/home/wangyuhan/MPPI/configs/back_cam_info.yaml \
MPPI_PCL_VERBOSE=1 MPPI_PCL_PRINT_EVERY=10 MPPI_PCL_HEARTBEAT_S=10.0 \
MPPI_PCL_T_BASE_CAM_BACK_PATH=/home/wangyuhan/MPPI/configs/T_base_cam_back.yaml \
MPPI_PCL_CAM_INFO_SIDE_PATH=/home/wangyuhan/MPPI/configs/side_cam_info.yaml \
MPPI_PCL_T_BASE_CAM_SIDE_PATH=/home/wangyuhan/MPPI/configs/T_base_cam_side.yaml \
PYTHONPATH=/home/wangyuhan/MPPI/src \
python3 -m mppi.comm.ws_server_async_pcl \
  --host 0.0.0.0 \
  --port 9011 \
  --open-loop-horizon 8 \
  --policy dummy_hold \
  --cam-id back
```

预期：server 进程常驻不退出。

### 1.2 单次请求（双视角 client：back+side）
调试：PCL client 双视角编码（jpeg + npy_zlib）→ ObsPCL.cameras → server 回包

```bash
cd /home/wangyuhan/MPPI
PYTHONPATH=/home/wangyuhan/MPPI/src \
python3 -m mppi.comm.ws_client_sync_pcl \
  --url ws://127.0.0.1:9011 \
  --rgb-back /home/datasets/FrankaNav/ep_00152/back/0000.jpg \
  --depth-back /home/datasets/FrankaNav/ep_00152/back_depth/0000.npy \
  --rgb-side /home/datasets/FrankaNav/ep_00152/side/0000.jpg \
  --depth-side /home/datasets/FrankaNav/ep_00152/side_depth/0000.npy \
  --depth-unit-scale 1.0 \
  --step-id 0 \
  --request-timeout-s 10 \
  --print-actions
```

预期：打印 `infer_ms=... policy=dummy_hold`，并能打印 `actions[0]`。

通过标准：连续多次请求无超时/无 schema 错误。

---

## Stage 2：本地回放验收（收口 PASS/FAIL）

### 2.1 两终端：server + replay + acceptance（原生 episode，双视角）
调试：PointWorld window/tracking + 关键字段稳定产出

episode 目录结构（与 Terminal#133-134 对齐）：
- `back/` + `back_depth/`
- `side/` + `side_depth/`
- `data.pkl`

Terminal A（起 server，保持常驻）：
```bash
cd /home/wangyuhan/MPPI
bash /home/wangyuhan/MPPI/tests/run_pw_replay_acceptance.sh server obs_infl
```

Terminal B（跑 replay + acceptance，收口 PASS/FAIL）：
```bash
EPISODE_DIR=/home/datasets/FrankaNav/ep_00152 \
DUAL_VIEW=1 \
START_IDX=0 \
MAX_STEPS=16 \
REQUEST_TIMEOUT_S=120 \
bash /home/wangyuhan/MPPI/tests/run_pw_replay_acceptance.sh replay obs_infl
```

预期：最后输出 `FINAL: PASS`。

---

## Stage 3：Franka 侧 client 数据测试（不闭环执行动作）

### 目标与通过标准
调试目标：Franka 侧仅采集并上行 PCL obs，云端 server 能回包。

通过标准（达到即可开始“数据测试”部署）：
- 本地 Stage 2 已 `FINAL: PASS`
- Franka 侧能持续发送（建议先 1~5Hz）并稳定回包
- `step_id` 单调递增、`t_client_send_ns` 单调递增
- `depth_unit_scale` 与真实深度格式一致（float32 米 或 uint16 + scale）

### 最小落地方式（Franka 侧双视角：先单次请求再回放）
先在 Franka 侧把一帧 back+side 的 rgb/depth 落盘，用双视角 client 做单次请求验证网络与协议；通过后再把 episode_dir 落盘并跑回放验收。

#### A) 单次请求（最小 smoke）
```bash
PYTHONPATH=/home/wangyuhan/MPPI/src \
python3 -m mppi.comm.ws_client_sync_pcl \
  --url ws://<CLOUD_IP>:9011 \
  --rgb-back /tmp/back.jpg \
  --depth-back /tmp/back_depth.npy \
  --rgb-side /tmp/side.jpg \
  --depth-side /tmp/side_depth.npy \
  --depth-unit-scale 1.0 \
  --step-id 0 \
  --request-timeout-s 10
```

#### B) episode_dir 回放（对齐 Terminal#133-134 结构）
```bash
EPISODE_DIR=/tmp/ep_franka_smoke \
DUAL_VIEW=1 \
URL=ws://<CLOUD_IP>:9011 \
START_IDX=0 \
MAX_STEPS=8 \
REQUEST_TIMEOUT_S=120 \
bash /home/wangyuhan/MPPI/tests/run_pw_replay_acceptance.sh replay obs_infl
```

预期：打印 `infer_ms=... policy=...`。

---

## Stage 4：离线性能扫参（PointWorld cost 可用性与时延）

目标：把 `pw_ms` 与 `infer_ms` 的瓶颈量化出来，确认 PointWorld cost 在当前硬件/模型设置下是否可用于在线联调。

通用约定：
- 统一使用同一条 episode：`EPISODE_DIR=/home/datasets/FrankaNav/ep_00152`，双视角 `DUAL_VIEW=1`
- 为了跳过 window warmup 的前 10 帧，建议默认 `START_IDX=10 MAX_STEPS=12`（能覆盖 window ready 后 2 帧以上）
- 每次测试至少看 3 个输出：
  - 终端每帧 `policy` 中的 `pw{0/1}:{reason}:{ms}`
  - `== Replay Summary ==` 的 `infer_ms.mean/p95/max`
  - `FINAL: PASS`

### 4.1 Baseline：no_pw vs obs_infl（量化 PW cost 纯开销）
目的：确认开启 PW cost 后 `infer_ms` 的增量，以及 `pw_ms` 的量级。

```bash
EPISODE_DIR=/home/datasets/FrankaNav/ep_00152 \
DUAL_VIEW=1 \
START_IDX=10 \
MAX_STEPS=12 \
bash /home/wangyuhan/MPPI/tests/run_pw_replay_acceptance.sh replay no_pw
```

预期输出：
- policy 中 `pw0:...`（no_pw 不应启用 cost term）
- `infer_ms.p95` 明显小于 obs_infl（作为 baseline）

```bash
EPISODE_DIR=/home/datasets/FrankaNav/ep_00152 \
DUAL_VIEW=1 \
START_IDX=10 \
MAX_STEPS=12 \
REQUEST_TIMEOUT_S=120 \
bash /home/wangyuhan/MPPI/tests/run_pw_replay_acceptance.sh replay obs_infl
```

预期输出：
- policy 中出现 `pw1:ok:<ms>ms`
- `FINAL: PASS`

### 4.2 扫 Ns：MPPI_PW_MAX_SCENE_POINTS（1024/512/256）
目的：验证 `pw_ms` 是否主要随 scene 点数线性增长，找到最小可用 Ns。

```bash
for NS in 1024 512 256; do
  echo "== Ns=${NS} =="
  EPISODE_DIR=/home/datasets/FrankaNav/ep_00152 \
  DUAL_VIEW=1 \
  START_IDX=10 \
  MAX_STEPS=12 \
  REQUEST_TIMEOUT_S=120 \
  MPPI_PW_MAX_SCENE_POINTS=${NS} \
  bash /home/wangyuhan/MPPI/tests/run_pw_replay_acceptance.sh replay obs_infl || exit 1
  echo ""
done
```

预期输出：
- Ns 降低后 `pw1:ok:<ms>ms` 应明显下降
- `FINAL: PASS` 始终为 PASS（不应再出现 task indices 越界）

### 4.3 扫 batch：MPPI_PW_EVAL_BATCH_SIZE（32/16/8）
目的：验证 cost model 在当前 GPU 上的吞吐最优 batch size。

```bash
for BS in 32 16 8; do
  echo "== eval_batch_size=${BS} =="
  EPISODE_DIR=/home/datasets/FrankaNav/ep_00152 \
  DUAL_VIEW=1 \
  START_IDX=10 \
  MAX_STEPS=12 \
  REQUEST_TIMEOUT_S=120 \
  MPPI_PW_EVAL_BATCH_SIZE=${BS} \
  bash /home/wangyuhan/MPPI/tests/run_pw_replay_acceptance.sh replay obs_infl || exit 1
  echo ""
done
```

预期输出：
- `pw_ms` 会随 BS 变化（可能存在甜点区间），选择 `pw_ms` 最低且稳定的配置。

### 4.4 置信度开关：MPPI_PW_USE_MODEL_CONFIDENCE / MPPI_PW_USE_TRACK_CONFIDENCE
目的：评估 confidence gating 对 `pw_ms` 和稳定性的影响（以及是否仍有 shape/contract 问题）。

```bash
EPISODE_DIR=/home/datasets/FrankaNav/ep_00152 \
DUAL_VIEW=1 \
START_IDX=10 \
MAX_STEPS=12 \
REQUEST_TIMEOUT_S=120 \
MPPI_PW_USE_MODEL_CONFIDENCE=1 \
MPPI_PW_USE_TRACK_CONFIDENCE=1 \
bash /home/wangyuhan/MPPI/tests/run_pw_replay_acceptance.sh replay obs_infl
```

```bash
EPISODE_DIR=/home/datasets/FrankaNav/ep_00152 \
DUAL_VIEW=1 \
START_IDX=10 \
MAX_STEPS=12 \
REQUEST_TIMEOUT_S=120 \
MPPI_PW_USE_MODEL_CONFIDENCE=0 \
MPPI_PW_USE_TRACK_CONFIDENCE=0 \
bash /home/wangyuhan/MPPI/tests/run_pw_replay_acceptance.sh replay obs_infl
```

预期输出：
- 两次都应 `FINAL: PASS`
- `pw1:ok:<ms>ms` 可能变化（用于判断 gating 开销与收益）

### 4.5 长跑稳定性（60 帧）
目的：排查偶发 exception、显存泄漏与时延发散。

```bash
EPISODE_DIR=/home/datasets/FrankaNav/ep_00152 \
DUAL_VIEW=1 \
START_IDX=0 \
MAX_STEPS=60 \
REQUEST_TIMEOUT_S=120 \
bash /home/wangyuhan/MPPI/tests/run_pw_replay_acceptance.sh replay obs_infl
```

预期输出：
- window ready 后持续 `pw1:ok:...ms`
- `FINAL: PASS`

### 4.6 加速路线图（基于 timing_breakdown 的稳定瓶颈）
现象（见 `data/pw_acceptance/<profile>/server/*.json` 的 `timing_breakdown`）：
- `t_pw_build_ms`：每帧稳定 ~3.3s（PointWorld build 持续重成本，并非 step10 warmup 偶发）
- `pw_ms`：每帧稳定 ~5–6s，且 `t_solver_ms ≈ pw_ms`（solver 的大头就是 PW cost term）
- 非瓶颈：`t_decode_ms≈0.1ms`、`t_cameras_ms≈10ms`、`t_pcd_ms≈6–8ms`（pcd_points ~10^5 级别也不慢）

瓶颈在代码中的对应位置：
- PW build（`t_pw_build_ms`）：`mppi/comm/ws_server_async_pcl.py` → `pw.push_and_maybe_build()` → `OnlineSceneFlowBuilder.build()` → `CoTrackerOnlinePointTracker.track_window()`（每相机一次，双视角会叠加）
- PW cost（`pw_ms`）：`mppi/mpc/solver.py` → `_rollout_cost()` 调 `pointworld_cost_fn()` → `mppi/pointworld_ext/wrapper.py` → `build_scene_features_torch()`（包含 `dist2robot` 的 `torch.cdist` 循环）+ 模型前向

#### 4.6.1 不改代码的“立刻可做”加速（建议先做这 4 个扫参）
A) 降 MPPI 样本数 K（对 `pw_ms` 近似线性，收益最大）
- 入口：server 读 `MPPI_NUM_SAMPLES`（默认 256）

```bash
for K in 256 128 64; do
  echo "== K=${K} =="
  EPISODE_DIR=/home/datasets/FrankaNav/ep_00152 \
  DUAL_VIEW=1 START_IDX=10 MAX_STEPS=12 \
  REQUEST_TIMEOUT_S=120 \
  MPPI_NUM_SAMPLES=${K} \
  bash /home/wangyuhan/MPPI/tests/run_pw_replay_acceptance.sh replay obs_infl || exit 1
  echo ""
done
```

B) 降 cost 侧 scene 点数 Ns（对 `dist2robot` 与模型前向都降负载）
- 入口：`MPPI_PW_MAX_SCENE_POINTS`（默认会被 cap 到 `min(model_contract, 1024)`）

```bash
for NS in 1024 512 256 128; do
  echo "== Ns=${NS} =="
  EPISODE_DIR=/home/datasets/FrankaNav/ep_00152 \
  DUAL_VIEW=1 START_IDX=10 MAX_STEPS=12 \
  REQUEST_TIMEOUT_S=120 \
  MPPI_PW_MAX_SCENE_POINTS=${NS} \
  bash /home/wangyuhan/MPPI/tests/run_pw_replay_acceptance.sh replay obs_infl || exit 1
  echo ""
done
```

C) 降 build 侧 query 点数（直接砍 `t_pw_build_ms`，并避免 build 4096 点但 cost 只用 1024 的浪费）
- 入口：`MPPI_PW_MAX_QUERY_POINTS_PER_CAMERA`（当前默认 2048 → 双视角总点数 4096）

```bash
for Q in 2048 1024 512 256; do
  echo "== Q_per_cam=${Q} =="
  EPISODE_DIR=/home/datasets/FrankaNav/ep_00152 \
  DUAL_VIEW=1 START_IDX=10 MAX_STEPS=12 \
  REQUEST_TIMEOUT_S=120 \
  MPPI_PW_MAX_QUERY_POINTS_PER_CAMERA=${Q} \
  bash /home/wangyuhan/MPPI/tests/run_pw_replay_acceptance.sh replay obs_infl || exit 1
  echo ""
done
```

D) 降 robot 点数 Nr（显著影响 `dist2robot` 计算量）
- 入口：`MPPI_PW_MAX_ROBOT_POINTS`（默认会被 cap 到 `min(model_contract, 256)`）

```bash
for NR in 256 128 64 32; do
  echo "== Nr=${NR} =="
  EPISODE_DIR=/home/datasets/FrankaNav/ep_00152 \
  DUAL_VIEW=1 START_IDX=10 MAX_STEPS=12 \
  REQUEST_TIMEOUT_S=120 \
  MPPI_PW_MAX_ROBOT_POINTS=${NR} \
  bash /home/wangyuhan/MPPI/tests/run_pw_replay_acceptance.sh replay obs_infl || exit 1
  echo ""
done
```

补充（吞吐 sweet spot）：继续沿用 4.3 的 `MPPI_PW_EVAL_BATCH_SIZE` 扫参，并可以额外试 `64`（显存够的话）。

#### 4.6.2 需要改代码的加速（如果目标是在线闭环，就要动这里）

目标：把离线回放里稳定暴露出来的两块重成本，变成“可控可降级”的在线可用版本。

落地顺序建议（先快收益、再做并行）：
1) B：`dist2robot=t0_repeat`（最直接砍 `pw_ms`，实现面最小、收益最大）
2) A：CoTracker iters/分辨率/fast path（直接砍 `t_pw_build_ms`）
3) D：多 GPU 并行（有多卡时分摊 `K`，进一步砍 `pw_ms`）

---

### A) `t_pw_build_ms`：CoTracker 加 iters/分辨率/fast path 可调开关

为什么慢（功能与成本来源）：
- PW build 需要在 window=11 的时序上，对双视角 query points 做跨帧跟踪（CoTracker），并在每帧做 2D→3D 回投与过滤融合；跟踪网络推理 + 多帧处理是主要成本。

改哪些代码（建议最小改动面）：
- `src/mppi/pointworld_ext/tracker_interface.py`（核心）：`CoTrackerOnlinePointTracker.track_window()`
- （可选）`src/mppi/pointworld_ext/scene_flow_builder.py`（若把缩放/坐标变换放到 build 侧做）

新逻辑应该是什么样（建议的开关与语义）：
- `MPPI_PW_COTRACKER_ITERS`：默认 6，可取 3/4/6；控制跟踪迭代次数
- `MPPI_PW_COTRACKER_FAST_PATH`：默认 0；=1 时尽量走更轻的 predictor 输出路径，避免重路径（以“质量换速度”）
- `MPPI_PW_COTRACKER_TRACK_HW`（或 `..._TRACK_SCALE`）：默认不缩放；设置后将 tracking 输入帧与 query points 缩放到更小分辨率跟踪，并把输出 uv_tracks 缩放回 contract 分辨率用于深度回投

风险与验收口径：
- 风险：iters/分辨率下降会降低跟踪质量，导致 `scene_exists` 稀疏、task 点数减少，进而影响 cost 稳定性/有效性，但不应 crash。
- 验收：
  - window ready 后仍稳定产出 `scene_flows/scene_exists/scene_visibility/scene_depth_valid_mask`
  - `task_n_obs/task_n_infl` 不长期为 0（除非确实不命中 AABB）
  - `timing_breakdown.t_pw_build_ms` 相比 baseline 明显下降

A 补充：robot mask 优化待办（表格版）

| 条目 | 改哪些代码 | 为什么 | 新逻辑应该是什么样 | 验收口径 |
|---|---|---|---|---|
| 去掉 per-point `cv2.circle` 循环 | `src/mppi/pointworld_ext/scene_flow_builder.py` 中 `_RobotMask2DBuilder.build_mask()` | 当前对每个投影点逐个 `cv2.circle` 是 Python 热点；step10 已看到 `ms_robot_mask0/ms_shift_robot_mask` 单次 ~0.4–0.55s | 先把投影点一次性栅格化到二值 mask，再用 `cv2.dilate` / `cv2.morphologyEx` 做膨胀与闭运算；不再逐点画圆 | `ms_robot_mask0/ms_shift_robot_mask` 明显下降；mask 形状/类型不变；query gating 不崩 |
| FK / mesh->world 对双视角共享 | `src/mppi/pointworld_ext/scene_flow_builder.py`；围绕 `_RobotMask2DBuilder` 增加“按时间点缓存 world points/FK 结果”的接口 | 当前 back/side 在同一时间点重复做 `visual_trimesh_fk + pts_local -> pts_world`，双视角重复计算 | 对 `t=0` 只算一次 FK/world points，供 back/side 共用；对 `t=shift` 也只算一次；每个相机只做投影与栅格化 | 双视角下 FK/world transform 相关耗时至少减半；`ms_robot_mask0/ms_shift_robot_mask` 同步下降 |
| shift mask 降频更新 | `src/mppi/pointworld_ext/scene_flow_builder.py` 中生成 `shift_robot_mask` 的位置 | 当前每帧每相机都算 `robot_mask0 + shift_robot_mask`，双视角共 4 次/帧，是 build 大头 | 新增开关（如 `MPPI_PW_SHIFT_MASK_UPDATE_EVERY`）；`robot_mask0` 保持每帧更新，`shift_robot_mask` 每 N 帧更新一次，其余帧复用缓存 | mask 调用次数从 4 次/帧降到约 2 次/帧；`t_pw_build_ms` 明显下降；稳定段 `task_n_obs/task_n_infl` 不显著恶化 |
| 缓存键与失效规则 | 同上 | 共享 FK/world points 与 shift mask 复用后，如果缓存失效规则不严，容易用错数据 | 缓存键至少覆盖：时间点（t0/shift）、相机分辨率、intrinsics/world2cam、seed；任何配置变化时强制失效 | 不出现“旧 mask 污染新帧”的异常；切 profile/改分辨率后行为正常 |
| timing 验证补强 | `src/mppi/pointworld_ext/scene_flow_builder.py` | 需要量化三项优化分别贡献多少收益，避免只看总时延 | 增加 `ms_fk_world_shared_*`、`ms_project_mask*`、`ms_rasterize_mask*`、`ms_morph_mask*`、`shift_mask_cache_hit/miss` 等字段 | 能直接从 acceptance json 看出收益来源，便于做 A/B test |

A 补充：shift mask 降频更新（仅验证 `update_every=2`）测试命令

- Terminal A（起 server，开启 shift mask 降频更新）：

```bash
cd /home/wangyuhan/MPPI
MPPI_PW_SHIFT_MASK_UPDATE_EVERY=2 \
bash /home/wangyuhan/MPPI/tests/run_pw_replay_acceptance.sh server obs_infl
```

- Terminal B（固定在 window ready 段，观察 build 侧收益）：

```bash
EPISODE_DIR=/home/datasets/FrankaNav/ep_00152 \
DUAL_VIEW=1 \
START_IDX=10 \
MAX_STEPS=12 \
REQUEST_TIMEOUT_S=120 \
bash /home/wangyuhan/MPPI/tests/run_pw_replay_acceptance.sh replay obs_infl
```

- 重点看 `data/pw_acceptance/obs_infl/server/*.json`：
  - `timing_breakdown.t_pw_build_ms`
  - `timing_breakdown.pw_build_breakdown.scene_build_breakdown.cams.<back|side>.ms_shift_robot_mask`
  - `task_n_obs / task_n_infl`

---

### B) `pw_ms`：`dist2robot` 近似模式（用于在线降本）

为什么慢（功能与成本来源）：
- `dist2robot` 特征会把“场景点到机器人点云的最近距离”编码进模型输入。
- 当前按时间步逐帧计算（每个 t 一次），并且对 MPPI 的每条候选轨迹（`B=K`）都计算，属于确定性重算。

改哪些代码（建议最小改动面）：
- `src/mppi/pointworld_ext/flows.py`：`build_scene_features_torch()`（必须）
- `src/mppi/pointworld_ext/flows.py`：`build_scene_features()`（建议同步，避免 torch/numpy 行为漂移）

新逻辑应该是什么样（保持模型输入维度不变）：
- 仅实现并强制启用 `t0_repeat`：只计算 t=0 的 dist2robot，然后沿 T 维 repeat 生成同 shape 的特征（把 T 次重算变成 1 次）。
- 强制离线/线上统一走 torch 特征构造（避免 torch/numpy 行为漂移）。

方案 B 优化后（t0_repeat）建议的对比测试命令：
- Terminal A（起 server，保持配置不变，用于和 baseline 对比）：

```bash
cd /home/wangyuhan/MPPI
bash /home/wangyuhan/MPPI/tests/run_pw_replay_acceptance.sh server obs_infl
```

- Terminal B（固定在 window ready 段对比 `pw_ms`）：

```bash
EPISODE_DIR=/home/datasets/FrankaNav/ep_00152 \
DUAL_VIEW=1 \
START_IDX=10 \
MAX_STEPS=12 \
REQUEST_TIMEOUT_S=120 \
bash /home/wangyuhan/MPPI/tests/run_pw_replay_acceptance.sh replay obs_infl
```

风险与验收口径：
- 风险：特征语义变化会影响策略质量，但应该保持数值稳定与不崩溃。
- 验收：
  - `pw_ms` 显著下降（尤其 `t0_repeat`）
  - cost 输出全 finite、无 shape/contract 错误
  - solver 仍稳定回包（可接受动作质量下降）

---

### D) 多 GPU 并行（有多卡时分摊 `K`）

为什么能加速：
- PointWorld cost 的计算主要沿 batch 维（`B=K` 候选轨迹）独立；多 GPU 可以把 `K` 分片到多个 replica 并行计算，再汇总回 CPU。

改哪些代码：
- 理想情况无需改代码：直接通过 env 把 model device 配成多卡（例如 `cuda:0,cuda:1`）。
- 为了可验收与防踩坑，建议补强：
  - 将“实际使用的设备列表/分片策略”写入 server 落盘摘要（复现实验时一眼确认是否生效）
  - 若检测到不合法 device 字符串（例如 `cuda` 不带 index），直接 fail-fast

建议的部署形态（减少 GPU 争抢）：
- CoTracker 与 cost 分离到不同卡：
  - `MPPI_PW_COTRACKER_DEVICE=cuda:0`
  - `MPPI_PW_MODEL_DEVICE=cuda:1,cuda:2`（视机器卡数而定）
  - `MPPI_PW_ROBOT_SAMPLER_DEVICE` 视情况跟随 model device 或固定到其中一张卡

验收口径：
- `pw_ms` 相比单卡下降，且能观察到多卡同时有负载；server 摘要里能核对到多 device 生效。

---

## Stage 5：Franka↔云端联合测试（分级推进，双视角）

### 需要修改/确认的代码点（双视角）
- `mppi.comm.ws_server_async_pcl`：确认 server 端对 `ObsPCL.cameras` 的 back+side 融合点云输入已经生效（用于 `mppi_joint` 的 `pcd_back_cam` 实际是 back+side 拼接后的 base 点云）。
- `mppi.comm.ws_client_sync_pcl`：已增参支持 `--rgb-back/--depth-back/--rgb-side/--depth-side`，并按 `ObsPCL.cameras` 回传；需要验证与 server 的强制双视角校验一致。
- `tests/pw_replay_acceptance.py`：仍作为 episode_dir 回放 reference，实现 back+side 的 `ObsPCL.cameras` 构造。

### Todo 字段清单（双视角）
- episode_dir 结构：`back/ back_depth/ side/ side_depth/ data.pkl`
- 每帧必需：
  - `step_id` 单调递增
  - `t_client_send_ns` 单调递增
  - `q`（7 dof）+ `gripper`
  - `cameras.back.rgb_bytes/depth_bytes` + shape + codec + depth_unit_scale
  - `cameras.side.rgb_bytes/depth_bytes` + shape + codec + depth_unit_scale
- server 侧标定：
  - `MPPI_PCL_CAM_INFO_BACK_PATH` / `MPPI_PCL_T_BASE_CAM_BACK_PATH`
  - `MPPI_PCL_CAM_INFO_SIDE_PATH` / `MPPI_PCL_T_BASE_CAM_SIDE_PATH`

### Level 1：dummy_hold（只测通信稳定性）
云端：`--policy dummy_hold`。

通过标准：连续运行（建议 10 分钟）0 超时、0 断连、server 不崩。

### Level 2：mppi_joint（测推理耗时与动作输出）
云端：用 `scripts/test_cuRobo_pcl.sh` 起 server（policy=mppi_joint）。

通过标准：
- actions shape 正确且数值有限（无 nan/inf）
- infer_ms p95 在你能接受的范围

### Level 3：PointWorld window + 三档 profile 切换
云端：必须 horizon=11，且 PointWorld 必要 ckpt/model/urdf 到位；用 `tests/run_pw_replay_acceptance.sh` 的同款环境变量策略。

通过标准：
- server 不出现频繁 reset/降级（`runtime_policy` 可用于定位）
- `MPPI_PW_ACCEPTANCE_DUMP_DIR` 打开时持续产出 server 摘要 json，字段齐全
- `no_pw/obs_only/obs_infl` 三档切换不引入新错误

---

## 输出位置（验收收口）
默认输出目录：

- server 摘要：`/home/wangyuhan/MPPI/data/pw_acceptance/<profile>/server/*.json`
- client 汇总：`/home/wangyuhan/MPPI/data/pw_acceptance/<profile>/client_report.json`

验收脚本结束输出：
- `FINAL: PASS` 或 `FINAL: FAIL`