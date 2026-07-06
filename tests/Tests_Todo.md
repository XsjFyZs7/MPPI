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

### 2.1 一键：server + replay + acceptance（原生 episode，双视角）
调试：PointWorld window/tracking + 关键字段稳定产出

episode 目录结构（与 Terminal#133-134 对齐）：
- `back/` + `back_depth/`
- `side/` + `side_depth/`
- `data.pkl`

```bash
EPISODE_DIR=/home/datasets/FrankaNav/ep_00152 \
DUAL_VIEW=1 \
bash /home/wangyuhan/MPPI/tests/run_pw_replay_acceptance.sh all obs_infl
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
MPPI_PW_USE_MODEL_CONFIDENCE=1 \
MPPI_PW_USE_TRACK_CONFIDENCE=1 \
bash /home/wangyuhan/MPPI/tests/run_pw_replay_acceptance.sh replay obs_infl
```

```bash
EPISODE_DIR=/home/datasets/FrankaNav/ep_00152 \
DUAL_VIEW=1 \
START_IDX=10 \
MAX_STEPS=12 \
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