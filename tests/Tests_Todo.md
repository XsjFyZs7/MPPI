# PointWorld 本地回放验收

## 目标
用同一套标准入口完成三件事：

- 启动 PCL + PointWorld server
- 回放 `data.json` 或原生 `episode_dir`
- 自动检查 server 端是否稳定产出：
  - `scene_flows`
  - `scene_visibility`
  - `scene_depth_valid_mask`
  - `task_n_obs`
  - `task_n_infl`（仅 `obs_infl`）
  - `runtime_policy`

## 文件
- 启动脚本：`/home/wangyuhan/MPPI/scripts/run_pw_replay_acceptance.sh`
- 回放脚本：`/home/wangyuhan/MPPI/scripts/pw_replay_acceptance.py`

## Profile
- `no_pw`
  - 仍构建 PointWorld 观测链路，但 task ablation 关闭
- `obs_only`
  - 只启用 `I_obs`
- `obs_infl`
  - 启用 `I_obs + I_infl`

## 最小前置条件
- `MPPI_PW_COTRACKER_CKPT` 可用
- `MPPI_PW_MODEL_PATH` 可用
- `POINTWORLD_ROOT` 可用
- `configs/pointworld_static_aabbs.json` 已固定
- `ws_server_async_pcl.py` 已接受验收落盘 diff

## 推荐命令

### 1. 原生 episode 双视角回放
```bash
EPISODE_DIR=/home/datasets/FrankaNav/ep_00152 \
DUAL_VIEW=1 \
bash /home/wangyuhan/MPPI/scripts/run_pw_replay_acceptance.sh all obs_infl
```

### 2. data.json 单视角回放
```bash
JSON_PATH=/home/wangyuhan/MPPI/data/test/data.json \
DATA_ROOT=/home/datasets/FrankaNav/test \
DUAL_VIEW=0 \
bash /home/wangyuhan/MPPI/scripts/run_pw_replay_acceptance.sh all obs_only
```

### 3. 只起 server
```bash
bash /home/wangyuhan/MPPI/scripts/run_pw_replay_acceptance.sh server obs_infl
```

### 4. 只做 replay
```bash
EPISODE_DIR=/home/datasets/FrankaNav/ep_00152 \
DUAL_VIEW=1 \
bash /home/wangyuhan/MPPI/scripts/run_pw_replay_acceptance.sh replay obs_infl
```

## 输出
默认输出目录：

- server 摘要：`/home/wangyuhan/MPPI/data/pw_acceptance/<profile>/server/*.json`
- client 汇总：`/home/wangyuhan/MPPI/data/pw_acceptance/<profile>/client_report.json`

## 通过标准
- 所有 server 摘要都包含：
  - `has_scene_flows = true`
  - `has_scene_visibility = true`
  - `has_scene_depth_valid_mask = true`
  - `has_runtime_policy = true`
- `obs_only` / `obs_infl`：
  - `has_task_n_obs = true`
- `obs_infl`：
  - `has_task_n_infl = true`

脚本结束时会输出：

- `FINAL: PASS`
- 或 `FINAL: FAIL`

## 备注
- 这套验收不改 websocket 协议，只依赖 server 本地落盘的摘要 JSON。
- 如果需要继续把验收接到 CI，可以直接复用 `client_report.json` 和 `server/*.json`。