# MPPI

本仓库提供一个面向 Franka 机械臂的 MPPI（Model Predictive Path Integral）关节空间推理服务，并带两套通信与场景输入链路：

- V1 链路：客户端直接发送（可选）点云 `pcd_back_cam`；服务器返回开环关节轨迹动作。
- PCL 链路：客户端发送 RGB+Depth（可压缩）；服务器在线反投影生成点云、构建碰撞场景（cuboids）并可用 cuRobo 做 GPU 碰撞距离/代价。

本 README 的目标是让同事能快速定位需要修改/测试的代码：通信功能检测、在线点云构建、以及 cuRobo 碰撞箱/碰撞计算的 GPU 加速与验证。

---

## 1. 快速上手（建议同事按这里跑通）

### 1.1 基础：设置 PYTHONPATH
```bash
export PYTHONPATH=/home/wangyuhan/MPPI/src:$PYTHONPATH
```

### 1.2 启动 V1 server（端口 9010）
- dummy_hold：不跑推理，只回传 hold 动作（用于纯通信测试）
```bash
python3 -m mppi.cli server --host 0.0.0.0 --port 9010 --open-loop-horizon 8 --policy dummy_hold
```

- mppi_joint：跑 MPPI 推理（可选接 cuRobo/场景）
```bash
python3 -m mppi.cli server --host 0.0.0.0 --port 9010 --open-loop-horizon 8 --policy mppi_joint
```

### 1.3 V1 client（发送 q/gripper/可选 pcd）
```bash
python3 -m mppi.cli client --url ws://127.0.0.1:9010 --run-seconds 10 --control-hz 20 --open-loop-horizon 8
```
如果要固定点云输入（npz 内含 points/colors）：
```bash
python3 -m mppi.cli client --url ws://127.0.0.1:9010 --pcd-npz /home/wangyuhan/MPPI/data/test/scene_points.ply
```
注：V1 client 的点云加载默认读 npz（见 `src/mppi/comm/ws_client_sync.py::_load_pcd_npz`），如果你提供的是 ply，需要先转换或改代码/脚本。

### 1.4 启动 PCL server（端口 9011，RGB+Depth -> 点云 -> 场景）
建议直接用现成脚本配置环境变量：
```bash
bash /home/wangyuhan/MPPI/scripts/test_cuRobo_pcl.sh
```
脚本最终执行：
```bash
python3 -m mppi.comm.ws_server_async_pcl --host 0.0.0.0 --port 9011 --open-loop-horizon 8 --policy mppi_joint --cam-id back
```

### 1.5 PCL client（发送 RGB+Depth）
- 单次请求（本地文件 rgb/depth）：
```bash
python3 -m mppi.comm.ws_client_sync_pcl --url ws://127.0.0.1:9011 --rgb <rgb_path> --depth <depth_path> --cam-id back --print-actions
```

- 回放数据集（json 里引用 images/depths 路径）：
```bash
python3 /home/wangyuhan/MPPI/scripts/playback_client_pcl.py --url ws://127.0.0.1:9011 --json <data.json> --data-root <data_root>
```

---

## 2. 目录结构与职责

### 2.1 顶层目录
- `src/`：核心 Python 包（mppi），推理/通信/场景构建均在这里
- `scripts/`：启动/测试/回放脚本（用于复现实验与性能压测）
- `configs/`：相机与外参标定、变换矩阵等配置（YAML）
- `data/`：示例数据（npz/ply/json 等）
- `tests/`：实验记录与少量测试脚本/占位内容
- `third_party/`：外部第三方代码（例如 co-tracker）

### 2.2 `src/mppi/`（核心包）
- `src/mppi/cli.py`
  - 统一入口：`mppi server` / `mppi client` / `mppi curobo-smoke`
  - 同事做“通信功能检测”时优先用这里的 server/client 快速验证链路
- `src/mppi/comm/`
  - 通信层（websocket + msgpack）
  - **V1：**
    - `ws_server_async.py`：服务端（schema_version=1），接收 `infer_request`，返回 `infer_response`
    - `ws_client_sync.py`：客户端/执行器（周期性请求、统计 RTT/jitter）
  - **PCL：**
    - `ws_server_async_pcl.py`：服务端（schema_version=100），接收 RGB/Depth，在线反投影生成点云，再走 mppi 推理/碰撞
    - `ws_client_sync_pcl.py`：客户端（把 RGB 编成 JPEG，把 Depth 编成 npy+zlib 发给服务端）
- `src/mppi/protocol/`
  - `types.py`：V1 协议数据结构（ObsV1/InferRequestV1/InferResponseV1/ErrorV1）
  - `types_pcl.py`：PCL 协议数据结构（ObsPCL/InferRequestPCL/InferResponsePCL/ErrorPCL）
  - `msgpack_codec.py`：msgpack 编解码（通信层所有消息都走这里）
- `src/mppi/mpc/solver.py`
  - `JointMPPISolver`：MPPI 核心推理（采样、代价、退化策略、时间预算）
  - 场景构建入口：`build_scene_cuboids_from_pcd_back_cam(...)`
  - cuRobo 碰撞入口：`get_curobo_collision_checker(...).batch_distance(...)` / `collision_penalty(...)`
  - 同事做“curobo 碰撞箱 GPU 加速/验证”主要会改这里与 `curobo_ext/`
- `src/mppi/curobo_ext/`
  - `collision_checker.py`
    - cuRobo 的封装与缓存（按 scene key 缓存 RobotCollisionChecker）
    - `batch_distance()` 会把 q_traj 放到 GPU 并取回距离矩阵
    - `get_robot_spheres_base()` 可生成 robot mask spheres（用于点云去掉机器人本体点）
  - `scene_builder.py`
    - 点云 -> base 坐标 -> ROI crop -> 去桌面/去墙 -> voxel downsample -> robot mask
    - 点云聚类得到 AABB，再转成 cuboids（给 cuRobo 当 world obstacles）
  - `check_depth_pcl.py`
    - PCL 链路的 RGBD->点云：`rgbd_to_pointcloud_base(...)`
    - 解析 intrinsics / T_base_cam（来自 ObsPCL 或环境变量）
  - `check_depth.py`
    - 读入/预处理 depth/rgb 的工具函数（PCL client 与调试脚本会用）
- `src/mppi/utils/`
  - `pointcloud.py`：纯 numpy 点云工具（voxel、AABB、聚类、mask 等）；在线点云构建的 CPU 热点很可能在这里
  - `se3.py`：SE(3) 相关工具（坐标变换）
- `src/mppi/robots/`
  - `franka_kinematics.py`：Franka 正运动学（代价项 ee/link7 位置等）
- `src/mppi/costs/`
  - `ee_pose.py`：末端/指定 link 的位置代价（MPPI cost term）

### 2.3 `configs/`（相机/外参）
- `configs/back_cam_info.yaml`：相机内参（PCL server 用来解析 intrinsics）
- `configs/T_base_cam.yaml`：base->camera 外参 4x4（PCL server 用来把点云变到 base）
- `configs/T_ee_wrist.yaml`、`configs/wrist_can_info.yaml`：其他标定信息（按实际 pipeline 使用）

### 2.4 `scripts/`（用于同事测试/复现）
- `scripts/test_cuRobo.sh`
  - 以 V1 server 跑 `mppi_joint`，并通过环境变量打开：从点云建场景 + cuRobo 碰撞
- `scripts/test_cuRobo_pcl.sh`
  - 以 PCL server 跑 `mppi_joint`，并通过环境变量打开：RGBD->点云->建场景 + cuRobo
- `scripts/playback_client.py`
  - V1 回放：从 json 里取 joint_states，再请求 server，统计 infer_ms/policy/key
- `scripts/playback_client_pcl.py`
  - PCL 回放：从 json 里取图像/深度路径，打包成 ObsPCL 请求 server
- `scripts/run_server_gpu.sh`、`scripts/run_client_cpu.sh`
  - 当前内容疑似不完整（文件里只有 `export`），建议同事优先按本 README 的命令启动，或先补全脚本再使用

---

## 3. 通信协议（同事做通信检测重点关注）

### 3.1 V1（schema_version = 1）
- 请求 envelope：
  - `type = "infer_request"`
  - payload = `ObsV1`（包含 `q/gripper/step_id/t_client_send_ns`，可选 `pcd_back_cam`）
- 响应 envelope：
  - `type = "infer_response"`
  - payload = `ActionChunkV1`（包含 actions、时戳、server_timing）
- 错误：
  - `type = "error"`，payload 包含 `code/message`

代码位置：
- `src/mppi/protocol/types.py`
- server：`src/mppi/comm/ws_server_async.py`
- client：`src/mppi/comm/ws_client_sync.py`

### 3.2 PCL（schema_version = 100）
- 请求 envelope：
  - `type = "infer_request_pcl"`
  - payload = `ObsPCL`：支持 `rgb_bytes(jpeg)` + `depth_bytes(npy_zlib)`，以及 `cam_id`/`intrinsics`/`T_base_cam`
- 响应 envelope：
  - `type = "infer_response_pcl"`
- 错误：
  - `type = "error_pcl"`

代码位置：
- `src/mppi/protocol/types_pcl.py`
- server：`src/mppi/comm/ws_server_async_pcl.py`
- client：`src/mppi/comm/ws_client_sync_pcl.py`

---

## 4. 同事任务：应该改哪些代码、怎么测

### 4.1 通信功能检测（优先级最高）
目标：确认 websocket + msgpack 的编解码、协议字段、超时/异常路径稳定。

建议测试项（最小闭环）：
1) dummy_hold：server 能收包、回包、client 能解析 actions
   - server：`mppi.cli server --policy dummy_hold`
   - client：`mppi.cli client --url ...`
2) schema/version/type 错误的容错
   - 在 client 侧构造错误 `schema_version/type`，应收到 `error`/`error_pcl`
   - 关注代码：`ws_server_async.py::_handle_connection`、`ws_server_async_pcl.py::_handle_connection`
3) payload 边界与 max_size
   - PCL 模式发送大图/深度是否会被 websocket 限制（server/client 使用 `max_size=None`）
4) RTT/抖动统计
   - 关注：`ws_client_sync.py` 的 `chunk_rtt_ms/jitter_ms` 汇总打印

你需要同事主要会修改/新增的位置：
- `src/mppi/comm/ws_server_async.py`
- `src/mppi/comm/ws_server_async_pcl.py`
- `src/mppi/protocol/msgpack_codec.py`
- `src/mppi/protocol/types.py` / `types_pcl.py`
- `src/mppi/comm/ws_client_sync.py` / `ws_client_sync_pcl.py`

### 4.2 在线点云构建（RGBD -> 点云 -> 过滤/聚类）
目标：把 PCL server 的在线点云链路跑稳、并明确性能瓶颈。

核心链路（PCL server 内）：
- 解码 RGB/Depth：`ws_server_async_pcl.py::_decode_rgb/_decode_depth`
- 反投影生成点云（base 坐标）：`curobo_ext/check_depth_pcl.py::rgbd_to_pointcloud_base`
- 点云过滤/降采样/robot mask：`curobo_ext/scene_builder.py::build_scene_points_base_and_colors_from_pcd_back_cam`
- 点云聚类 -> cuboids：`curobo_ext/scene_builder.py::build_scene_cuboids_from_*` + `utils/pointcloud.py::cluster_points_to_aabbs`

建议同事测试与可视化：
- 开启保存中间结果（用于离线复盘/可视化）：
  - 在 `scripts/test_cuRobo_pcl.sh` 里设置 `MPPI_PCL_SAVE_PCD=1`，并指定 `MPPI_PCL_SAVE_PCD_OUT`
  - server 会保存 npz（包含 points_filtered/colors_filtered/cuboid 信息/robot_spheres）
- 如果要验证 robot mask 是否正确，优先看：
  - `mppi.curobo_ext.collision_checker::get_robot_spheres_base`
  - `mppi.curobo_ext.scene_builder::mask_robot_points` / `build_scene_points_base_and_colors_from_pcd_back_cam`

你需要同事主要会修改/优化的位置：
- `src/mppi/curobo_ext/check_depth_pcl.py`
- `src/mppi/curobo_ext/scene_builder.py`
- `src/mppi/utils/pointcloud.py`

### 4.3 cuRobo 碰撞箱/碰撞计算的 GPU 加速与验证
目标：确保 cuRobo 路径可用（GPU 环境），并对“碰撞距离/代价”做性能优化与正确性检查。

关键入口：
- 构建 checker + 缓存：`src/mppi/curobo_ext/collision_checker.py::get_curobo_collision_checker`
- 计算距离矩阵：`CuRoboCollisionChecker.batch_distance`
- MPPI 中使用距离：`src/mppi/mpc/solver.py`（cost terms 与 debug stats）

建议同事验证方法：
1) 先跑 smoke test（只测 cuRobo 能否在当前 URDF/tool_frame 下工作）：
   - `python3 -m mppi.cli curobo-smoke --device cuda:0 --robot-yaml franka.yml --urdf <urdf> --tool-frame <link>`
2) 再跑 server 推理并观察 policy 字符串中的 `curobo/nocurobo`、`dmin_scene/dmin_self` 等调试字段：
   - `ws_server_async.py` 有 debug_cost_stats 输出拼到 policy
3) 性能优化方向（同事改动点）：
   - `collision_checker.py` 当前在 CUDA 情况下每次都会 `torch.cuda.synchronize(...)`（batch_distance/robot_spheres），这会强制同步影响吞吐；需要结合 profiling 决定是否降低同步频率或只在 debug 时同步
   - scene key 缓存策略：scene cuboids 每帧波动会导致 checker 频繁重建；可考虑量化/稳定化 cuboids 或复用 scene model
   - MPPI 的时间预算退化策略在 `JointMPPISolver.infer_actions` 内（用于保证实时性），优化 GPU 部分后可相应调参

你需要同事主要会修改/测试的位置：
- `src/mppi/curobo_ext/collision_checker.py`
- `src/mppi/mpc/solver.py`

---

## 5. 关键环境变量（跑 PCL/curobo 时必看）

以下变量在 `ws_server_async.py::_get_joint_solver` 与 `scripts/test_cuRobo*.sh` 中使用，影响推理与场景构建：

- cuRobo 开关与权重：
  - `MPPI_USE_CUROBO_COLLISION=1`
  - `MPPI_W_SCENE_COLLISION=...`
  - `MPPI_W_SELF_COLLISION=...`
  - `MPPI_CUROBO_DEVICE=cuda:0`
  - `MPPI_CUROBO_ROBOT_YAML=franka.yml`
  - `MPPI_URDF_PATH=...`
  - `MPPI_EE_LINK=...`

- 场景（点云 -> cuboids）：
  - `MPPI_SCENE_FROM_PCD_BACK_CAM=1`
  - `MPPI_T_BASE_CAM_BACK_PATH=/home/wangyuhan/MPPI/configs/T_base_cam.yaml`
  - `MPPI_SCENE_ROI_MIN='x,y,z'` / `MPPI_SCENE_ROI_MAX='x,y,z'`
  - `MPPI_SCENE_VOXEL_SIZE_M=0.01`
  - `MPPI_SCENE_MIN_CLUSTER_VOXELS=...`
  - table/wall 过滤相关：`MPPI_SCENE_REMOVE_TABLE_POINTS`、`MPPI_SCENE_REMOVE_WALL_POINTS` 等

- PCL server（RGBD -> 点云）：
  - `MPPI_PCL_CAM_INFO_BACK_PATH=/home/wangyuhan/MPPI/configs/back_cam_info.yaml`
  - `MPPI_PCL_T_BASE_CAM_BACK_PATH=/home/wangyuhan/MPPI/configs/T_base_cam.yaml`
  - `MPPI_PCL_DEPTH_UNIT_SCALE=1.0`
  - `MPPI_PCL_DEPTH_MIN_M` / `MPPI_PCL_DEPTH_MAX_M` / `MPPI_PCL_STRIDE`
  - 保存调试：`MPPI_PCL_SAVE_PCD=1`，`MPPI_PCL_SAVE_PCD_OUT=...`

---

## 6. 建议的交付物（给同事的明确输出）

为了让“通信检测 / 在线点云 / GPU 加速”工作可验收，建议同事最终提交：

1) 通信测试脚本/用例：
- 能覆盖 V1 与 PCL 两种 schema
- 覆盖正常与错误路径（bad schema/type/缺字段/超时）
- 输出 RTT、服务器 infer_ms、错误码统计

2) 在线点云构建的验证与性能报告：
- 至少给出 1 组输入（RGB+Depth 或回放 json），保存 npz 中间结果
- ROI/table/wall/robot mask 参数的推荐值与可复现实验命令

3) cuRobo GPU 加速改动与对比：
- smoke test 可跑通的命令
- 加速前后（或不同同步策略）的吞吐/延迟对比（以 infer_ms 或端到端 RTT 为准）
- 正确性：碰撞距离的基本 sanity check（例如 dmin_scene/dmin_self 的范围与变化趋势）

---