# MVP 五个子任务测试结果（未加速 baseline）

本文件记录“当前未做加速/未做降采样扫参”情况下的耗时表现，用于后续对比。

## A) PointWorld offline 瓶颈（obs_infl，window ready 后稳定段）

结论：瓶颈非常稳定，主要卡在两块。

- PW build（`t_pw_build_ms`）：负责把最近一段双视角 RGBD 观测和机器人状态整理成 PointWorld 可用的时序场景表示。它耗时长，是因为要连续处理多帧数据，并完成跨帧跟踪、三维重建、筛选和融合；每帧稳定 ~3.34–3.60 秒。
- PW cost（`pw_ms`）：负责基于上一步得到的时序场景信息，对 MPPI 采样出的候选轨迹逐条计算任务代价，用来决定哪条动作更优。它耗时长，是因为要对大量候选轨迹做特征构造、模型推理和代价汇总；每帧稳定 ~4.96–6.19 秒，且 `t_solver_ms ≈ pw_ms`，说明 solver 的主要时间基本都花在这一步。

逐帧耗时（ms）：
- step 10：`t_pw_build_ms=3597.95`，`pw_ms=4591.55`
- step 11：`t_pw_build_ms=3431.46`，`pw_ms=4959.87`
- step 12：`t_pw_build_ms=3351.19`，`pw_ms=6185.13`
- step 13：`t_pw_build_ms=3352.64`，`pw_ms=5106.23`
- step 14：`t_pw_build_ms=3339.92`，`pw_ms=5130.77`
- step 15：`t_pw_build_ms=3478.13`，`pw_ms=5194.90`

哪些不是瓶颈（确认可忽略）：
- `t_decode_ms ≈ 0.14ms`
- `t_cameras_ms ≈ 11ms`
- `t_pcd_ms ≈ 5.6–8.3ms`，`pcd_points ≈ 10.7 万`（反投影 + 拼接不慢）

## B) 三项耗时优化结果（dist2robot + robot mask + 多 GPU）

### B.1 dist2robot（PointWorld cost 特征构造）
- 优化点：`dist2robot` 改为 `t0_repeat`（只算 `t=0` 一次，然后沿 `T` 维 repeat），离线/线上统一走 torch 特征构造。
- 优化前：`dist2robot` 属于确定性重算（逐 `t` 计算），在 profiling 中曾出现数百 ms 级波动风险。
- 优化后：稳定段 `t_dist2robot_ms ≈ 11ms/帧`（单 GPU；`K=256`）。

### B.2 robot mask（PW build 的 seed mask）
- 优化点：
  - 去掉逐点 `cv2.circle` 循环，改为“批量栅格化 + dilate/close”。
  - `FK/mesh->world` 对双视角共享（同一时间点只算一次）。
  - `shift mask` 支持 `update_every=2` 降频更新（命中缓存帧直接复用）。
- 优化前：单次 `robot_mask0/shift_robot_mask` 约 `0.4–0.55s`，双视角每帧合计接近 `~2s/帧`。
- 优化后：稳定段单相机 `ms_robot_mask0 ≈ 9–16ms`；`ms_shift_robot_mask` 在命中缓存帧可降到 `~0.05ms`。

### B.3 build / cost / 端到端 对比（K=256，obs_infl）
- 单 GPU（做完 dist2robot + robot mask 之后，window ready 稳定段）：
  - `t_pw_build_ms ≈ 1.64–1.69s/帧`
  - `pw_ms ≈ 4.48–5.01s/帧`（大头仍是模型前向）
  - 端到端约 `~6.2–6.7s/帧`
- 多 GPU（2 卡分片，`cuda:0,cuda:1`）：
  - `pw_ms ≈ 2.87s/帧`，`t_solver_ms ≈ 2.92s/帧`
  - `t_pw_build_ms ≈ 0.88s/帧`
  - 端到端约 `~3.8s/帧`
  - 相比单 GPU cost：`pw_ms` 加速约 `1.6–1.7×`（非线性，受 build/prepare/reduce/调度开销影响）

### B.4 当前新瓶颈（优化后）
- build 侧：`ms_track ≈ 0.77s/相机`（双视角约 `~1.5s/帧`），是 `t_pw_build_ms` 的最大头。
- cost 侧：模型前向 `t_model_ms` 仍是 `pw_ms` 最大头（2 卡时每卡约 `2.7–2.8s`）。

## 1) 端侧闭环执行时序（按频率消费 action chunk）

```bash
== Playback Summary ==
steps: 200
infer_ms mean: 44.94904888328165
infer_ms p50 : 44.482744531705976
infer_ms p95 : 46.67431563138961
infer_ms max : 67.27478886023164
unique policies: 65
   mppi_joint+curobo+ess0.830+tab1+cub3+sph41
   mppi_joint+curobo+ess0.831+tab1+cub3+sph41
   mppi_joint+curobo+ess0.843+tab1+cub3+sph41
   mppi_joint+curobo+ess0.844+tab1+cub3+sph41
   mppi_joint+curobo+ess0.848+tab1+cub3+sph41
   mppi_joint+curobo+ess0.849+tab1+cub3+sph41
   mppi_joint+curobo+ess0.850+tab1+cub3+sph41
   mppi_joint+curobo+ess0.851+tab1+cub3+sph41
   mppi_joint+curobo+ess0.858+tab1+cub3+sph41
   mppi_joint+curobo+ess0.859+tab1+cub3+sph41
```

## 2) cuRobo 自碰 self-collision 的启用与有效性

```bash
== Playback Summary ==
steps: 50
infer_ms mean: 6.495910361409187
infer_ms p50 : 6.320232525467873
infer_ms p95 : 7.446401030756532
infer_ms max : 7.746322080492973
unique policies: 20
   mppi_joint+curobo+ess0.946+tab0+cub0+sph0
   mppi_joint+curobo+ess0.950+tab0+cub0+sph0
   mppi_joint+curobo+ess0.951+tab0+cub0+sph0
   mppi_joint+curobo+ess0.952+tab0+cub0+sph0
   mppi_joint+curobo+ess0.955+tab0+cub0+sph0
   mppi_joint+curobo+ess0.956+tab0+cub0+sph0
   mppi_joint+curobo+ess0.957+tab0+cub0+sph0
   mppi_joint+curobo+ess0.958+tab0+cub0+sph0
   mppi_joint+curobo+ess0.959+tab0+cub0+sph0
   mppi_joint+curobo+ess0.960+tab0+cub0+sph0
```

## 3) 更新策略必要性验证（A/B 场景交替导致的闪烁）

```bash
== Playback Summary ==
steps: 200
infer_ms mean: 93.65502858301625
infer_ms p50 : 93.63348898477852
infer_ms p95 : 95.0465643312782
infer_ms max : 96.81511297821999
unique policies: 88
   mppi_joint+curobo+ess0.735+tab1+cub4+sph41
   mppi_joint+curobo+ess0.736+tab1+cub4+sph41
   mppi_joint+curobo+ess0.751+tab1+cub4+sph41
   mppi_joint+curobo+ess0.762+tab1+cub4+sph41
   mppi_joint+curobo+ess0.763+tab1+cub4+sph41
   mppi_joint+curobo+ess0.766+tab1+cub4+sph41
   mppi_joint+curobo+ess0.769+tab1+cub4+sph41
   mppi_joint+curobo+ess0.770+tab1+cub4+sph41
   mppi_joint+curobo+ess0.771+tab1+cub4+sph41
   mppi_joint+curobo+ess0.772+tab1+cub4+sph41
```

## 4) （缺失）

该条在当前文件中没有对应的回放输出记录；如需要补齐，请把第 4 项的 `== Playback Summary ==` 原始输出追加进来。

## 5) time budget 压力测试（为降级策略提供依据）

```bash
== Playback Summary ==
steps: 200
infer_ms mean: 23.052096699830145
infer_ms p50 : 22.483322769403458
infer_ms p95 : 24.24152479507029
infer_ms max : 66.26260187476873
unique policies: 51
   mppi_joint+curobo+ess0.893+tab1+cub7+sph41
   mppi_joint+curobo+ess0.896+tab1+cub7+sph41
   mppi_joint+curobo+ess0.901+tab1+cub7+sph41
   mppi_joint+curobo+ess0.903+tab1+cub7+sph41
   mppi_joint+curobo+ess0.905+tab1+cub7+sph41
   mppi_joint+curobo+ess0.906+tab1+cub7+sph41
   mppi_joint+curobo+ess0.908+tab1+cub7+sph41
   mppi_joint+curobo+ess0.909+tab1+cub7+sph41
   mppi_joint+curobo+ess0.910+tab1+cub7+sph41
   mppi_joint+curobo+ess0.911+tab1+cub7+sph41
```