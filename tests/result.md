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