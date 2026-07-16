from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import os
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import torch.nn as nn

from mppi.costs.pointworld_cost import (
    PointWorldCostConfig,
    reduce_pointworld_cost_torch,
)
from mppi.pointworld_ext.flows import (
    RobotFlowAdapter,
    build_robot_inputs,
    build_scene_features_torch,
    normalize_dist2robot_mode,
    prepare_scene_inputs,
)
from mppi.utils.paths import default_pointworld_root, default_urdf_path, ensure_sys_path_for_runtime


@dataclass(frozen=True)
class _PointWorldReplica:
    device: str
    model: Any
    robot: RobotFlowAdapter


@dataclass(frozen=True)
class PointWorldModelConfig:
    checkpoint_path: str
    device: str = "cuda"
    domain: Optional[str] = None
    urdf_path: str = field(default_factory=default_urdf_path)
    max_scene_points: Optional[int] = None
    max_robot_points: Optional[int] = None
    robot_sampler_device: Optional[str] = None
    robot_gripper_only: bool = True
    seed: int = 1
    disable_compile: bool = True
    eval_batch_size: int = 32
    dist2robot_mode: str = "t0_repeat"
    cost: PointWorldCostConfig = field(default_factory=PointWorldCostConfig)


class PointWorldCostModel:
    def __init__(self, cfg: PointWorldModelConfig) -> None:
        self.cfg = cfg
        self._torch = self._require_torch()
        self._checkpoint = self._load_checkpoint(cfg.checkpoint_path)
        self._model_contract, self._data_contract = self._read_contract(self._checkpoint)
        self._domain = str(cfg.domain or self._data_contract["domains"][0])
        self._devices = self._expand_device_list(str(cfg.device), fallback="cuda")
        self._robot_devices = self._resolve_robot_devices()
        self._dist2robot_mode = normalize_dist2robot_mode(cfg.dist2robot_mode)
        self.last_timing: dict[str, Any] = {}
        self.last_eval_ranges: tuple[dict[str, Any], ...] = ()

        self.max_scene_points = int(
            cfg.max_scene_points
            if cfg.max_scene_points is not None
            else min(int(self._model_contract["max_scene_points"]), 1024)
        )
        self.max_robot_points = int(
            cfg.max_robot_points
            if cfg.max_robot_points is not None
            else min(int(self._model_contract["max_robot_points"]), 256)
        )

        self._replicas = [
            self._build_replica(device=model_device, robot_device=robot_device)
            for model_device, robot_device in zip(self._devices, self._robot_devices)
        ]

    def __call__(
        self,
        *,
        q_traj: np.ndarray,
        u_traj: np.ndarray,
        pointworld_obs: dict[str, Any],
        gripper: Optional[float],
    ) -> np.ndarray:
        del u_traj
        return self.evaluate_cost(q_traj=q_traj, pointworld_obs=pointworld_obs, gripper=gripper)

    def _require_torch(self):
        try:
            import torch
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("Missing dependency: torch") from e
        return torch

    def _load_checkpoint(self, checkpoint_path: str) -> dict[str, Any]:
        ckpt = str(checkpoint_path).strip()
        if not ckpt:
            raise ValueError("PointWorld checkpoint_path is required")
        try:
            return self._torch.load(ckpt, map_location="cpu", weights_only=False)
        except TypeError:
            return self._torch.load(ckpt, map_location="cpu")

    def _expand_device_list(self, raw: str, *, fallback: str) -> list[str]:
        parts = [p.strip() for p in str(raw).split(",") if p.strip()]
        if parts:
            return parts
        return [str(fallback)]

    def _resolve_robot_devices(self) -> list[str]:
        raw = str(self.cfg.robot_sampler_device or "").strip()
        parts = self._expand_device_list(raw, fallback=self._devices[0]) if raw else list(self._devices)
        if len(parts) == 1 and len(self._devices) > 1:
            parts = parts * len(self._devices)
        if len(parts) != len(self._devices):
            raise ValueError(
                f"robot_sampler_device count {len(parts)} must match model devices {len(self._devices)} or be a single device"
            )
        return parts

    def _read_contract(self, checkpoint: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        ensure_sys_path_for_runtime()
        from pointworld.checkpoint_contract import read_checkpoint_contract

        return read_checkpoint_contract(checkpoint, context=f"PointWorld checkpoint '{self.cfg.checkpoint_path}'")

    def _resolve_norm_stats_path(self, raw_path: str) -> str:
        p = Path(str(raw_path))
        if p.is_absolute():
            return str(p)
        return str(Path(default_pointworld_root()) / p)

    def _ensure_dinov3_weights(self) -> None:
        root = Path(default_pointworld_root())
        dinov3_root = root / "third_party" / "dinov3"
        if not dinov3_root.is_dir():
            return

        ckpt_dir = dinov3_root / "checkpoints"
        if ckpt_dir.is_dir() and any(ckpt_dir.glob("dinov3_vitl16_pretrain*.pth")):
            return

        source = Path("/home/models/DINOv3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
        if not source.is_file():
            return

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        target = ckpt_dir / source.name
        if target.exists():
            return
        target.symlink_to(source)

    def _build_replica(self, *, device: str, robot_device: str) -> _PointWorldReplica:
        return _PointWorldReplica(
            device=str(device),
            model=self._build_model(device=str(device)),
            robot=RobotFlowAdapter(
                urdf_path=str(self.cfg.urdf_path),
                max_robot_points=int(self.max_robot_points),
                device=str(robot_device),
                seed=int(self.cfg.seed),
                gripper_only=bool(self.cfg.robot_gripper_only),
            ),
        )

    def _build_args(self, *, device: str) -> SimpleNamespace:
        mc = self._model_contract
        domains = list(self._data_contract["domains"])
        return SimpleNamespace(
            device=str(device),
            distributed=False,
            predictor_dim=int(mc["predictor_dim"]),
            ptv3_size=str(mc["ptv3_size"]),
            ptv3_patch_size=int(mc["ptv3_patch_size"]),
            grid_size=float(mc["grid_size"]),
            depth_threshold=float(mc["depth_threshold"]),
            norm_stats_path=self._resolve_norm_stats_path(str(mc["norm_stats_path"])),
            disable_compile=bool(self.cfg.disable_compile),
            domains=domains,
            robot_features=[
                "robot_flows",
                "robot_colors",
                "robot_normals",
                "gripper_open",
                "robot_velocity",
                "robot_acceleration",
            ],
            scene_features=[
                "scene_flows",
                "scene_colors",
                "scene_normals",
                "gripper_open",
                "dist2robot",
            ],
            seed=int(self.cfg.seed),
            deterministic_train=False,
            deterministic_algorithms=False,
            amp=True,
            dynamics_head_init_scale=(1.0 if any(str(d).startswith("droid") for d in domains) else 0.0),
        )

    def _infer_feature_dims(self, checkpoint: dict[str, Any]) -> dict[str, int]:
        state = checkpoint.get("model")
        if not isinstance(state, dict):
            raise KeyError("PointWorld checkpoint missing 'model' state dict")
        scene_key = "scene_feature_encoder.scene_raw_feat_proj.weight"
        robot_key = "robot_proj.fc1.weight"
        if scene_key not in state or robot_key not in state:
            raise KeyError("PointWorld checkpoint is missing required projection weights")
        return {
            "scene_features_dim": int(state[scene_key].shape[1]),
            "robot_features_dim": int(state[robot_key].shape[1]),
        }

    def _build_model(self, *, device: str):
        ensure_sys_path_for_runtime()
        self._ensure_dinov3_weights()
        args = self._build_args(device=str(device))
        data_info = self._infer_feature_dims(self._checkpoint)
        state = self._checkpoint.get("model")
        if not isinstance(state, dict):
            raise KeyError("PointWorld checkpoint missing 'model' state dict")

        try:
            from pointworld.base import BaseModel

            model = BaseModel(args, data_info, rank=0, cpu_pg=None)
            model.load_state_dict(state, strict=True)
            model.to(self._torch.device(str(device)))
            model.eval()
            return model
        except Exception as exc:
            if "DINOv3" not in str(exc) and "dinov3" not in str(exc):
                raise
            if os.getenv("MPPI_PW_ALLOW_RAW_SCENE_FALLBACK", "0").strip().lower() not in {"1", "true", "yes"}:
                raise
            return self._build_model_without_dinov3(args=args, data_info=data_info, state=state, device=str(device))

    def _build_model_without_dinov3(self, *, args: Any, data_info: dict[str, int], state: dict[str, Any], device: str):
        ensure_sys_path_for_runtime()
        import scene_featurizer
        from pointworld.base import BaseModel

        predictor_dim = int(self._model_contract["predictor_dim"])
        feat_proj_w = state.get("scene_feature_encoder.scene_encoder.feat_proj.weight")
        if feat_proj_w is None:
            raise KeyError("Checkpoint missing scene_feature_encoder.scene_encoder.feat_proj.weight")
        dino_in_dim = int(feat_proj_w.shape[1])

        class _ZeroSceneEncoder2D(nn.Module):
            def __init__(self, args, channels, data_info_dict, rank: int = 0):  # noqa: ARG002
                super().__init__()
                self.args = args
                self.rank = rank
                self.device = args.device
                self.channels = channels
                self.feat_proj = nn.Linear(dino_in_dim, channels)

            def forward(self, scene_coord, scene_exists, camera_data):  # noqa: ARG002
                import torch

                B, Ns, _ = scene_coord.shape
                out = torch.zeros((B, Ns, predictor_dim), device=scene_coord.device, dtype=scene_coord.dtype)
                out[~scene_exists] = 0.0
                return out

            def _extract_camera_data(self, data_dict):
                return {}

        original_cls = scene_featurizer.SceneEncoder2D
        scene_featurizer.SceneEncoder2D = _ZeroSceneEncoder2D
        try:
            model = BaseModel(args, data_info, rank=0, cpu_pg=None)
        finally:
            scene_featurizer.SceneEncoder2D = original_cls

        missing, unexpected = model.load_state_dict(state, strict=False)
        allowed_missing = {
            "scene_feature_encoder.scene_encoder.feat_proj.weight",
            "scene_feature_encoder.scene_encoder.feat_proj.bias",
        }
        extra_missing = [k for k in missing if k not in allowed_missing]
        if extra_missing:
            raise RuntimeError(f"Unexpected missing keys in PointWorld fallback load: {extra_missing}")
        if not all(k.startswith("scene_feature_encoder.scene_encoder.dinov3.") for k in unexpected):
            raise RuntimeError(f"Unexpected non-DINO keys in PointWorld fallback load: {unexpected}")

        model.to(self._torch.device(str(device)))
        model.eval()
        return model

    def _encode_scene_raw_only(self, scene_features_t: Any, *, model: Any, device: str) -> Any:
        torch = self._torch

        B = int(scene_features_t.shape[0])
        device_t = torch.device(str(device))
        idx = [model._domain_to_index[self._domain] for _ in range(B)]
        model._current_domain_indices = torch.as_tensor(idx, device=device_t, dtype=torch.long)

        sfe = model.scene_feature_encoder
        raw = model.normalize_scene_features(scene_features_t)
        zeros = torch.zeros((B, raw.shape[1], model.channels), device=device_t, dtype=raw.dtype)
        fused = torch.cat(
            [
                sfe.scene_encoder_norm(zeros),
                sfe.scene_raw_norm(sfe.scene_raw_feat_proj(raw)),
            ],
            dim=-1,
        )
        return sfe.scene_proj(fused)

    def _prepare_batch(
        self,
        *,
        q_traj: np.ndarray,
        pointworld_obs: dict[str, Any],
        gripper: Optional[float],
        robot: RobotFlowAdapter,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        q = np.asarray(q_traj, dtype=np.float32)
        if q.ndim != 3 or q.shape[-1] != 7:
            raise ValueError(f"q_traj must be (B,T,7), got {q.shape}")
        B, T, _ = q.shape

        scene = prepare_scene_inputs(
            scene_flows=np.asarray(pointworld_obs["scene_flows"], dtype=np.float32),
            scene_colors=np.asarray(pointworld_obs.get("scene_colors")),
            scene_exists=np.asarray(pointworld_obs["scene_exists"], dtype=bool),
            scene_track_confidence=pointworld_obs.get("scene_track_confidence"),
            batch_size=B,
            max_scene_points=int(self.max_scene_points),
        )

        Ns = int(scene["scene_flows"].shape[2])

        def _filter_task(idx_key: str, goal_key: str, w_key: Optional[str], w_default: float) -> None:
            idx = np.asarray(pointworld_obs[idx_key], dtype=np.int64).reshape(-1)
            goal = np.asarray(pointworld_obs[goal_key], dtype=np.float32)

            if goal.ndim >= 1 and int(goal.shape[0]) == int(idx.shape[0]):
                keep = (idx >= 0) & (idx < Ns)
                idx = idx[keep]
                goal = goal[keep]
            else:
                idx = idx[(idx >= 0) & (idx < Ns)]

            scene[idx_key] = idx
            scene[goal_key] = goal
            if w_key is not None:
                scene[w_key] = float(pointworld_obs.get(w_key, w_default))

        if "task_point_indices_obs" in pointworld_obs and "task_goal_positions_obs" in pointworld_obs:
            _filter_task("task_point_indices_obs", "task_goal_positions_obs", "task_weight_obs", 1.0)

        if "task_point_indices_infl" in pointworld_obs and "task_goal_positions_infl" in pointworld_obs:
            _filter_task("task_point_indices_infl", "task_goal_positions_infl", "task_weight_infl", 0.5)

        if "task_point_indices" in pointworld_obs and "task_goal_positions" in pointworld_obs:
            _filter_task("task_point_indices", "task_goal_positions", None, 1.0)

        if int(scene["scene_flows"].shape[1]) != int(T):
            raise ValueError(
                f"PointWorld scene window T={int(scene['scene_flows'].shape[1])} must match q_traj horizon T={int(T)}"
            )

        if gripper is None:
            gripper_arr = np.asarray(pointworld_obs.get("gripper_positions_window"), dtype=np.float32).reshape(1, -1)
            if gripper_arr.shape[1] != T:
                if gripper_arr.shape[1] == 0:
                    gripper_arr = np.zeros((1, T), dtype=np.float32)
                else:
                    gripper_arr = np.repeat(gripper_arr[:, :1], T, axis=1)
            gripper_arr = np.repeat(gripper_arr, B, axis=0)
        else:
            gripper_arr = np.full((B, T), float(gripper), dtype=np.float32)

        robot_flows, robot_colors, robot_normals = robot.build(q_traj=q, gripper_positions=gripper_arr)
        robot = build_robot_inputs(
            robot_flows=robot_flows,
            robot_colors=robot_colors,
            robot_normals=robot_normals,
            gripper_positions=gripper_arr,
            max_robot_points=int(self.max_robot_points),
        )
        batch = {
            "gripper_positions": gripper_arr,
            "robot_flows": robot["robot_flows"],
            "robot_features": robot["robot_features"],
            "robot_exists": robot["robot_exists"],
        }
        return scene, batch

    def _numpy_batch_to_torch(self, batch: dict[str, Any], *, device: str) -> dict[str, Any]:
        torch = self._torch
        device_s = str(device).split(",")[0].strip()
        device_t = torch.device(device_s)
        out: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, np.ndarray):
                if value.dtype == np.bool_:
                    out[key] = torch.as_tensor(value, device=device_t, dtype=torch.bool)
                elif np.issubdtype(value.dtype, np.integer):
                    out[key] = torch.as_tensor(value, device=device_t, dtype=torch.long)
                else:
                    out[key] = torch.as_tensor(value, device=device_t, dtype=torch.float32)
            else:
                out[key] = value
        return out

    def _slice_batch(self, batch: dict[str, Any], start: int, end: int) -> dict[str, Any]:
        out: dict[str, Any] = {}

        B = None
        if "robot_flows" in batch and isinstance(batch["robot_flows"], np.ndarray) and batch["robot_flows"].ndim >= 1:
            B = int(batch["robot_flows"].shape[0])

        for key, value in batch.items():
            if B is not None:
                if isinstance(value, np.ndarray) and value.ndim >= 1 and int(value.shape[0]) == B:
                    out[key] = value[start:end]
                    continue
                if isinstance(value, list) and len(value) == B:
                    out[key] = value[start:end]
                    continue
            out[key] = value
        return out

    def _evaluate_batch_on_replica(
        self,
        *,
        replica: _PointWorldReplica,
        scene_np: dict[str, np.ndarray],
        batch_np: dict[str, np.ndarray],
        valid_pred_steps: Optional[int] = None,
        return_next_scene_p0: bool = False,
    ) -> tuple[np.ndarray, dict[str, Any]] | tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        B = int(batch_np["robot_flows"].shape[0])
        chunk_size = max(1, min(int(self.cfg.eval_batch_size), B))
        costs: list[np.ndarray] = []
        next_p0: list[np.ndarray] = []
        torch = self._torch

        timing: dict[str, Any] = {
            "device": str(replica.device),
            "B": int(B),
            "chunk_size": int(chunk_size),
            "chunks": 0,
            "t_scene_to_torch_ms": 0.0,
            "t_scene_features_ms": 0.0,
            "t_dist2robot_ms": 0.0,
            "t_model_ms": 0.0,
            "t_reduce_ms": 0.0,
            "t_total_ms": 0.0,
        }

        t_rep0 = time.perf_counter()

        device_s = str(replica.device).split(",")[0].strip()
        device_t = torch.device(device_s)
        if device_t.type == "cuda":
            torch.cuda.set_device(device_t.index if device_t.index is not None else 0)

        t_s2t0 = time.perf_counter()
        scene_t = self._numpy_batch_to_torch(scene_np, device=replica.device)
        timing["t_scene_to_torch_ms"] = (time.perf_counter() - t_s2t0) * 1000.0

        for start in range(0, B, chunk_size):
            end = min(B, start + chunk_size)
            batch_chunk = self._slice_batch(batch_np, start, end)
            robot_t = self._numpy_batch_to_torch(batch_chunk, device=replica.device)
            chunk_b = int(end - start)

            scene_flows_t = scene_t["scene_flows"][start:end]
            scene_colors_t = scene_t["scene_colors"][start:end]
            scene_exists_t = scene_t["scene_exists"][start:end]
            t_feat0 = time.perf_counter()
            out = build_scene_features_torch(
                scene_flows=scene_flows_t,
                scene_colors=scene_colors_t,
                gripper_positions=robot_t["gripper_positions"],
                robot_flows=robot_t["robot_flows"],
                robot_exists=robot_t["robot_exists"],
                dist2robot_mode=self._dist2robot_mode,
                return_timing=True,
            )
            if isinstance(out, tuple) and len(out) == 2:
                scene_features_t, feat_timing = out
            else:
                scene_features_t, feat_timing = out, {}
            timing["t_scene_features_ms"] = float(timing["t_scene_features_ms"]) + (time.perf_counter() - t_feat0) * 1000.0
            if isinstance(feat_timing, dict):
                timing["t_dist2robot_ms"] = float(timing["t_dist2robot_ms"]) + float(feat_timing.get("t_dist2robot_ms", 0.0) or 0.0)
                timing["dist2robot_mode"] = str(feat_timing.get("dist2robot_mode", self._dist2robot_mode))

            batch_t = {
                "scene_flows": scene_flows_t,
                "scene_exists": scene_exists_t,
                "scene_features": scene_features_t,
                "robot_flows": robot_t["robot_flows"],
                "robot_features": robot_t["robot_features"],
                "robot_exists": robot_t["robot_exists"],
                "__domain__": [self._domain] * chunk_b,
            }

            t_m0 = time.perf_counter()
            with torch.no_grad():
                encoded_scene = self._encode_scene_raw_only(
                    batch_t["scene_features"][:, 0],
                    model=replica.model,
                    device=replica.device,
                )
                outputs = replica.model(batch_t, training=False, encoded_scene_feat0=encoded_scene)
            timing["t_model_ms"] = float(timing["t_model_ms"]) + (time.perf_counter() - t_m0) * 1000.0

            if bool(return_next_scene_p0) and "scene_relative" in outputs:
                t_last = int(valid_pred_steps) if valid_pred_steps is not None else int(scene_flows_t.shape[1] - 1)
                t_last = max(0, min(int(t_last), int(outputs["scene_relative"].shape[1]) - 1))
                p0 = scene_flows_t[:, 0]
                rel_last = outputs["scene_relative"][:, t_last]
                p_next = (p0 + rel_last).detach().cpu().numpy().astype(np.float32)
                next_p0.append(p_next)

            model_conf = outputs.get("confidence")
            track_conf_t = None
            if "scene_track_confidence" in scene_t:
                track_conf_t = scene_t["scene_track_confidence"][start:end]

            mode = str(self.cfg.cost.mode)
            task_mode = mode in {"task_point_goal_l2", "final_task_point_goal_l2"}

            terms = []
            if "task_point_indices_obs" in scene_t and "task_goal_positions_obs" in scene_t:
                terms.append((scene_t.get("task_point_indices_obs"), scene_t.get("task_goal_positions_obs"), float(scene_t.get("task_weight_obs", 1.0))))
            if "task_point_indices_infl" in scene_t and "task_goal_positions_infl" in scene_t:
                terms.append((scene_t.get("task_point_indices_infl"), scene_t.get("task_goal_positions_infl"), float(scene_t.get("task_weight_infl", 0.5))))
            if "task_point_indices" in scene_t and "task_goal_positions" in scene_t:
                terms.append((scene_t.get("task_point_indices"), scene_t.get("task_goal_positions"), 1.0))

            if terms:
                t_r0 = time.perf_counter()
                total = torch.zeros((chunk_b,), device=replica.device, dtype=torch.float32)
                for idx_term, goal_term, w_term in terms:
                    if float(w_term) == 0.0:
                        continue
                    total = total + float(w_term) * reduce_pointworld_cost_torch(
                        scene_relative=outputs["scene_relative"],
                        scene_exists=batch_t["scene_exists"],
                        model_confidence=model_conf,
                        track_confidence=track_conf_t,
                        scene_p0=scene_flows_t[:, 0],
                        task_point_indices=idx_term,
                        task_goal_positions=goal_term,
                        valid_pred_steps=valid_pred_steps,
                        cfg=self.cfg.cost,
                    )
                timing["t_reduce_ms"] = float(timing["t_reduce_ms"]) + (time.perf_counter() - t_r0) * 1000.0
                costs.append(total.detach().cpu().numpy().astype(np.float32))
            elif task_mode:
                costs.append(torch.zeros((chunk_b,), device=replica.device, dtype=torch.float32).detach().cpu().numpy().astype(np.float32))
            else:
                t_r0 = time.perf_counter()
                costs.append(
                    reduce_pointworld_cost_torch(
                        scene_relative=outputs["scene_relative"],
                        scene_exists=batch_t["scene_exists"],
                        model_confidence=model_conf,
                        track_confidence=track_conf_t,
                        valid_pred_steps=valid_pred_steps,
                        cfg=self.cfg.cost,
                    ).detach().cpu().numpy().astype(np.float32)
                )
                timing["t_reduce_ms"] = float(timing["t_reduce_ms"]) + (time.perf_counter() - t_r0) * 1000.0

            timing["chunks"] = int(timing["chunks"]) + 1

        timing["t_total_ms"] = (time.perf_counter() - t_rep0) * 1000.0
        out_cost = np.concatenate(costs, axis=0).astype(np.float32, copy=False)
        if bool(return_next_scene_p0):
            out_p0 = np.concatenate(next_p0, axis=0).astype(np.float32, copy=False) if next_p0 else np.zeros((int(out_cost.shape[0]), 0, 3), dtype=np.float32)
            return out_cost, out_p0, timing
        return out_cost, timing

    def _evaluate_cost_on_replica(
        self,
        *,
        replica: _PointWorldReplica,
        q_traj: np.ndarray,
        pointworld_obs: dict[str, Any],
        gripper: Optional[float],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        t_p0 = time.perf_counter()
        scene_np, batch_np = self._prepare_batch(
            q_traj=q_traj,
            pointworld_obs=pointworld_obs,
            gripper=gripper,
            robot=replica.robot,
        )
        t_prepare_ms = (time.perf_counter() - t_p0) * 1000.0
        costs, timing = self._evaluate_batch_on_replica(replica=replica, scene_np=scene_np, batch_np=batch_np)
        timing = dict(timing)
        timing["t_prepare_batch_ms"] = float(t_prepare_ms)
        timing["t_total_with_prepare_ms"] = float(t_prepare_ms) + float(timing.get("t_total_ms", 0.0) or 0.0)
        return costs, timing

    def _evaluate_chunk_on_replica(
        self,
        *,
        replica: _PointWorldReplica,
        q_traj: np.ndarray,
        pointworld_obs: dict[str, Any],
        gripper: Optional[float],
        valid_pred_steps: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        t_p0 = time.perf_counter()
        scene_np, batch_np = self._prepare_batch(
            q_traj=q_traj,
            pointworld_obs=pointworld_obs,
            gripper=gripper,
            robot=replica.robot,
        )
        t_prepare_ms = (time.perf_counter() - t_p0) * 1000.0
        result = self._evaluate_batch_on_replica(
            replica=replica,
            scene_np=scene_np,
            batch_np=batch_np,
            valid_pred_steps=int(valid_pred_steps),
            return_next_scene_p0=True,
        )
        if not (isinstance(result, tuple) and len(result) == 3):
            raise RuntimeError("Expected (costs,next_scene_p0,timing) from _evaluate_batch_on_replica")
        costs, next_p0, timing = result
        timing = dict(timing)
        timing["t_prepare_batch_ms"] = float(t_prepare_ms)
        timing["t_total_with_prepare_ms"] = float(t_prepare_ms) + float(timing.get("t_total_ms", 0.0) or 0.0)
        return np.asarray(costs, dtype=np.float32), np.asarray(next_p0, dtype=np.float32), timing

    def _make_ranges(self, total: int, parts: int) -> list[tuple[int, int]]:
        if total <= 0 or parts <= 0:
            return []
        parts = min(int(parts), int(total))
        base = total // parts
        rem = total % parts
        out: list[tuple[int, int]] = []
        start = 0
        for idx in range(parts):
            width = base + (1 if idx < rem else 0)
            end = start + width
            out.append((start, end))
            start = end
        return out

    def evaluate_chunk(
        self,
        *,
        q_traj: np.ndarray,
        pointworld_obs: dict[str, Any],
        gripper: Optional[float],
        valid_pred_steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(q_traj, dtype=np.float32)
        if q.ndim != 3 or q.shape[-1] != 7:
            raise ValueError(f"q_traj must be (B,T,7), got {q.shape}")
        B = int(q.shape[0])
        if B == 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0, 0, 3), dtype=np.float32)

        if len(self._replicas) == 1:
            costs, next_p0, timing = self._evaluate_chunk_on_replica(
                replica=self._replicas[0],
                q_traj=q,
                pointworld_obs=pointworld_obs,
                gripper=gripper,
                valid_pred_steps=int(valid_pred_steps),
            )
            self.last_eval_ranges = (
                {
                    "device": str(self._replicas[0].device),
                    "start": 0,
                    "end": int(B),
                    "samples": int(B),
                },
            )
            self.last_timing = {
                "mode": "single",
                "devices": [str(self._replicas[0].device)],
                "replicas": [dict(timing)],
            }
            return np.asarray(costs, dtype=np.float32), np.asarray(next_p0, dtype=np.float32)

        ranges = self._make_ranges(B, len(self._replicas))
        costs_out = np.zeros((B,), dtype=np.float32)
        next_out = np.zeros((B, 0, 3), dtype=np.float32)

        rep_timings: list[dict[str, Any]] = []

        def _worker(replica: _PointWorldReplica, start: int, end: int) -> tuple[int, int, np.ndarray, np.ndarray, dict[str, Any]]:
            c, p0, t = self._evaluate_chunk_on_replica(
                replica=replica,
                q_traj=q[start:end],
                pointworld_obs=pointworld_obs,
                gripper=gripper,
                valid_pred_steps=int(valid_pred_steps),
            )
            return int(start), int(end), np.asarray(c, dtype=np.float32), np.asarray(p0, dtype=np.float32), dict(t)

        with ThreadPoolExecutor(max_workers=len(ranges)) as pool:
            futures = [
                pool.submit(_worker, self._replicas[idx], start, end)
                for idx, (start, end) in enumerate(ranges)
            ]
            got_shape = None
            for fut in futures:
                start, end, c, p0, t = fut.result()
                costs_out[start:end] = c
                if got_shape is None:
                    got_shape = tuple(p0.shape[1:])
                    next_out = np.zeros((B,) + got_shape, dtype=np.float32)
                next_out[start:end] = p0
                t = dict(t)
                t["start"] = int(start)
                t["end"] = int(end)
                t["samples"] = int(end - start)
                rep_timings.append(t)

        self.last_eval_ranges = tuple(
            {
                "device": str(self._replicas[idx].device),
                "start": int(start),
                "end": int(end),
                "samples": int(end - start),
            }
            for idx, (start, end) in enumerate(ranges)
        )
        self.last_timing = {
            "mode": "multi",
            "devices": [str(r.device) for r in self._replicas],
            "replicas": rep_timings,
        }
        return costs_out, next_out

    def evaluate_cost(
        self,
        *,
        q_traj: np.ndarray,
        pointworld_obs: dict[str, Any],
        gripper: Optional[float],
    ) -> np.ndarray:
        q = np.asarray(q_traj, dtype=np.float32)
        if q.ndim != 3 or q.shape[-1] != 7:
            raise ValueError(f"q_traj must be (B,T,7), got {q.shape}")
        B = int(q.shape[0])
        if B == 0:
            return np.zeros((0,), dtype=np.float32)

        if len(self._replicas) == 1:
            result = self._evaluate_cost_on_replica(
                replica=self._replicas[0],
                q_traj=q,
                pointworld_obs=pointworld_obs,
                gripper=gripper,
            )
            if isinstance(result, tuple) and len(result) == 2:
                out, timing = result
            else:
                out, timing = result, {}
            self.last_eval_ranges = (
                {
                    "device": str(self._replicas[0].device),
                    "start": 0,
                    "end": int(B),
                    "samples": int(B),
                },
            )
            self.last_timing = {
                "mode": "single",
                "devices": [str(self._replicas[0].device)],
                "replicas": [dict(timing)],
            }
            return np.asarray(out, dtype=np.float32)

        ranges = self._make_ranges(B, len(self._replicas))
        costs = np.zeros((B,), dtype=np.float32)

        def _worker(replica: _PointWorldReplica, start: int, end: int) -> tuple[int, int, np.ndarray, dict[str, Any]]:
            result = self._evaluate_cost_on_replica(
                replica=replica,
                q_traj=q[start:end],
                pointworld_obs=pointworld_obs,
                gripper=gripper,
            )
            if isinstance(result, tuple) and len(result) == 2:
                out, timing = result
            else:
                out, timing = result, {}
            return (int(start), int(end), out, dict(timing))

        with ThreadPoolExecutor(max_workers=len(ranges)) as pool:
            futures = [
                pool.submit(_worker, self._replicas[idx], start, end)
                for idx, (start, end) in enumerate(ranges)
            ]
            rep_timings: list[dict[str, Any]] = []
            for fut in futures:
                start, end, arr, timing = fut.result()
                costs[start:end] = arr
                timing = dict(timing)
                timing["start"] = int(start)
                timing["end"] = int(end)
                timing["samples"] = int(end - start)
                rep_timings.append(dict(timing))

        self.last_eval_ranges = tuple(
            {
                "device": str(self._replicas[idx].device),
                "start": int(start),
                "end": int(end),
                "samples": int(end - start),
            }
            for idx, (start, end) in enumerate(ranges)
        )
        self.last_timing = {
            "mode": "multi",
            "devices": [str(r.device) for r in self._replicas],
            "replicas": rep_timings,
        }
        return costs
