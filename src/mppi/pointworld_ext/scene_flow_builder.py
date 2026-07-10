from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from mppi.pointworld_ext.geometry import (
    PinholeIntrinsics,
    apply_robot_sphere_mask_to_points,
    apply_workspace_mask_to_points,
    compute_workspace_mask_2d,
    invert_T,
    lift_tracked_pixels_to_3d,
    project_points_to_pixels,
    transform_points,
)
from mppi.pointworld_ext.input_config import PointWorldInputConfig
from mppi.pointworld_ext.query_manager import QueryPointManager
from mppi.pointworld_ext.tracker_interface import OnlinePointTracker
from mppi.pointworld_ext.window_buffer import PointWorldWindowBuffer


@dataclass(frozen=True)
class SceneFlowBuildOutput:
    scene_flows: np.ndarray
    scene_colors: np.ndarray
    scene_exists: np.ndarray
    scene_visibility: np.ndarray
    scene_depth_valid_mask: np.ndarray
    scene_track_confidence: np.ndarray
    camera_track_slices: Tuple[Tuple[int, int], ...]
    camera_track_ids: np.ndarray
    cameras_used: Tuple[str, ...]


def _sample_rgb_at_uv(rgb: np.ndarray, uv: np.ndarray) -> np.ndarray:
    img = np.asarray(rgb)
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Expected rgb shape (H,W,3), got {img.shape}")

    q = np.asarray(uv, dtype=np.float32)
    if q.ndim != 2 or q.shape[1] != 2:
        raise ValueError(f"Expected uv shape (N,2), got {q.shape}")

    H, W = img.shape[0], img.shape[1]
    u = np.rint(q[:, 0]).astype(np.int32)
    v = np.rint(q[:, 1]).astype(np.int32)
    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)
    cols = img[v, u]
    if cols.dtype != np.uint8:
        cols = np.clip(cols, 0, 255).astype(np.uint8)
    return cols


def _as_spheres_array(spheres: Sequence[Tuple[float, float, float, float]]) -> np.ndarray:
    if not spheres:
        return np.zeros((0, 4), dtype=np.float32)
    s = np.asarray(spheres, dtype=np.float32)
    if s.ndim != 2 or s.shape[1] != 4:
        raise ValueError(f"Expected spheres shape (M,4), got {s.shape}")
    return s


def _stable_sig(x: np.ndarray | Sequence[float] | object) -> bytes:
    a = np.asarray(x, dtype=np.float32)
    return np.ascontiguousarray(np.round(a, 6)).tobytes()


def _require_cv2():
    try:
        import cv2
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing dependency: cv2 (opencv-python) is required for 2D seed masks") from e
    return cv2


def _require_trimesh_urdfpy():
    try:
        import trimesh  # noqa: F401
        import urdfpy  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing dependency: trimesh and urdfpy are required for URDF-based masks") from e


def _get_mesh_stable_id(mesh: object, idx: int | None = None) -> str:
    try:
        src = getattr(mesh, "source", None)
        if src is not None:
            fname = getattr(src, "file_name", None)
            if isinstance(fname, (str, bytes)) and fname:
                base = str(fname).lower()
                return f"{base}_{idx}" if idx is not None else base
    except Exception:
        pass
    try:
        metadata = getattr(mesh, "metadata", None) or {}
        for key in ("name", "file_name"):
            val = metadata.get(key) if isinstance(metadata, dict) else None
            if isinstance(val, (str, bytes)) and val:
                base = str(val).lower()
                return f"{base}_{idx}" if idx is not None else base
    except Exception:
        pass
    try:
        bounds = np.asarray(getattr(mesh, "bounds"), dtype=np.float32).reshape(-1)
        bounds = np.round(bounds, 6)
        vcount = len(getattr(mesh, "vertices", []))
        fcount = len(getattr(mesh, "faces", []))
        base = f"b{','.join(map(str, bounds))}_v{vcount}_f{fcount}"
        return f"{base}_{idx}" if idx is not None else base
    except Exception:
        base = f"unknown_{id(mesh)}"
        return f"{base}_{idx}" if idx is not None else base


class _URDFHelper:
    def __init__(self, *, urdf_path: str) -> None:
        _require_trimesh_urdfpy()
        import urdfpy

        self._urdf = urdfpy.URDF.load(str(urdf_path))
        self._link_cache: Dict[str, object] = {}

    def _cfg_from_state(self, joint_positions: np.ndarray, gripper_positions: np.ndarray) -> Dict[str, float]:
        jp = np.asarray(joint_positions, dtype=np.float32).reshape(-1)
        if jp.shape[0] < 7:
            raise ValueError("joint_positions must have at least 7 elements")
        gp = np.asarray(gripper_positions, dtype=np.float32).reshape(-1)
        g0 = float(gp[0]) if gp.size > 0 else 0.0

        cfg: Dict[str, float] = {"finger_joint": float(g0)}
        for ji in range(7):
            cfg[f"panda_joint{ji + 1}"] = float(jp[ji])
        return cfg

    def _resolve_link(self, link_name: str) -> object:
        name = str(link_name)
        if name in self._link_cache:
            return self._link_cache[name]

        link = None
        if hasattr(self._urdf, "link_map") and isinstance(self._urdf.link_map, dict):
            link = self._urdf.link_map.get(name)
        if link is None and hasattr(self._urdf, "links"):
            for lk in self._urdf.links:
                if getattr(lk, "name", None) == name:
                    link = lk
                    break
        if link is None:
            raise ValueError(f"URDF link not found: {name}")
        self._link_cache[name] = link
        return link

    def visual_trimesh_fk(self, *, joint_positions: np.ndarray, gripper_positions: np.ndarray):
        cfg = self._cfg_from_state(joint_positions, gripper_positions)
        return self._urdf.visual_trimesh_fk(cfg=cfg)

    def transform_spheres_from_link(
        self,
        *,
        joint_positions: np.ndarray,
        gripper_positions: np.ndarray,
        link_name: str,
        spheres_link: np.ndarray,
    ) -> np.ndarray:
        s = np.asarray(spheres_link, dtype=np.float32)
        if s.size == 0:
            return np.zeros((0, 4), dtype=np.float32)
        if s.ndim != 2 or s.shape[1] != 4:
            raise ValueError(f"Expected spheres_link shape (M,4), got {s.shape}")

        import numpy as _np

        cfg = self._cfg_from_state(joint_positions, gripper_positions)
        link = self._resolve_link(link_name)
        fk = self._urdf.link_fk(cfg)
        T = _np.asarray(fk[link], dtype=_np.float32).reshape(4, 4)
        R = T[:3, :3]
        t = T[:3, 3]

        centers_l = s[:, :3]
        centers_w = (centers_l @ R.T) + t[None, :]
        out = _np.empty_like(s)
        out[:, :3] = centers_w.astype(_np.float32)
        out[:, 3] = s[:, 3]
        return out.astype(_np.float32)


class _RobotMask2DBuilder:
    def __init__(self, *, urdf_helper: _URDFHelper) -> None:
        self._urdf_helper = urdf_helper
        self._ready = False
        self._mesh_presampled_points: Dict[str, np.ndarray] = {}
        self._last_seed: Optional[int] = None

    def build_world_points(
        self,
        *,
        joint_positions: np.ndarray,
        gripper_positions: np.ndarray,
        seed: Optional[int],
    ) -> Dict[str, np.ndarray]:
        fk_result = self._urdf_helper.visual_trimesh_fk(joint_positions=joint_positions, gripper_positions=gripper_positions)
        if (not self._ready) or (self._last_seed != seed):
            self._presample_mesh_points(fk_result, seed=seed)

        gripper_keywords = ["finger", "knuckle", "robotiq"]
        out: Dict[str, np.ndarray] = {"standard": [], "gripper": []}
        for i, mesh in enumerate(fk_result.keys()):
            mesh_name = _get_mesh_stable_id(mesh, i)
            if float(getattr(mesh, "area", 0.0)) <= 0.0:
                continue
            pts_local = self._mesh_presampled_points.get(mesh_name)
            if pts_local is None or pts_local.size == 0:
                continue
            T_wm = np.asarray(fk_result[mesh], dtype=np.float32)
            pts_world = transform_points(T_wm, pts_local)
            key = "gripper" if any(k in mesh_name.lower() for k in gripper_keywords) else "standard"
            out[key].append(np.asarray(pts_world, dtype=np.float32))

        return {
            "standard": np.concatenate(out["standard"], axis=0).astype(np.float32) if out["standard"] else np.zeros((0, 3), dtype=np.float32),
            "gripper": np.concatenate(out["gripper"], axis=0).astype(np.float32) if out["gripper"] else np.zeros((0, 3), dtype=np.float32),
        }

    def project_world_points_to_mask(
        self,
        *,
        world_points: Dict[str, np.ndarray],
        intr: PinholeIntrinsics,
        world2cam: np.ndarray,
        height: int,
        width: int,
    ) -> np.ndarray:
        cv2 = _require_cv2()

    def _presample_mesh_points(self, fk_result: Dict[object, np.ndarray], *, seed: Optional[int]) -> None:
        _require_trimesh_urdfpy()
        import trimesh

        self._mesh_presampled_points = {}
        mesh_names, mesh_objs, mesh_areas = [], [], []
        for i, mesh in enumerate(fk_result.keys()):
            name = _get_mesh_stable_id(mesh, i)
            if float(getattr(mesh, "area", 0.0)) <= 0.0:
                continue
            eff_area = float(getattr(mesh, "area"))
            if "hand_camera_part" in name.lower():
                eff_area *= 1e-6
            mesh_names.append(name)
            mesh_objs.append(mesh)
            mesh_areas.append(eff_area)

        if not mesh_names:
            self._ready = True
            self._last_seed = seed
            return

        total_area = float(np.sum(mesh_areas))
        total_samples = 100000
        gripper_multiplier = 2.0
        min_per_mesh = 500

        rng_state = None
        if seed is not None:
            rng_state = np.random.get_state()
            np.random.seed(int(seed) % (2**32 - 1))

        try:
            for name, mesh, area in zip(mesh_names, mesh_objs, mesh_areas):
                frac = 0.0 if total_area <= 0 else float(area / total_area)
                n = int(total_samples * frac)
                if any(k in name.lower() for k in ["finger", "knuckle", "robotiq"]):
                    n = int(n * gripper_multiplier)
                n = max(min_per_mesh, n)
                try:
                    pts = mesh.sample(int(n))
                except Exception:
                    pts, _ = trimesh.sample.sample_surface_even(mesh, int(n))
                self._mesh_presampled_points[name] = np.asarray(pts, dtype=np.float32)
        finally:
            if rng_state is not None:
                np.random.set_state(rng_state)

        self._ready = True
        self._last_seed = seed

    def build_mask(
        self,
        *,
        intr: PinholeIntrinsics,
        world2cam: np.ndarray,
        height: int,
        width: int,
        joint_positions: np.ndarray,
        gripper_positions: np.ndarray,
        seed: Optional[int],
    ) -> np.ndarray:
        world_points = self.build_world_points(
            joint_positions=joint_positions,
            gripper_positions=gripper_positions,
            seed=seed,
        )
        return self.project_world_points_to_mask(
            world_points=world_points,
            intr=intr,
            world2cam=world2cam,
            height=height,
            width=width,
        )

    def project_world_points_to_mask(
        self,
        *,
        world_points: Dict[str, np.ndarray],
        intr: PinholeIntrinsics,
        world2cam: np.ndarray,
        height: int,
        width: int,
        return_timing: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, float]]:
        cv2 = _require_cv2(); H, W = int(height), int(width)
        std = np.zeros((H, W), dtype=np.uint8); grip = np.zeros((H, W), dtype=np.uint8)
        scale = max(1e-6, min(float(H) / 180.0, float(W) / 320.0))
        rs, rg = max(1, int(round(30.0 * scale))), max(1, int(round(20.0 * scale)))
        ks = max(3, int(round(30.0 * scale))); ks += (ks % 2 == 0)
        kstd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rs + 1, 2 * rs + 1))
        kgrip = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rg + 1, 2 * rg + 1))
        kclose = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        ms_proj = ms_rast = 0.0
        for key, target in (("standard", std), ("gripper", grip)):
            pts = np.asarray(world_points.get(key, np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
            if pts.size == 0: continue
            t0 = time.perf_counter(); uv, okz = project_points_to_pixels(pts, world2cam=world2cam, intr=intr); ms_proj += (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter(); uv = np.round(uv[okz]).astype(np.int32); keep = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H); uv = uv[keep]
            if uv.size: target[uv[:, 1], uv[:, 0]] = 1
            ms_rast += (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter(); mask = np.zeros((H, W), dtype=np.uint8)
        if np.any(std): mask |= cv2.dilate(std, kstd, iterations=1)
        if np.any(grip): mask |= cv2.dilate(grip, kgrip, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kclose); ms_morph = (time.perf_counter() - t0) * 1000.0
        mask = mask.astype(bool)
        return (mask, {"ms_project": float(ms_proj), "ms_rasterize": float(ms_rast), "ms_morph": float(ms_morph)}) if return_timing else mask


class OnlineSceneFlowBuilder:
    def __init__(
        self,
        *,
        cfg: PointWorldInputConfig,
        window_buffer: PointWorldWindowBuffer,
        tracker: OnlinePointTracker,
        query_manager: QueryPointManager,
    ) -> None:
        self.cfg = cfg
        self.window_buffer = window_buffer
        self.tracker = tracker
        self.query_manager = query_manager
        self._urdf_helper = None
        self._robot_mask_builder = None
        self.last_timing: dict[str, object] = {}
        self._shift_mask_build_counter = 0
        self._shift_mask_cache: dict[tuple[str, int, int, int], np.ndarray] = {}
        self._shift_mask_update_every = max(1, int(os.environ.get("MPPI_PW_SHIFT_MASK_UPDATE_EVERY", "1")))

    def _ensure_urdf_helper(self) -> None:
        if self._urdf_helper is not None:
            return
        if not self.cfg.urdf_path:
            raise ValueError("cfg.urdf_path is required for URDF-based robot mask / ee filter")
        self._urdf_helper = _URDFHelper(urdf_path=str(self.cfg.urdf_path))

    def _ensure_robot_mask_builder(self) -> None:
        if self._robot_mask_builder is not None:
            return
        self._ensure_urdf_helper()
        self._robot_mask_builder = _RobotMask2DBuilder(urdf_helper=self._urdf_helper)

    def build(
        self,
        *,
        window_shift: int = 1,
        robot_spheres_base: Optional[Sequence[np.ndarray]] = None,
    ) -> SceneFlowBuildOutput:
        t_build0 = time.perf_counter()
        steps = self.window_buffer.get_window()
        T = len(steps)
        if int(T) != 11:
            raise RuntimeError("PointWorld requires window length T == 11")

        available = self.window_buffer.get_available_cameras()
        cameras_used = self.cfg.select_cameras(available)
        if not cameras_used:
            raise RuntimeError("No cameras selected")

        if robot_spheres_base is not None and len(robot_spheres_base) != T:
            raise ValueError("robot_spheres_base must be length T when provided")

        per_cam_xyz: Dict[str, np.ndarray] = {}
        per_cam_exists: Dict[str, np.ndarray] = {}
        per_cam_visibility: Dict[str, np.ndarray] = {}
        per_cam_depth_valid: Dict[str, np.ndarray] = {}
        per_cam_conf: Dict[str, np.ndarray] = {}
        per_cam_cols: Dict[str, np.ndarray] = {}

        ee_spheres_link = _as_spheres_array(self.cfg.robot_filter.ee_filter_spheres)

        seed_robot_mask_enabled = bool(getattr(self.cfg, "seed_robot_mask_enabled", False))
        if bool(getattr(self.cfg.robot_filter, "ee_filter_enabled", False)):
            self._ensure_urdf_helper()
        if seed_robot_mask_enabled:
            self._ensure_robot_mask_builder()

        cam_timing: dict[str, object] = {}
        robot_mask_seed = (int(self.cfg.robot_mask_seed) if getattr(self.cfg, "robot_mask_seed", None) is not None else None)
        shift = max(0, min(int(window_shift), T - 1))
        refresh_shift_mask = (self._shift_mask_update_every <= 1) or (self._shift_mask_build_counter % self._shift_mask_update_every == 0)
        shift_epoch = self._shift_mask_build_counter // self._shift_mask_update_every
        shared_world_points0 = None
        shared_world_points_shift = None
        if seed_robot_mask_enabled:
            shared_world_points0 = self._robot_mask_builder.build_world_points(
                joint_positions=np.asarray(steps[0].joint_positions, dtype=np.float32),
                gripper_positions=np.asarray(steps[0].gripper_positions, dtype=np.float32),
                seed=robot_mask_seed,
            )
            if refresh_shift_mask:
                shared_world_points_shift = self._robot_mask_builder.build_world_points(
                    joint_positions=np.asarray(steps[shift].joint_positions, dtype=np.float32),
                    gripper_positions=np.asarray(steps[shift].gripper_positions, dtype=np.float32),
                    seed=robot_mask_seed,
                )

        for cam_name in cameras_used:
            t_cam0 = time.perf_counter()

            t_frames0 = time.perf_counter()
            frames = [np.asarray(s.cameras[cam_name].rgb) for s in steps]
            ms_frames = (time.perf_counter() - t_frames0) * 1000.0

            cam0 = steps[0].cameras[cam_name]
            rgb0 = cam0.rgb
            depth0 = cam0.depth

            H0, W0 = int(depth0.shape[0]), int(depth0.shape[1])
            world2cam0 = invert_T(cam0.extrinsics)

            t_valid0 = time.perf_counter()
            valid_mask0 = np.isfinite(depth0) & (np.asarray(depth0) > 0)
            ms_valid0 = (time.perf_counter() - t_valid0) * 1000.0

            t_ws0 = time.perf_counter()
            workspace_mask0 = compute_workspace_mask_2d(
                height=H0,
                width=W0,
                intr=cam0.intrinsics,
                world2cam=world2cam0,
                workspace_min=self.cfg.workspace_filter.workspace_min,
                workspace_max=self.cfg.workspace_filter.workspace_max,
            )
            ms_ws0 = (time.perf_counter() - t_ws0) * 1000.0

            t_rm0 = time.perf_counter()
            if seed_robot_mask_enabled:
                robot_mask0, rm0_timing = self._robot_mask_builder.project_world_points_to_mask(
                    world_points=shared_world_points0, intr=cam0.intrinsics, world2cam=world2cam0, height=H0, width=W0, return_timing=True
                )
            else:
                robot_mask0 = np.zeros((H0, W0), dtype=bool); rm0_timing = {"ms_project": 0.0, "ms_rasterize": 0.0, "ms_morph": 0.0}
            ms_rm0 = (time.perf_counter() - t_rm0) * 1000.0

            t_q0 = time.perf_counter()
            try:
                q0 = self.query_manager.get_or_create(
                    cam_name,
                    rgb=rgb0,
                    depth=depth0,
                    valid_mask=valid_mask0,
                    workspace_mask=workspace_mask0,
                    robot_mask=robot_mask0,
                )
            except TypeError:
                q0 = self.query_manager.get_or_create(cam_name, rgb=rgb0, depth=depth0)
            q_ms = (time.perf_counter() - t_q0) * 1000.0

            t_cols0 = time.perf_counter()
            cols0 = _sample_rgb_at_uv(rgb0, q0)
            ms_cols0 = (time.perf_counter() - t_cols0) * 1000.0

            t_tr0 = time.perf_counter()
            track_out = self.tracker.track_window(frames, q0)
            tr_ms = (time.perf_counter() - t_tr0) * 1000.0

            uv_tracks = np.asarray(track_out.uv_tracks, dtype=np.float32)
            visibility = np.asarray(track_out.visibility).astype(bool)
            confidence = np.asarray(track_out.confidence, dtype=np.float32)

            if uv_tracks.shape != (T, q0.shape[0], 2):
                raise ValueError(f"uv_tracks shape {uv_tracks.shape} must be (T,N,2)={(T, q0.shape[0], 2)}")
            if visibility.shape != (T, q0.shape[0]):
                raise ValueError(f"visibility shape {visibility.shape} must be (T,N)={(T, q0.shape[0])}")
            if confidence.shape != (T, q0.shape[0]):
                raise ValueError(f"confidence shape {confidence.shape} must be (T,N)={(T, q0.shape[0])}")

            xyz_base = np.zeros((T, q0.shape[0], 3), dtype=np.float32)
            exists = np.zeros((T, q0.shape[0]), dtype=bool)
            vis_out = np.zeros((T, q0.shape[0]), dtype=bool)
            depth_valid_out = np.zeros((T, q0.shape[0]), dtype=bool)
            conf_out = np.zeros((T, q0.shape[0]), dtype=np.float32)
            ws_ok = np.zeros((T, q0.shape[0]), dtype=bool)

            t_loop0 = time.perf_counter()
            ms_lift3d = 0.0
            ms_tf = 0.0
            ms_ws_filter = 0.0
            ms_robot_filter = 0.0
            ms_ee_filter = 0.0
            ms_ok = 0.0
            ms_pack = 0.0

            ee_enabled = bool(self.cfg.robot_filter.ee_filter_enabled)
            min_conf = float(self.cfg.tracking.min_track_confidence)

            for t in range(T):
                cam = steps[t].cameras[cam_name]
                intr: PinholeIntrinsics = cam.intrinsics
                depth_t = cam.depth

                t_l0 = time.perf_counter()
                xyz_cam, z_ok = lift_tracked_pixels_to_3d(
                    depth_t,
                    uv_tracks[t],
                    intr=intr,
                    depth_min_m=float(self.cfg.tracking.depth_min_m),
                    depth_max_m=float(self.cfg.tracking.depth_max_m),
                )
                ms_lift3d += (time.perf_counter() - t_l0) * 1000.0

                t_tf0 = time.perf_counter()
                xyzb = transform_points(cam.extrinsics, xyz_cam.reshape(-1, 3)).reshape(q0.shape[0], 3)
                ms_tf += (time.perf_counter() - t_tf0) * 1000.0

                t_ws1 = time.perf_counter()
                keep_ws = apply_workspace_mask_to_points(
                    xyzb,
                    workspace_min=self.cfg.workspace_filter.workspace_min,
                    workspace_max=self.cfg.workspace_filter.workspace_max,
                )
                ms_ws_filter += (time.perf_counter() - t_ws1) * 1000.0
                ws_ok[t] = keep_ws & z_ok

                if robot_spheres_base is None:
                    keep_robot = np.ones((q0.shape[0],), dtype=bool)
                else:
                    t_rb0 = time.perf_counter()
                    keep_robot = apply_robot_sphere_mask_to_points(
                        xyzb,
                        spheres=np.asarray(robot_spheres_base[t], dtype=np.float32),
                        margin=float(self.cfg.robot_filter.robot_mask_margin_m),
                    )
                    ms_robot_filter += (time.perf_counter() - t_rb0) * 1000.0

                if ee_enabled:
                    t_ee0 = time.perf_counter()
                    link_name = str(getattr(self.cfg.robot_filter, "ee_filter_link", ""))
                    if not link_name:
                        raise ValueError("ee_filter_enabled is True but cfg.robot_filter.ee_filter_link is empty")

                    spheres_w = self._urdf_helper.transform_spheres_from_link(
                        joint_positions=np.asarray(steps[t].joint_positions, dtype=np.float32),
                        gripper_positions=np.asarray(steps[t].gripper_positions, dtype=np.float32),
                        link_name=link_name,
                        spheres_link=ee_spheres_link,
                    )
                    keep_ee = apply_robot_sphere_mask_to_points(
                        xyzb,
                        spheres=np.asarray(spheres_w, dtype=np.float32),
                        margin=float(self.cfg.robot_filter.ee_filter_margin_m),
                    )
                    ms_ee_filter += (time.perf_counter() - t_ee0) * 1000.0
                else:
                    keep_ee = np.ones((q0.shape[0],), dtype=bool)

                t_ok0 = time.perf_counter()
                ok = visibility[t] & z_ok & keep_ws & keep_robot & keep_ee & (confidence[t] >= float(min_conf))
                ms_ok += (time.perf_counter() - t_ok0) * 1000.0

                t_pk0 = time.perf_counter()
                vis_out[t] = visibility[t]
                depth_valid_out[t] = z_ok
                xyz_base[t] = np.where(ok[:, None], xyzb, np.zeros_like(xyzb))
                exists[t] = ok
                conf_out[t] = np.where(ok, confidence[t], 0.0).astype(np.float32)
                ms_pack += (time.perf_counter() - t_pk0) * 1000.0

            ms_loop_total = (time.perf_counter() - t_loop0) * 1000.0

            t_stab0 = time.perf_counter()
            r_ws = ws_ok.astype(np.float32).mean(axis=0) if ws_ok.size else np.zeros((q0.shape[0],), dtype=np.float32)
            cur = np.zeros((q0.shape[0],), dtype=np.int32)
            mx = np.zeros((q0.shape[0],), dtype=np.int32)
            for t in range(T):
                cur = (cur + 1) * ws_ok[t].astype(np.int32)
                mx = np.maximum(mx, cur)

            thr = float(self.cfg.workspace_filter.stability_ws_ratio_thresh)
            run_thr = int(self.cfg.workspace_filter.stability_ws_run_len_thresh)
            stable = (r_ws >= thr) & (mx >= run_thr)
            ms_stability = (time.perf_counter() - t_stab0) * 1000.0

            strict_enabled = bool(getattr(self.cfg.workspace_filter, "strict_all_time_enabled", False))
            strict_all = np.all(ws_ok, axis=0) if ws_ok.size else np.zeros((q0.shape[0],), dtype=bool)

            if strict_enabled:
                exists2 = exists & strict_all[None, :]
                xyz_base = np.where(exists2[..., None], xyz_base, np.zeros_like(xyz_base))
                conf_out = np.where(exists2, conf_out, 0.0).astype(np.float32)
                exists = exists2

            if bool(self.cfg.workspace_filter.stability_apply_to_confidence):
                conf_out = (conf_out * r_ws[None, :]).astype(np.float32)

            if bool(self.cfg.workspace_filter.stability_apply_to_exists):
                exists2 = exists & stable[None, :]
                xyz_base = np.where(exists2[..., None], xyz_base, np.zeros_like(xyz_base))
                conf_out = np.where(exists2, conf_out, 0.0).astype(np.float32)
                exists = exists2

            stable_mask0 = (stable & strict_all) if strict_enabled else stable

            per_cam_xyz[cam_name] = xyz_base
            per_cam_exists[cam_name] = exists
            per_cam_visibility[cam_name] = vis_out
            per_cam_depth_valid[cam_name] = depth_valid_out
            per_cam_conf[cam_name] = conf_out
            per_cam_cols[cam_name] = np.repeat(cols0[None, :, :], T, axis=0)

            shift = shift

            rgb_shift = steps[shift].cameras[cam_name].rgb
            depth_shift = steps[shift].cameras[cam_name].depth

            cam_s = steps[shift].cameras[cam_name]
            rgb_shift = cam_s.rgb
            depth_shift = cam_s.depth

            Hs, Ws = int(depth_shift.shape[0]), int(depth_shift.shape[1])
            world2cams = invert_T(cam_s.extrinsics)

            t_svalid0 = time.perf_counter()
            valid_masks = np.isfinite(depth_shift) & (np.asarray(depth_shift) > 0)
            ms_shift_valid = (time.perf_counter() - t_svalid0) * 1000.0

            t_sws0 = time.perf_counter()
            workspace_masks = compute_workspace_mask_2d(
                height=Hs,
                width=Ws,
                intr=cam_s.intrinsics,
                world2cam=world2cams,
                workspace_min=self.cfg.workspace_filter.workspace_min,
                workspace_max=self.cfg.workspace_filter.workspace_max,
            )
            ms_ws_shift = (time.perf_counter() - t_sws0) * 1000.0

            t_srm0 = time.perf_counter()
            shift_cache_hit = 0; shift_cache_miss = 0
            if seed_robot_mask_enabled:
                geom_key = (str(cam_name), int(Hs), int(Ws), int(shift), int(shift_epoch), int(robot_mask_seed or -1), _stable_sig(world2cams), _stable_sig((cam_s.intrinsics.fx, cam_s.intrinsics.fy, cam_s.intrinsics.cx, cam_s.intrinsics.cy)))
                if refresh_shift_mask or (geom_key not in self._shift_mask_cache):
                    if shared_world_points_shift is None:
                        shared_world_points_shift = self._robot_mask_builder.build_world_points(joint_positions=np.asarray(steps[shift].joint_positions, dtype=np.float32), gripper_positions=np.asarray(steps[shift].gripper_positions, dtype=np.float32), seed=robot_mask_seed)
                    robot_masks, rms_timing = self._robot_mask_builder.project_world_points_to_mask(world_points=shared_world_points_shift, intr=cam_s.intrinsics, world2cam=world2cams, height=Hs, width=Ws, return_timing=True)
                    self._shift_mask_cache[geom_key] = np.asarray(robot_masks, dtype=bool); shift_cache_miss = 1
                else:
                    robot_masks = self._shift_mask_cache[geom_key]; rms_timing = {"ms_project": 0.0, "ms_rasterize": 0.0, "ms_morph": 0.0}; shift_cache_hit = 1
            else:
                robot_masks = np.zeros((Hs, Ws), dtype=bool); rms_timing = {"ms_project": 0.0, "ms_rasterize": 0.0, "ms_morph": 0.0}
            ms_rm_shift = (time.perf_counter() - t_srm0) * 1000.0

            lift_ms = float(ms_loop_total)

            t_adv0 = time.perf_counter()
            try:
                self.query_manager.advance_window(
                    cam_name,
                    uv_tracks=uv_tracks,
                    visibility=visibility,
                    confidence=confidence,
                    stable_mask0=stable_mask0,
                    new_query_index=shift,
                    rgb0=rgb_shift,
                    depth0=depth_shift,
                    valid_mask0=valid_masks,
                    workspace_mask0=workspace_masks,
                    robot_mask0=robot_masks,
                )
            except TypeError:
                self.query_manager.advance_window(
                    cam_name,
                    uv_tracks=uv_tracks,
                    visibility=visibility,
                    confidence=confidence,
                    new_query_index=shift,
                    rgb0=rgb_shift,
                    depth0=depth_shift,
                    valid_mask0=valid_masks,
                    workspace_mask0=workspace_masks,
                    robot_mask0=robot_masks,
                )
            adv_ms = (time.perf_counter() - t_adv0) * 1000.0

            cam_timing[str(cam_name)] = {
                "ms_total": (time.perf_counter() - t_cam0) * 1000.0,
                "ms_frames": float(ms_frames),
                "ms_valid0": float(ms_valid0),
                "ms_workspace0": float(ms_ws0),
                "ms_robot_mask0": float(ms_rm0),
                "ms_project_mask0": float(rm0_timing["ms_project"]),
                "ms_rasterize_mask0": float(rm0_timing["ms_rasterize"]),
                "ms_morph_mask0": float(rm0_timing["ms_morph"]),
                "ms_query": float(q_ms),
                "ms_sample_rgb0": float(ms_cols0),
                "ms_track": float(tr_ms),
                "ms_lift": float(lift_ms),
                "ms_lift3d": float(ms_lift3d),
                "ms_transform": float(ms_tf),
                "ms_ws_filter": float(ms_ws_filter),
                "ms_robot_filter": float(ms_robot_filter),
                "ms_ee_filter": float(ms_ee_filter),
                "ms_ok": float(ms_ok),
                "ms_pack": float(ms_pack),
                "ms_stability": float(ms_stability),
                "ms_shift_valid": float(ms_shift_valid),
                "ms_shift_workspace": float(ms_ws_shift),
                "ms_shift_robot_mask": float(ms_rm_shift),
                "ms_project_mask_shift": float(rms_timing["ms_project"]),
                "ms_rasterize_mask_shift": float(rms_timing["ms_rasterize"]),
                "ms_morph_mask_shift": float(rms_timing["ms_morph"]),
                "shift_mask_cache_hit": int(shift_cache_hit),
                "shift_mask_cache_miss": int(shift_cache_miss),
                "ms_advance": float(adv_ms),
                "n_query": int(q0.shape[0]),
                "hw": [int(H0), int(W0)],
            }

        xyz_list = [per_cam_xyz[n] for n in cameras_used]
        cols_list = [per_cam_cols[n] for n in cameras_used]
        exists_list = [per_cam_exists[n] for n in cameras_used]
        vis_list = [per_cam_visibility[n] for n in cameras_used]
        depth_valid_list = [per_cam_depth_valid[n] for n in cameras_used]
        conf_list = [per_cam_conf[n] for n in cameras_used]

        scene_flows = np.concatenate(xyz_list, axis=1).astype(np.float32)
        scene_colors = np.concatenate(cols_list, axis=1).astype(np.uint8)
        scene_exists = np.concatenate(exists_list, axis=1).astype(bool)
        scene_visibility = np.concatenate(vis_list, axis=1).astype(bool)
        scene_depth_valid_mask = np.concatenate(depth_valid_list, axis=1).astype(bool)
        scene_track_confidence = np.concatenate(conf_list, axis=1).astype(np.float32)

        camera_track_slices = []
        camera_track_ids = np.empty((scene_flows.shape[1],), dtype=np.int32)
        start = 0
        for i, name in enumerate(cameras_used):
            n = per_cam_xyz[name].shape[1]
            end = start + n
            camera_track_slices.append((start, end))
            camera_track_ids[start:end] = int(i)
            start = end

        self.last_timing = {
            "ms_total": (time.perf_counter() - t_build0) * 1000.0,
            "cams": cam_timing,
        }
        self._shift_mask_build_counter += 1

        return SceneFlowBuildOutput(
            scene_flows=scene_flows,
            scene_colors=scene_colors,
            scene_exists=scene_exists,
            scene_visibility=scene_visibility,
            scene_depth_valid_mask=scene_depth_valid_mask,
            scene_track_confidence=scene_track_confidence,
            camera_track_slices=tuple(camera_track_slices),
            camera_track_ids=camera_track_ids,
            cameras_used=tuple(cameras_used),
        )