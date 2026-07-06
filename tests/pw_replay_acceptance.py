from __future__ import annotations

import argparse
import asyncio
import io
import json
import pickle
import re
import time
import uuid
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mppi.curobo_ext.check_depth import load_depth_any, load_rgb_any
from mppi.protocol.msgpack_codec import decode_message, encode_message
from mppi.protocol.types_pcl import InferRequestPCL, ObsPCL, SCHEMA_VERSION_PCL


def _require_websockets():
    try:
        import websockets  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing dependency: websockets") from e
    return websockets


def _encode_rgb_jpeg(rgb: np.ndarray, *, quality: int = 90) -> bytes:
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected rgb shape (H,W,3), got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    try:
        import cv2  # type: ignore

        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return bytes(buf.tobytes())
    except Exception:
        pass

    try:
        from PIL import Image  # type: ignore

        img = Image.fromarray(arr, mode="RGB")
        bio = io.BytesIO()
        img.save(bio, format="JPEG", quality=int(quality), optimize=True)
        return bio.getvalue()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Failed to encode RGB as JPEG: {e}") from e


def _encode_depth_npy_zlib(depth_m: np.ndarray, *, level: int = 3) -> bytes:
    d = np.asarray(depth_m)
    if d.ndim == 3 and d.shape[-1] == 1:
        d = d[..., 0]
    if d.ndim != 2:
        raise ValueError(f"Expected depth shape (H,W), got {d.shape}")
    d = np.asarray(d, dtype=np.float32)

    bio = io.BytesIO()
    np.save(bio, d, allow_pickle=False)
    return zlib.compress(bio.getvalue(), level=int(level))


def _normalize_rel_path(p: str) -> str:
    s = str(p).replace("\\", "/").strip()
    s = re.sub(r"^(\./)+", "", s)
    return s


def _resolve_asset_path(*, json_path: str, data_root: str, rel_or_abs: str) -> str:
    p0 = _normalize_rel_path(rel_or_abs)
    if p0.startswith("/"):
        return p0

    p1 = re.sub(r"^ep_\d+/", "", p0)
    base_dir = str(Path(json_path).resolve().parent)
    dr = str(data_root).strip() or base_dir

    candidates = [
        Path(dr) / p0,
        Path(dr) / p1,
        Path(base_dir) / p0,
        Path(base_dir) / p1,
        Path(base_dir).parent / p0,
        Path(base_dir).parent / p1,
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return str(candidates[0])


def _extract_q(obj: Dict[str, Any]) -> List[float]:
    if isinstance(obj, dict) and "/franka/joint_states" in obj:
        js = obj["/franka/joint_states"]
        if isinstance(js, dict) and isinstance(js.get("position"), list) and len(js["position"]) == 7:
            return [float(x) for x in js["position"]]
    for key in ("joint_positions", "q", "qpos"):
        if isinstance(obj, dict) and isinstance(obj.get(key), list) and len(obj[key]) == 7:
            return [float(x) for x in obj[key]]
    raise ValueError("Missing joint positions with len=7")


def _extract_step_id(obj: Dict[str, Any], fallback: int) -> int:
    try:
        return int(obj.get("frame_id", fallback))
    except Exception:
        return int(fallback)


def _load_json_items(json_path: str) -> List[Dict[str, Any]]:
    obj = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError("data.json must be a list")
    return [dict(x) for x in obj]


def _load_episode_items(episode_dir: str) -> List[Dict[str, Any]]:
    ep = Path(episode_dir)
    steps = pickle.load(open(ep / "data.pkl", "rb"))
    if not isinstance(steps, list) or not steps:
        raise ValueError("data.pkl must be a non-empty list")
    return [dict(x) if isinstance(x, dict) else {"frame_id": i, "q": list(x)} for i, x in enumerate(steps)]


def _find_rgb_with_stem(rgb_dir: Path, stem: str) -> str:
    for ext in (".jpg", ".png", ".jpeg"):
        p = rgb_dir / f"{stem}{ext}"
        if p.is_file():
            return str(p)
    raise FileNotFoundError(f"RGB not found for stem={stem} under {rgb_dir}")


def _frame_paths_from_episode(*, episode_dir: str, idx: int, dual_view: bool) -> Dict[str, Tuple[str, str]]:
    ep = Path(episode_dir)
    back_depths = sorted((ep / "back_depth").glob("*.npy"))
    if idx < 0 or idx >= len(back_depths):
        raise IndexError(f"idx out of range: {idx} / {len(back_depths)}")
    stem = back_depths[idx].stem

    out = {
        "back": (
            _find_rgb_with_stem(ep / "back", stem),
            str(back_depths[idx]),
        )
    }
    if dual_view:
        out["side"] = (
            _find_rgb_with_stem(ep / "side", stem),
            str(ep / "side_depth" / f"{stem}.npy"),
        )
    return out


def _frame_paths_from_json(*, json_path: str, data_root: str, item: Dict[str, Any], dual_view: bool) -> Dict[str, Tuple[str, str]]:
    images = item.get("images", {})
    depths = item.get("depths", {})
    if not isinstance(images, dict) or not isinstance(depths, dict):
        raise ValueError("Bad item format: images/depths must be dict")

    out = {
        "back": (
            _resolve_asset_path(json_path=json_path, data_root=data_root, rel_or_abs=str(images["back"])),
            _resolve_asset_path(json_path=json_path, data_root=data_root, rel_or_abs=str(depths["back_depth"])),
        )
    }
    if dual_view:
        out["side"] = (
            _resolve_asset_path(json_path=json_path, data_root=data_root, rel_or_abs=str(images["side"])),
            _resolve_asset_path(json_path=json_path, data_root=data_root, rel_or_abs=str(depths["side_depth"])),
        )
    return out


def _camera_payload(rgb_path: str, depth_path: str, depth_unit_scale: float) -> Dict[str, Any]:
    rgb = np.asarray(load_rgb_any(rgb_path))
    depth = np.asarray(load_depth_any(depth_path))
    return {
        "rgb_codec": "jpeg",
        "rgb_bytes": _encode_rgb_jpeg(rgb),
        "rgb_shape_hw": [int(rgb.shape[0]), int(rgb.shape[1])],
        "depth_codec": "npy_zlib",
        "depth_bytes": _encode_depth_npy_zlib(depth),
        "depth_shape_hw": [int(depth.shape[0]), int(depth.shape[1])],
        "depth_unit_scale": float(depth_unit_scale),
    }


def _stats(vals: List[float]) -> Dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": float("nan"), "p50": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
    }


async def run(args: argparse.Namespace) -> None:
    websockets = _require_websockets()

    if bool(args.episode_dir):
        items = _load_episode_items(str(args.episode_dir))
    else:
        items = _load_json_items(str(args.json))

    start_idx = int(args.start_idx)
    if start_idx < 0 or start_idx >= len(items):
        raise ValueError(f"start_idx out of range: {start_idx} / {len(items)}")
    end_idx = len(items) if int(args.max_steps) <= 0 else min(len(items), start_idx + int(args.max_steps))

    infer_ms_list: List[float] = []
    policy_list: List[str] = []
    row_list: List[Dict[str, Any]] = []

    async with websockets.connect(str(args.url), max_size=None) as ws:
        for idx in range(start_idx, end_idx):
            item = items[idx]
            q = _extract_q(item)
            step_id = _extract_step_id(item, idx)

            if bool(args.episode_dir):
                frame_paths = _frame_paths_from_episode(episode_dir=str(args.episode_dir), idx=idx, dual_view=bool(args.dual_view))
            else:
                frame_paths = _frame_paths_from_json(
                    json_path=str(args.json),
                    data_root=str(args.data_root),
                    item=item,
                    dual_view=bool(args.dual_view),
                )

            if bool(args.dual_view):
                cameras = {
                    cam_name: _camera_payload(rgb_path=rgb_path, depth_path=depth_path, depth_unit_scale=float(args.depth_unit_scale))
                    for cam_name, (rgb_path, depth_path) in frame_paths.items()
                }
                obs = ObsPCL(
                    t_client_send_ns=time.time_ns(),
                    step_id=int(step_id),
                    q=q,
                    gripper=float(args.gripper),
                    cam_id=str(args.primary_cam_id),
                    depth_unit_scale=float(args.depth_unit_scale),
                    cameras=cameras,
                )
            else:
                rgb_path, depth_path = frame_paths["back"]
                single = _camera_payload(rgb_path=rgb_path, depth_path=depth_path, depth_unit_scale=float(args.depth_unit_scale))
                obs = ObsPCL(
                    t_client_send_ns=time.time_ns(),
                    step_id=int(step_id),
                    q=q,
                    gripper=float(args.gripper),
                    cam_id=str(args.primary_cam_id),
                    depth_unit_scale=float(args.depth_unit_scale),
                    rgb_codec=str(single["rgb_codec"]),
                    rgb_bytes=single["rgb_bytes"],
                    rgb_shape_hw=list(single["rgb_shape_hw"]),
                    depth_codec=str(single["depth_codec"]),
                    depth_bytes=single["depth_bytes"],
                    depth_shape_hw=list(single["depth_shape_hw"]),
                )

            req = InferRequestPCL(request_id=str(uuid.uuid4()), obs=obs).to_envelope()
            await ws.send(encode_message(req))
            resp_raw = await asyncio.wait_for(ws.recv(), timeout=float(args.request_timeout_s))

            if isinstance(resp_raw, str):
                raise RuntimeError("Expected binary msgpack payload, got text frame.")
            resp = decode_message(resp_raw)

            if int(resp.get("schema_version", -1)) != SCHEMA_VERSION_PCL:
                raise RuntimeError(f"Bad schema_version: {resp.get('schema_version')}")
            if resp.get("type") == "error_pcl":
                raise RuntimeError(f"Server error: {resp}")
            if resp.get("type") != "infer_response_pcl":
                raise RuntimeError(f"Unexpected response type: {resp.get('type')}")

            payload = dict(resp.get("payload", {}))
            timing = payload.get("server_timing", {}) if isinstance(payload, dict) else {}
            infer_ms = float(timing.get("infer_ms", float("nan"))) if isinstance(timing, dict) else float("nan")
            policy = str(timing.get("policy", "")) if isinstance(timing, dict) else ""

            infer_ms_list.append(infer_ms)
            policy_list.append(policy)
            row = {"idx": int(idx), "step_id": int(step_id), "infer_ms": infer_ms, "policy": policy}
            row_list.append(row)
            print(f"[{idx}] step_id={step_id} infer_ms={infer_ms:.3f} policy={policy}")

            if bool(args.print_actions):
                actions = payload.get("actions", None)
                if actions is not None:
                    arr = np.asarray(actions)
                    if arr.ndim >= 2 and arr.shape[0] > 0:
                        print("  actions[0]:", arr[0].tolist())

            if float(args.sleep_s) > 0.0:
                await asyncio.sleep(float(args.sleep_s))

    summary = {
        "url": str(args.url),
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "steps": int(len(row_list)),
        "dual_view": bool(args.dual_view),
        "primary_cam_id": str(args.primary_cam_id),
        "infer_ms": _stats(infer_ms_list),
        "unique_policy_count": int(len(set(policy_list))),
        "rows": row_list,
    }

    print("")
    print("== Replay Summary ==")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False, indent=2))

    if str(args.report_json).strip():
        out = Path(str(args.report_json)).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved report: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="pw_replay_acceptance")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--json", type=str, default="")
    src.add_argument("--episode-dir", type=str, default="")
    ap.add_argument("--data-root", type=str, default="/home/datasets/FrankaNav/test")
    ap.add_argument("--url", type=str, default="ws://127.0.0.1:9011")
    ap.add_argument("--primary-cam-id", type=str, default="back")
    ap.add_argument("--dual-view", action="store_true")
    ap.add_argument("--start-idx", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=16)
    ap.add_argument("--sleep-s", type=float, default=0.0)
    ap.add_argument("--gripper", type=float, default=0.0)
    ap.add_argument("--depth-unit-scale", type=float, default=1.0)
    ap.add_argument("--request-timeout-s", type=float, default=10.0)
    ap.add_argument("--report-json", type=str, default="")
    ap.add_argument("--print-actions", action="store_true")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()