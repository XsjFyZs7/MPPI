#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-all}"          # server | replay | all
PROFILE="${2:-obs_infl}"    # no_pw | obs_only | obs_infl

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
POINTWORLD_ROOT="${POINTWORLD_ROOT:-/workspace/pointworld}"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/third_party/co-tracker:${REPO_ROOT}/third_party/curobo:${PYTHONPATH:-}"
if [[ -d "${POINTWORLD_ROOT}" ]]; then
  export PYTHONPATH="${POINTWORLD_ROOT}:${PYTHONPATH}"
fi

PORT="${PORT:-9011}"
URL="${URL:-ws://127.0.0.1:${PORT}}"
PRIMARY_CAM_ID="${PRIMARY_CAM_ID:-back}"
SERVER_WAIT_S="${SERVER_WAIT_S:-8}"

OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/data/pw_acceptance/${PROFILE}}"
ACCEPT_DIR="${MPPI_PW_ACCEPTANCE_DUMP_DIR:-${OUT_ROOT}/server}"
REPORT_JSON="${REPORT_JSON:-${OUT_ROOT}/client_report.json}"

JSON_PATH="${JSON_PATH:-}"
DATA_ROOT="${DATA_ROOT:-/home/datasets/FrankaNav/test}"
EPISODE_DIR="${EPISODE_DIR:-}"
DUAL_VIEW="${DUAL_VIEW:-1}"
START_IDX="${START_IDX:-0}"
MAX_STEPS="${MAX_STEPS:-16}"
SLEEP_S="${SLEEP_S:-0.0}"
GRIPPER="${GRIPPER:-0.0}"
DEPTH_UNIT_SCALE="${DEPTH_UNIT_SCALE:-1.0}"

mkdir -p "${OUT_ROOT}"
rm -rf "${ACCEPT_DIR}"
mkdir -p "${ACCEPT_DIR}"

case "${PROFILE}" in
  no_pw)
    export MPPI_PW_TASK_ABLATION="no_pw"
    ;;
  obs_only)
    export MPPI_PW_TASK_ABLATION="obs_only"
    ;;
  obs_infl)
    export MPPI_PW_TASK_ABLATION="obs_infl"
    ;;
  *)
    echo "Unknown PROFILE=${PROFILE}, expected: no_pw | obs_only | obs_infl" >&2
    exit 2
    ;;
esac

export MPPI_PW_ENABLE="${MPPI_PW_ENABLE:-1}"
export MPPI_USE_POINTWORLD_COST="${MPPI_USE_POINTWORLD_COST:-1}"
export MPPI_PW_COST_MODE="${MPPI_PW_COST_MODE:-task_point_goal_l2}"
export MPPI_PW_AABB_CONFIG_PATH="${MPPI_PW_AABB_CONFIG_PATH:-${REPO_ROOT}/configs/pointworld_static_aabbs.json}"
export MPPI_PW_TASK_W_OBS="${MPPI_PW_TASK_W_OBS:-1.0}"
export MPPI_PW_TASK_W_INFL="${MPPI_PW_TASK_W_INFL:-0.5}"
export MPPI_PW_ACCEPTANCE_DUMP_DIR="${ACCEPT_DIR}"

run_server() {
  export MPPI_PCL_SAVE_PCD="${MPPI_PCL_SAVE_PCD:-0}"
  export MPPI_CAM_ID="${PRIMARY_CAM_ID}"
  export MPPI_POLICY="${MPPI_POLICY:-mppi_joint}"
  export MPPI_OPEN_LOOP_HORIZON="${MPPI_OPEN_LOOP_HORIZON:-11}"
  bash "${REPO_ROOT}/scripts/test_cuRobo_pcl.sh"
}

run_replay() {
  local -a cmd
  cmd=(
    python3 "${REPO_ROOT}/scripts/pw_replay_acceptance.py"
    --url "${URL}"
    --primary-cam-id "${PRIMARY_CAM_ID}"
    --start-idx "${START_IDX}"
    --max-steps "${MAX_STEPS}"
    --sleep-s "${SLEEP_S}"
    --gripper "${GRIPPER}"
    --depth-unit-scale "${DEPTH_UNIT_SCALE}"
    --report-json "${REPORT_JSON}"
  )

  if [[ -n "${EPISODE_DIR}" ]]; then
    cmd+=(--episode-dir "${EPISODE_DIR}")
  elif [[ -n "${JSON_PATH}" ]]; then
    cmd+=(--json "${JSON_PATH}" --data-root "${DATA_ROOT}")
  else
    echo "Set either EPISODE_DIR or JSON_PATH before replay." >&2
    exit 2
  fi

  if [[ "${DUAL_VIEW}" == "1" ]]; then
    cmd+=(--dual-view)
  fi

  "${cmd[@]}"
}

summarize_acceptance() {
  python3 - "${ACCEPT_DIR}" "${PROFILE}" "${REPORT_JSON}" <<'PY'
import json
import sys
from pathlib import Path

accept_dir = Path(sys.argv[1])
profile = sys.argv[2]
report_json = Path(sys.argv[3])

files = sorted(accept_dir.glob("*.json"))
if not files:
    print("FAIL: no server acceptance json generated")
    raise SystemExit(1)

rows = [json.loads(p.read_text(encoding="utf-8")) for p in files]

checks = [
    ("scene_flows", all(bool(r.get("has_scene_flows")) for r in rows)),
    ("scene_visibility", all(bool(r.get("has_scene_visibility")) for r in rows)),
    ("scene_depth_valid_mask", all(bool(r.get("has_scene_depth_valid_mask")) for r in rows)),
    ("runtime_policy", all(bool(r.get("has_runtime_policy")) for r in rows)),
]

if profile != "no_pw":
    checks.append(("task_n_obs", all(bool(r.get("has_task_n_obs")) for r in rows)))
if profile == "obs_infl":
    checks.append(("task_n_infl", all(bool(r.get("has_task_n_infl")) for r in rows)))

print("")
print("== Acceptance Summary ==")
print("server_rows:", len(rows))
for name, ok in checks:
    print(f"{name}: {'PASS' if ok else 'FAIL'}")

if report_json.is_file():
    rep = json.loads(report_json.read_text(encoding="utf-8"))
    infer = rep.get("infer_ms", {})
    print("client_steps:", rep.get("steps"))
    print("client_infer_ms_mean:", infer.get("mean"))
    print("client_infer_ms_p95:", infer.get("p95"))
    print("client_unique_policy_count:", rep.get("unique_policy_count"))
else:
    print("client_report: MISSING")

failed = [name for name, ok in checks if not ok]
if failed:
    print("FINAL: FAIL", ",".join(failed))
    raise SystemExit(1)

print("FINAL: PASS")
PY
}

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

case "${ACTION}" in
  server)
    run_server
    ;;
  replay)
    run_replay
    summarize_acceptance
    ;;
  all)
    run_server &
    SERVER_PID=$!
    sleep "${SERVER_WAIT_S}"
    run_replay
    summarize_acceptance
    ;;
  *)
    echo "Unknown ACTION=${ACTION}, expected: server | replay | all" >&2
    exit 2
    ;;
esac