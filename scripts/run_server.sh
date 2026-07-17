cd /home/wangyuhan/MPPI

HOST=${HOST:-0.0.0.0} \
PORT=${PORT:-9011} \
MPPI_OPEN_LOOP_HORIZON=${MPPI_OPEN_LOOP_HORIZON:-11} \
MPPI_POLICY=${MPPI_POLICY:-mppi_joint} \
CAM_ID=${CAM_ID:-back} \
MPPI_PCL_CAM_INFO_BACK_PATH=/home/wangyuhan/MPPI/configs/back_cam_info.yaml \
MPPI_PCL_T_BASE_CAM_BACK_PATH=/home/wangyuhan/MPPI/configs/T_base_cam_back.yaml \
MPPI_PCL_CAM_INFO_SIDE_PATH=/home/wangyuhan/MPPI/configs/side_cam_info.yaml \
MPPI_PCL_T_BASE_CAM_SIDE_PATH=/home/wangyuhan/MPPI/configs/T_base_cam_side.yaml \
MPPI_PCL_VERBOSE=1 \
MPPI_PCL_PRINT_EVERY=1 \
MPPI_PCL_HEARTBEAT_S=10.0 \
MPPI_PW_ENABLE=1 \
MPPI_USE_POINTWORLD_COST=1 \
MPPI_W_POINTWORLD=1.0 \
MPPI_PW_COST_MODE=task_point_goal_l2 \
MPPI_PW_TASK_ABLATION=obs_infl \
MPPI_PW_TASK_W_OBS=1.0 \
MPPI_PW_TASK_W_INFL=0.5 \
MPPI_PW_AABB_CONFIG_PATH=/home/wangyuhan/MPPI/configs/pointworld_static_aabbs.json \
MPPI_USE_CUROBO_COLLISION=0 \
MPPI_W_EE_POS=1.0 \
MPPI_W_SMOOTH=0.0 \
MPPI_W_ACTION=0.0 \
MPPI_W_JOINT_LIMIT=0.0 \
MPPI_TEMPERATURE=0.05 \
MPPI_NOISE_MODE=${MPPI_NOISE_MODE:-spline} \
MPPI_NOISE_NKNOTS=${MPPI_NOISE_NKNOTS:-4} \
MPPI_NOISE_DEGREE=${MPPI_NOISE_DEGREE:-3} \
MPPI_NOISE_STD_MIN=${MPPI_NOISE_STD_MIN:-0.05} \
MPPI_NOISE_STD_MAX=${MPPI_NOISE_STD_MAX:-0.50} \
MPPI_NOISE_SCHEDULE=${MPPI_NOISE_SCHEDULE:-linear} \
MPPI_PW_MODEL_PATH=/home/models/PointWorld/PointWorld_models/large-droid/model-best.pt \
MPPI_PW_COTRACKER_CKPT=/home/models/Co-tracker/scaled_online.pth \
POINTWORLD_ROOT=/home/wangyuhan/PointWorld \
DINOv3_ROOT=/home/wangyuhan/PointWorld/third_party/dinov3 \
MPPI_PW_MODEL_DEVICE=cuda:0,cuda:1 \
MPPI_PW_COTRACKER_DEVICE=cuda:0,cuda:1 \
MPPI_PW_ROBOT_SAMPLER_DEVICE=cuda:0,cuda:1 \
MPPI_PW_MODEL_DOMAIN=droid \
MPPI_URDF_PATH=/home/wangyuhan/PointWorld/assets/franka_description/franka_panda_robotiq_2f85.urdf \
MPPI_PW_URDF_PATH=/home/wangyuhan/PointWorld/assets/franka_description/franka_panda_robotiq_2f85.urdf \
PYTHONPATH=/home/wangyuhan/MPPI/src:/home/wangyuhan/MPPI/third_party/co-tracker:/home/wangyuhan/MPPI/third_party/curobo:/home/wangyuhan/PointWorld:/home/wangyuhan/PointWorld/third_party/dinov3 \
python3 -u -m mppi.comm.ws_server_async_pcl \
  --host "${HOST}" \
  --port "${PORT}" \
  --open-loop-horizon "${MPPI_OPEN_LOOP_HORIZON}" \
  --policy "${MPPI_POLICY}" \
  --cam-id "${CAM_ID}"