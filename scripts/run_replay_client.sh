cd /home/wangyuhan/MPPI

PYTHONPATH=/home/wangyuhan/MPPI/src:/home/wangyuhan/MPPI/third_party/co-tracker:/home/wangyuhan/MPPI/third_party/curobo:/home/wangyuhan/PointWorld:/home/wangyuhan/PointWorld/third_party/dinov3 \
python3 /home/wangyuhan/MPPI/tests/pw_replay_acceptance.py \
  --url ws://127.0.0.1:9011 \
  --episode-dir /home/datasets/FrankaNav/ep_00152 \
  --primary-cam-id back \
  --dual-view \
  --start-idx 0 \
  --max-steps 30 \
  --sleep-s 0.0 \
  --gripper 0.0 \
  --depth-unit-scale 1.0 \
  --request-timeout-s 120 \
  --goal-ee-xyz 0.55,0.00,0.20 \
  --report-json /home/wangyuhan/MPPI/data/pw_acceptance/obs_infl/client_report.json