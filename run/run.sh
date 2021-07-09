#!/usr/bin/env bash
#declare algo_alias="td3doubletwin"
#declare algo="TD3DoubleTwin"

declare algo_alias="ot"
declare algo="OT"

#declare algo_alias="ddq_100000"
#declare algo="AMC"

#declare root="/home/hh/gem_mujoco/"
declare root="/home/hh/continous/amc_baselines"
declare max_steps=1000

#declare -a gpus=(2)
#declare -a envs=("Humanoid-v2")
#declare -a envs_alias=("humanoid")

#declare -a gpus=(3 4 5)
#declare -a envs=("Hopper-v2" "Walker2d-v2" "Humanoid-v2")
#declare -a envs_alias=("hopper" "walker" "humanoid")

#declare -a gpus=(3 4)
#declare -a envs=("Ant-v2" "Humanoid-v2")
#declare -a envs_alias=("ant" "humanoid")

declare -a gpus=(0 1)
declare -a envs=("Ant-v2" "Humanoid-v2")
declare -a envs_alias=("ant" "humanoid")

export PYTHONPATH=$root

for ((i = 0; i < ${#gpus[@]}; i++)); do
  for ((seed = 0; seed < 5; seed++)); do
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=${gpus[$i]} OPENAI_LOGDIR=/home/hh/log_ddq/$algo_alias nohup python $root/run/run_baseline.py --agent=$algo --comment="${envs_alias[$i]}"_${algo_alias}_$1_"$seed" --env-id="${envs[$i]}" >./logs/"${envs_alias[$i]}"_${algo_alias}_$1_"$seed".out &
  done
  sleep 3
done
