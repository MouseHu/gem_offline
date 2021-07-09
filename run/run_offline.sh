#!/usr/bin/env bash
declare algo_alias="gem_offline"
declare root="/home/hh/offline/mbmarl"
declare max_steps=1000

declare -a gpus=(0)
declare -a envs=("antmaze-umaze-diverse-v0")
declare -a envs_alias=("ant_umaze")

export PYTHONPATH=$root

for ((i = 0; i < ${#gpus[@]}; i++)); do
  for ((seed = 0; seed < 5; seed++)); do
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=${gpus[$i]} OPENAI_LOGDIR=/home/hh/log_gem_offline/$algo_alias nohup python $root/run/run_offline.py --comment="${envs_alias[$i]}"_${algo_alias}_$1_"$seed" --env-id="${envs[$i]}" >./logs/"${envs_alias[$i]}"_${algo_alias}_$1_"$seed".out &
  done
  sleep 1
done
