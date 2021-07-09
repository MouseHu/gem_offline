#!/usr/bin/env bash
#declare -a gpus=(0)
#declare -a envs=("HalfCheetah-v2")
#declare -a envs_alias=("halfcheetah")
declare -a gpus=(0 1 2 3 6 7)
declare -a envs=("Ant-v2" "HalfCheetah-v2" "Hopper-v2" "Walker2d-v2" "Humanoid-v2" "Swimmer-v2")
declare -a envs_alias=("ant" "halfcheetah" "hopper" "walker" "humanoid" "swimmer")
export PYTHONPATH=/home/hh/amc_sbaselines/
for ((i = 0 ; i < ${#gpus[@]} ; i++)); do
    for ((seed = 0 ; seed < 5 ; seed++));do
        CUDA_VISIBLE_DEVICES=${gpus[$i]} OPENAI_LOGDIR=/home/hh/ddq_baselines nohup python /home/hh/amc_sbaselines/run/run_baseline.py --agent=SAC --comment=${envs_alias[$i]}_sac_baseline_$1_${seed} --env-id=${envs[$i]} > ./logs/${envs_alias[$i]}_sac_baseline_$1_${seed}.out &
    done
done