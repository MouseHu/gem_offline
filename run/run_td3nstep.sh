#!/usr/bin/env bash
declare -a gpus=(0 1 2 3)
declare -a envs=("Ant-v2" "HalfCheetah-v2" "Walker2d-v2" "Humanoid-v2")
declare -a envs_alias=("ant" "halfcheetah" "walker" "humanoid")
#declare -a gpus=(0 1)
#declare -a envs=("Hopper-v2" "Swimmer-v2")
#declare -a envs_alias=("hopper" "swimmer")
export PYTHONPATH=/home/hh/gem_mujoco/
#export PYTHONPATH=/home/hh/continous/amc_baselines/
for ((i = 0 ; i < ${#gpus[@]} ; i++)); do
    for ((seed = 0 ; seed < 5 ; seed++));do
        CUDA_VISIBLE_DEVICES=${gpus[$i]} OPENAI_LOGDIR=/home/hh/log_ddq/ddq_supp nohup python /home/hh/amc_sbaselines/run/run_baseline.py --agent=TD3NSTEP --comment=${envs_alias[$i]}_td3nstep_1000_$1_$seed --env-id=${envs[$i]} > ./logs/${envs_alias[$i]}_td3nstep_1000_$1_$seed.out &
#        CUDA_VISIBLE_DEVICES=${gpus[$i]} OPENAI_LOGDIR=/home/hh/ddq_supp nohup python /home/hh/continous/amc_baselines/run/run_baseline.py --agent=TD3NSTEP --comment=${envs_alias[$i]}_td3nstep_1000_$1_$seed --env-id=${envs[$i]} > ./logs/${envs_alias[$i]}_td3nstep_1000_$1_$seed.out &
    done
done