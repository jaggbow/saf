#!/bin/bash

#SBATCH --job-name=mappo_3m
#SBATCH --partition=long                        
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=60G                                     
#SBATCH --time=24:00:00
#SBATCH --array=1-10
#SBATCH -o /network/scratch/o/oussama.boussif/slurms/mappo_3m-slurm-%A_%a.out

param_store=scripts/seeds.txt
seed=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $1}')
# 1. Load the required modules
module --quiet load anaconda/3
conda activate marl

python run.py \
policy=mappo \
env=starcraft \
env.name=3m \
runner.params.lr_decay=False \
runner.params.comet.project_name=starcraft \
runner.params.total_timesteps=10000000 \
n_agents=3 \
seed=$seed \
continuous_action=False \
env_steps=60