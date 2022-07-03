#!/bin/bash

#SBATCH --job-name=ippo_waterworld
#SBATCH --partition=long                        
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=60G                                     
#SBATCH --time=16:00:00
#SBATCH --array=1-10
#SBATCH -o /network/scratch/o/oussama.boussif/slurms/ippo_waterworld-slurm-%A_%a.out

param_store=scripts/seeds.txt
seed=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $1}')
# 1. Load the required modules
module --quiet load anaconda/3
conda activate marl

python run.py \
policy=ippo \
env=waterworld \
runner.params.lr_decay=False \
runner.params.comet.project_name=waterworld \
runner.params.total_timesteps=10000000 \
n_agents=5 \
seed=$seed \
continuous_action=True \
env_steps=500