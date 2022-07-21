#!/bin/bash

#SBATCH --job-name=ippo_spread
#SBATCH --partition=long                        
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=60G                                     
#SBATCH --time=4:00:00
#SBATCH --array=1-10


#param_store=scripts/seeds.txt
#seed=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $1}')
seed=1
# 1. Load the required modules
module --quiet load anaconda/3
#conda activate marl
conda activate PettingZoo


HYDRA_FULL_ERROR=1 python run.py \
env=marlgrid \
env.name=ClutteredGoalTileCoordinationEnv \
env.params.max_steps=200 \
env.params.coordination=2 \
n_agents=3 \
env_steps=100 \
env.params.num_goals=100 \
policy=mappo \
policy.params.type=conv \
runner.params.comet.project_name=test \
runner.params.comet.experiment_name=test
