#!/bin/bash

#SBATCH --job-name=marlgrid_baseline
#SBATCH --partition=long                        
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=70G                                     
#SBATCH --time=02-00:00:00



# 1. Load the required modules
module --quiet load anaconda/3
conda activate marl
#conda activate PettingZoo

HYDRA_FULL_ERROR=1 python testing.py \
env=marlgrid  \
env.name=ClutteredGoalTileTeamsupportNHeterogneityEnv \
env.params.max_steps=50 \
env.params.coordination=2 \
env.params.heterogeneity=1 \
seed=6 \
n_agents=2 \
env_steps=50 \
env.params.num_goals=30 \
env.params.grid_size=10 \
experiment_name=test \
policy=mappo \
policy.params.type=conv \
policy.params.activation=tanh \
policy.params.update_epochs=10 \
policy.params.num_minibatches=1 \
policy.params.learning_rate=0.0007 \
policy.params.shared_actor=False \
policy.params.shared_critic=False \
policy.params.clip_vloss=True \
runner.params.lr_decay=False \
runner.params.comet.project_name=test
