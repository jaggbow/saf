#!/bin/bash

#SBATCH --job-name=saf_spread
#SBATCH --partition=long                        
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=60G                                     
#SBATCH --time=3:00:00
#SBATCH --array=1-10
#SBATCH -o /network/scratch/o/oussama.boussif/slurms/saf_spread-slurm-%A_%a.out  

param_store=scripts/seeds.txt
seed=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $1}')
# 1. Load the required modules
module --quiet load anaconda/3
conda activate marl

python run.py \
policy=saf \
policy.params.activation=tanh \
policy.params.update_epochs=10 \
policy.params.num_minibatches=1 \
policy.params.learning_rate=0.0007 \
policy.params.shared_actor=False \
policy.params.shared_critic=False \
policy.params.clip_vloss=True \
env=simple_spread \
n_agents=3 \
seed=$seed \
continuous_action=False \
env_steps=25 \
rollout_threads=128 \
runner.params.lr_decay=False \
use_policy_pool = True

HYDRA_FULL_ERROR=1 python run.py \
policy=saf \
policy.params.activation=tanh \
policy.params.update_epochs=10 \
policy.params.num_minibatches=1 \
policy.params.learning_rate=0.0007 \
policy.params.shared_actor=False \
policy.params.shared_critic=False \
policy.params.clip_vloss=True \
env=simple_spread \
n_agents=3 \
seed=1 \
continuous_action=False \
env_steps=25 \
rollout_threads=128 \
runner.params.lr_decay=False \
use_policy_pool=True
