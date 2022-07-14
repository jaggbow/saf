#!/bin/bash

#SBATCH --job-name=mappo_corridor
#SBATCH --partition=long                        
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=60G                                     
#SBATCH --time=24:00:00
#SBATCH --array=1-10
#SBATCH -o /network/scratch/o/oussama.boussif/slurms/mappo_corridor-slurm-%A_%a.out

param_store=scripts/seeds.txt
seed=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $1}')
# 1. Load the required modules
module --quiet load anaconda/3
conda activate marl

python run.py \
policy=mappo \
env=starcraft \
env.name=corridor \
runner.params.lr_decay=False \
runner.params.comet.project_name=starcraft \
runner.params.total_timesteps=10000000 \
policy.params.activation=relu \
policy.params.update_epochs=5 \
policy.params.num_minibatches=1 \
policy.params.learning_rate=0.0005 \
policy.params.shared_actor=True \
policy.params.shared_critic=True \
policy.params.clip_vloss=True \
n_agents=6 \
seed=$seed \
continuous_action=False \
env_steps=400 \
rollout_threads=8