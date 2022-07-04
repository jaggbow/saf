#!/bin/bash

#SBATCH --job-name=mappo_reference
#SBATCH --partition=long                        
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=60G                                     
#SBATCH --time=3:00:00
#SBATCH -o /network/scratch/o/oussama.boussif/slurms/mappo_reference-slurm-%j.out  

# 1. Load the required modules
module --quiet load anaconda/3
conda activate marl

python run.py \
policy=mappo \
env=simple_world_comm \
runner.params.lr_decay=False \
policy.params.shared_critic=False \
policy.params.shared_actor=False \
n_agents=6 \
continuous_action=False \
env_steps=25 \
runner.params.comet.project_name=simple_world_comm \
runner.params.total_timesteps=2000000