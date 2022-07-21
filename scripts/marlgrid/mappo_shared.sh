#!/bin/bash

#SBATCH --job-name=mappo_shared
#SBATCH --partition=long                        
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=60G                                     
#SBATCH --time=3:00:00
#SBATCH -o /network/scratch/o/oussama.boussif/slurms/mappo_shared-slurm-%j.out  

# 1. Load the required modules
module --quiet load anaconda/3
conda activate marl

python run.py \
policy=mappo \
env=simple_spread \
runner.params.lr_decay=False \
n_agents=3 \
continuous_action=False \
env_steps=25 \
policy.params.shared_critic=True