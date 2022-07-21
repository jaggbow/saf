#!/bin/bash

#SBATCH --job-name=ippo_shared_actor
#SBATCH --partition=long                        
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=60G                                     
#SBATCH --time=4:00:00
#SBATCH -o /network/scratch/o/oussama.boussif/slurms/ippo_shared_actor-slurm-%j.out  

# 1. Load the required modules
module --quiet load anaconda/3
conda activate marl

python run.py \
policy=ippo \
env=simple_spread \
runner.params.lr_decay=False \
policy.params.shared_actor=True \
n_agents=3 \
continuous_action=False \
env_steps=25 \
policy.params.shared_critic=False