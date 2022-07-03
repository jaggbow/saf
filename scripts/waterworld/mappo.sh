#!/bin/bash

#SBATCH --job-name=mappo_waterworld
#SBATCH --partition=long                        
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=60G                                     
#SBATCH --time=16:00:00
#SBATCH -o /network/scratch/o/oussama.boussif/slurms/mappo_waterworld-slurm-%j.out  

# 1. Load the required modules
module --quiet load anaconda/3
conda activate marl

python run.py \
policy=mappo \
env=waterworld \
runner.params.lr_decay=False \
runner.params.comet.project_name=waterworld \
runner.params.total_timesteps=10000000 \
n_agents=5 \
continuous_action=True \
env_steps=500