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
env=simple_reference \
runner.params.lr_decay=False \
n_agents=2 \
continuous_action=False \
env_steps=25 \
runner.params.comet.project_name=simple_reference \
runner.params.total_timesteps=3000000