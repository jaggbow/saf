#!/bin/bash

#SBATCH --job-name=ippo_spread
#SBATCH --partition=long                        
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=60G                                     
#SBATCH --time=4:00:00
#SBATCH -o /network/scratch/o/oussama.boussif/slurms/ippo_spread-slurm-%j.out  

# 1. Load the required modules
module --quiet load anaconda/3
conda activate marl

python run.py \
policy=ippo \
env=simple_spread \
n_agents=3 \
continuous_action=False \
env_steps=25 \
runner.params.lr_decay=False