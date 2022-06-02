#!/bin/bash

#SBATCH --job-name=mappo_reference_shared
#SBATCH --partition=long                        
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=60G                                     
#SBATCH --time=3:00:00
#SBATCH -o /network/scratch/o/oussama.boussif/slurms/mappo_reference_shared-slurm-%j.out  

# 1. Load the required modules
module --quiet load anaconda/3
conda activate marl

python run.py \
policy=mappo \
env=simple_reference \
runner.params.lr_decay=False \
policy.params.shared_critic=True \
policy.params.n_agents=2 \
runner.params.n_agents=2 \
buffer.n_agents=2 \
runner.params.comet.project_name=simple_reference \
runner.params.total_timesteps=3000000