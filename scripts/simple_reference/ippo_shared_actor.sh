#!/bin/bash

#SBATCH --job-name=ippo_reference_shared_actor
#SBATCH --partition=long                        
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=60G                                     
#SBATCH --time=4:00:00
#SBATCH -o /network/scratch/o/oussama.boussif/slurms/ippo_reference_shared_actor-slurm-%j.out  

# 1. Load the required modules
module --quiet load anaconda/3
conda activate marl

python run.py policy=ippo \
env=simple_reference \
runner.params.lr_decay=False \
policy.params.shared_actor=True \
policy.params.shared_critic=False \
policy.params.n_agents=2 \
runner.params.n_agents=2 \
buffer.n_agents=2 \
runner.params.comet.project_name=simple_reference \
runner.params.total_timesteps=3000000
