#!/bin/bash

#SBATCH --job-name=mappo_spread
#SBATCH --partition=long                        
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=60G                                     
#SBATCH --time=3:00:00
#SBATCH -o /network/scratch/o/oussama.boussif/slurms/mappo_spread-slurm-%j.out  

# 1. Load the required modules
module --quiet load anaconda/3
conda activate marl

python run.py policy=mappo env=simple_spread runner.params.lr_decay=False