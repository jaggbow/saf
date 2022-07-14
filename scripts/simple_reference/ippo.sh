#!/bin/bash

#SBATCH --job-name=ippo_reference
#SBATCH --partition=long                        
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=60G                                     
#SBATCH --time=4:00:00
#SBATCH --array=1-10
#SBATCH -o /network/scratch/o/oussama.boussif/slurms/ippo_reference-slurm-%A_%a.out  

param_store=scripts/seeds.txt
seed=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $1}')
# 1. Load the required modules
module --quiet load anaconda/3
conda activate marl

python run.py \
policy=ippo \
policy.params.activation=relu \
policy.params.update_epochs=15 \
policy.params.num_minibatches=1 \
policy.params.learning_rate=0.0007 \
policy.params.clip_vloss=True \
rollout_threads=128 \
env=simple_reference \
runner.params.lr_decay=False \
n_agents=2 \
seed=$seed \
continuous_action=False \
env_steps=25 \
runner.params.comet.project_name=simple_reference \
runner.params.total_timesteps=3000000