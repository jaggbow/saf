#!/bin/bash

#SBATCH --account=research-eemcs-me
#SBATCH --job-name=mappo_waterworld
#SBATCH --partition=gpu                       
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu
#SBATCH --mem=60G     
#SBATCH --ntasks=1                                
#SBATCH --time=23:59:00
#SBATCH --array=1-3
#SBATCH -o /scratch/cristianmeo/output/waterworld-%A_%a.out  
#SBATCH -e /scratch/cristianmeo/output/waterworld-%A_%a.err  

# 1. Load the required modules
#env_name=="TeamTogetherEnv"
#env_name=="TeamSupportEnv"
#env_name=="ClutteredGoalTileTeamsupportNHeterogneityEnv"

module --quiet load anaconda/3
conda activate MARL

env_name=$1 
N_agents=10
coordination=1
heterogeneity=1
policy=$2
use_policy_pool=$3
latent_kl=$4
seed=1
num_goals=100
grid_size=$5

HYDRA_FULL_ERROR=1 python run.py \
env=marlgrid \
env.name=${env_name} \
env.params.max_steps=50 \
env.params.coordination=${coordination} \
env.params.heterogeneity=${heterogeneity} \
seed=${seed} \
n_agents=${N_agents} \
env_steps=50 \
env.params.num_goals=100 \
experiment_name=OOD_coordination_test \
policy=${policy} \
policy.params.type=conv \
policy.params.activation=tanh \
policy.params.update_epochs=10 \
policy.params.num_minibatches=1 \
policy.params.learning_rate=0.0007 \
policy.params.shared_actor=False \
policy.params.shared_critic=False \
policy.params.clip_vloss=True \
runner.params.lr_decay=False \
runner.params.checkpoint_dir=${env_name}"_"${N_agents}"_"${coordination}"_"${heterogeneity}"_"${policy}"-"${use_policy_pool}"-"${latent_kl}"_"${seed} \
runner.params.comet.project_name=OOD_coordination_test \
test_mode = True \
latent_kl=${latent_kl} 
