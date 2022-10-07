#!/bin/bash

#SBATCH --account=research-eemcs-me
#SBATCH --job-name=mappo_waterworld
#SBATCH --partition=gpu                       
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu
#SBATCH --mem=60G     
#SBATCH --ntasks=1                                
#SBATCH --time=23:59:00
#SBATCH -o /scratch/cristianmeo/output/waterworld-%A_%a.out  
#SBATCH -e /scratch/cristianmeo/output/waterworld-%A_%a.err  

module --quiet load anaconda/3
conda activate MARL

env_name=$1 
N_agents=$2
coordination=$4
heterogeneity=$5
policy=$3
use_policy_pool=$6
latent_kl=$7
seed=$8
ProjectName=$9

ExpName="OOD_heterogeneity_test_"${env_name}"_"${N_agents}"_"${coordination}"_"${heterogeneity}"_"${policy}"-"${use_policy_pool}"-"${latent_kl}"_"${seed}
echo "doing experiment: ${ExpName}"

HYDRA_FULL_ERROR=1 python run.py \
env=marlgrid \
env.name=${env_name} \
env.params.max_steps=200 \
env.params.coordination=${coordination} \
env.params.heterogeneity=${heterogeneity} \
seed=${seed} \
n_agents=${N_agents} \
env_steps=100 \
env.params.num_goals=45 \
experiment_name=${ExpName} \
policy=${policy} \
policy.params.type=conv \
policy.params.activation=tanh \
policy.params.update_epochs=10 \
policy.params.num_minibatches=1 \
policy.params.learning_rate=0.0007 \
policy.params.clip_vloss=True \
runner.params.lr_decay=False \
runner.params.checkpoint_dir="outputs/"${env_name}"_"${N_agents}"_"1"_"2"_"${policy}"-"${use_policy_pool}"-"${latent_kl}"_"${seed} \
runner.params.comet.project_name=$ProjectName \
test_mode=True \
latent_kl=${latent_kl} \
use_policy_pool=${use_policy_pool} \

