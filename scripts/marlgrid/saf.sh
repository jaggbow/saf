#!/bin/bash

#SBATCH --job-name=saf_marlgrid
#SBATCH --partition=long                        
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=65G                                     
#SBATCH --time=23:50:00

env=$1
N_agents=$2
Method=$3
coordination=$4
heterogeneity=$5
use_policy_pool=$6
latent_kl=$7
seed=$8
ProjectName=$9
conda_env=${10}
# 1. Load the required modules
module --quiet load anaconda/3
conda activate ${conda_env}

ExpName=${env}"_"${N_agents}"_"${coordination}"_"${heterogeneity}"_"${Method}"-"${use_policy_pool}"-"${latent_kl}"_"${seed}
echo "doing experiment: ${ExpName}"




HYDRA_FULL_ERROR=1 python run.py \
env=marlgrid  \
env.name=${env} \
env.params.max_steps=50 \
env.params.coordination=${coordination} \
env.params.heterogeneity=${heterogeneity} \
seed=${seed} \
n_agents=${N_agents} \
env_steps=100 \
env.params.num_goals=100 \
experiment_name=${ExpName} \
policy=${Method} \
policy.params.type=conv \
policy.params.activation=tanh \
policy.params.update_epochs=10 \
policy.params.num_minibatches=1 \
policy.params.learning_rate=0.0007 \
policy.params.clip_vloss=True \
runner.params.lr_decay=False \
runner.params.comet.project_name=$ProjectName \
use_policy_pool=${use_policy_pool} \
latent_kl=${latent_kl} \

