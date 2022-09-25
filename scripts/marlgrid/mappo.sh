#!/bin/bash

#SBATCH --job-name=marlgrid_baseline
#SBATCH --account=rrg-bengioy-ad
#SBATCH --gres=gpu:1
#SBATCH --mem=65G                                     
#SBATCH --time=02-12:00:00
#SBATCH --output=/home/veds12/scratch/saf/slurms/mappo-%j.out

#param_store=scripts/seeds.txt
#seed=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $1}')
env=$1
N_agents=$2
Method=$3
coordination=$4
heterogeneity=$5
seed=$6
ProjectName=$7
conda_env=$8

# 1. Load the required modules
module load python/3
module load httpproxy
source /home/veds12/scratch/venv/saf/bin/activate

ExpName=${env}"_"${N_agents}"_"${coordination}"_"${heterogeneity}"_"${Method}"_"${seed}
echo "doing experiment: ${ExpName}"

HYDRA_FULL_ERROR=1 python run.py \
env=marlgrid  \
rollout_threads=32 \
env.name=${env} \
env.params.max_steps=200 \
env.params.coordination=${coordination} \
env.params.heterogeneity=${heterogeneity} \
seed=${seed} \
n_agents=${N_agents} \
env_steps=100 \
env.params.num_goals=45 \
experiment_name=${ExpName} \
policy=${Method} \
policy.params.type=conv \
policy.params.activation=tanh \
policy.params.update_epochs=10 \
policy.params.num_minibatches=1 \
policy.params.learning_rate=0.0007 \
policy.params.shared_actor=True \
policy.params.shared_critic=True \
policy.params.clip_vloss=True \
runner.params.lr_decay=False \
runner.params.comet.project_name=$ProjectName