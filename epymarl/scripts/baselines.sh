#!/bin/bash

#SBATCH --job-name=baselines
#SBATCH --partition=long                        
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=48G                                     
#SBATCH --time=50:00:00


env=$1
Method=$2
coordination=$3
heterogeneity=$4
seed=$5
# 1. Load the required modules
module --quiet load anaconda/3
conda activate pymarl

ExpName=${env}"_"${coordination}"_"${heterogeneity}"_"${Method}"_"${seed}
echo "doing experiment: ${ExpName}"

HYDRA_FULL_ERROR=1 python3 src/main.py \
--config=${Method} \
--env-config=gymma with env_args.time_limit=50 \
env_args.key="marlgrid:${env}-10Agents-100Goals-v0" \
env_args.coordination=${coordination} \
env_args.heterogeneity=${heterogeneity} \
seed=${seed}