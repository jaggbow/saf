#!/bin/bash

#SBATCH --account=research-eemcs-me
#SBATCH --job-name=OOD_test_coordination
#SBATCH --partition=gpu                       
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu   
#SBATCH --mem-per-cpu=32G
#SBATCH --ntasks=1                                
#SBATCH --time=00:59:00
#SBATCH -o /scratch/cristianmeo/output/OOD_test_coordination-%A_%a.out  
#SBATCH -e /scratch/cristianmeo/output/OOD_test_coordination-%A_%a.err  

# 1. Load the required modules
#env_name=="TeamTogetherEnv"
#env_name=="TeamSupportEnv"
#env_name=="ClutteredGoalTileTeamsupportNHeterogneityEnv"
#runner.params.checkpoint_dir=${env_name}"_"${N_agents}"_"2"_"1"_"${policy}"-"${use_policy_pool}"-"${latent_kl}"_"${seed} \
#experiment_name="OOD_coordination_test" \

module --quiet load anaconda/3
conda activate MARL

HYDRA_FULL_ERROR=1 python run.py \
env=marlgrid \
env.name="TeamTogetherEnv" \
test_mode=True \
runner.params.checkpoint_dir="checkpoints/TeamTogetherEnv_10_2_1_saf-True-True_1" \
env.params.max_steps=50 \
env.params.coordination=3 \
env.params.heterogeneity=1 \
seed=1 \
n_agents=10 \
env_steps=50 \
env.params.num_goals=100 \
policy=saf \
policy.params.type=conv \
policy.params.activation=tanh \
policy.params.update_epochs=10 \
policy.params.num_minibatches=1 \
policy.params.learning_rate=0.0007 \
policy.params.shared_actor=False \
policy.params.shared_critic=False \
policy.params.clip_vloss=True \
runner.params.lr_decay=False \
runner.params.comet.experiment_name="OOD_coordination_test" \
latent_kl=True \
use_policy_pool=True
