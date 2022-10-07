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

#module --quiet load anaconda/3
#conda activate MARL

ProjectName="OOD_tests"
conda_env="MARL" # Set the name of your conda environment

# OOD Grid size tests

declare -a All_Envs=("TeamSupportEnv" "TeamTogetherEnv" "") #"ClutteredCompoundGoalTileCoordinationHeterogeneityEnv"

declare -a All_N_agents=(10)

declare -a All_Methods=("ippo" "mappo")

declare -a All_coordination=(2)

declare -a All_heterogeneity=(2)

declare -a All_use_policy_pool=(False)

declare -a All_latent_kl=(False)

declare -a All_grid_sizes=(20 30 40 50 60)

Seeds=(1)


#for Env in "${All_Envs[@]}"
#do
#	for N_agents in "${All_N_agents[@]}"
#	do
#		for Method in "${All_Methods[@]}"
#		do	
#			for coordination in "${All_coordination[@]}"
#			do
#				for heterogeneity in "${All_heterogeneity[@]}"
#				do
#					for use_policy_pool in "${All_use_policy_pool[@]}"
#					do	
#						for latent_kl in "${All_latent_kl[@]}"
#						do					
#							for grid_size in "${All_grid_sizes[@]}"
#							do
#								for Seed in "${Seeds[@]}"
#								do
#									sbatch scripts/OOD_tests/OOD_size_env_test.sh $Env $N_agents $Method $coordination $heterogeneity $use_policy_pool $latent_kl $Seed $ProjectName $grid_size
#								done
#							done
#						done
#					done
#				done			
#			done
#		done
#	done
#done
#declare -a All_Methods=("saf")
#declare -a All_use_policy_pool=(False True)
#declare -a All_latent_kl=(False True)
#for Env in "${All_Envs[@]}"
#do
#	for N_agents in "${All_N_agents[@]}"
#	do
#		for Method in "${All_Methods[@]}"
#		do	
#			for coordination in "${All_coordination[@]}"
#			do
#				for heterogeneity in "${All_heterogeneity[@]}"
#				do
#					for use_policy_pool in "${All_use_policy_pool[@]}"
#					do	
#						for latent_kl in "${All_latent_kl[@]}"
#						do					
#							for grid_size in "${All_grid_sizes[@]}"
#							do
#								for Seed in "${Seeds[@]}"
#								do
#									sbatch scripts/OOD_tests/OOD_size_env_test.sh $Env $N_agents $Method $coordination $heterogeneity $use_policy_pool $latent_kl $Seed $ProjectName $grid_size		
#								done
#							done
#						done
#					done
#				done			
#			done
#		done
#	done
#done

# OOD N agents tests

#declare -a All_Methods=("saf")
#
#declare -a All_coordination=(2)
#
#declare -a All_heterogeneity=(1)
#
#declare -a All_N_agents=(10)
#
#declare -a All_use_policy_pool=(False True)
#
#declare -a All_latent_kl=(False True)
#declare -a All_grid_sizes=(20 30 40 50 60)
#
#Seeds=(1)
#
#for Env in "${All_Envs[@]}"
#do
#	for N_agents in "${All_N_agents[@]}"
#	do
#		for Method in "${All_Methods[@]}"
#		do
#			for coordination in "${All_coordination[@]}"
#			do
#				for heterogeneity in "${All_heterogeneity[@]}"
#				do
#					for use_policy_pool in "${All_use_policy_pool[@]}"
#					do	
#						for latent_kl in "${All_latent_kl[@]}"
#						do		
#							for grid_size in "${All_grid_sizes[@]}"			
#							do
#								for Seed in "${Seeds[@]}"
#								do
#									sbatch scripts/OOD_tests/OOD_size_env_test.sh $Env $N_agents $Method $coordination $heterogeneity $use_policy_pool $latent_kl $Seed $ProjectName $grid_size
#								done
#							done
#						done
#					done
#				done			
#			done
#		done
#	done
#done
#
#
## OOD N agents tests
#
#declare -a All_Methods=("saf")
#
#declare -a All_coordination=(2)
#
#declare -a All_heterogeneity=(1)
#
#declare -a All_N_agents=(5 10 15 20 25)
#
#declare -a All_use_policy_pool=(False True)
#
#declare -a All_latent_kl=(False True)
#
#Seeds=(1)
#
#for Env in "${All_Envs[@]}"
#do
#	for N_agents in "${All_N_agents[@]}"
#	do
#		for Method in "${All_Methods[@]}"
#		do
#			for coordination in "${All_coordination[@]}"
#			do
#				for heterogeneity in "${All_heterogeneity[@]}"
#				do
#					for use_policy_pool in "${All_use_policy_pool[@]}"
#					do	
#						for latent_kl in "${All_latent_kl[@]}"
#						do					
#							for Seed in "${Seeds[@]}"
#							do
#								sbatch scripts/OOD_tests/OOD_num_agents_test.sh $Env $N_agents $Method $coordination $heterogeneity $use_policy_pool $latent_kl $Seed $ProjectName
#							done
#						done
#					done
#				done			
#			done
#		done
#	done
#done
## #ProjectName="test"
#
## OOD Coordination Tests
#
##declare -a All_Envs=("TeamSupportEnv" "TeamTogetherEnv") #"ClutteredCompoundGoalTileCoordinationHeterogeneityEnv"
##
##declare -a All_N_agents=(10)
##
##declare -a All_Methods=("ippo" "mappo")
##
##declare -a All_coordination=(1 2 3 4 5)
##
##declare -a All_heterogeneity=(1)
##
##declare -a All_use_policy_pool=(False)
#
#declare -a All_latent_kl=(False)
#
#Seeds=(1)
#
#
#
#for Env in "${All_Envs[@]}"
#do
#	for N_agents in "${All_N_agents[@]}"
#	do
#		for Method in "${All_Methods[@]}"
#		do	
#			for coordination in "${All_coordination[@]}"
#			do
#				for heterogeneity in "${All_heterogeneity[@]}"
#				do
#					for use_policy_pool in "${All_use_policy_pool[@]}"
#					do	
#						for latent_kl in "${All_latent_kl[@]}"
#						do					
#							for Seed in "${Seeds[@]}"
#							do
#								sbatch scripts/OOD_tests/OOD_coordination_test.sh $Env $N_agents $Method $coordination $heterogeneity $use_policy_pool $latent_kl $Seed $ProjectName
#
#							done
#						done
#					done
#				done			
#			done
#		done
#	done
#done
#
#declare -a All_Methods=("saf")
#declare -a All_use_policy_pool=(False True)
#declare -a All_latent_kl=(False True)
#
#for Env in "${All_Envs[@]}"
#do
#	for N_agents in "${All_N_agents[@]}"
#	do
#		for Method in "${All_Methods[@]}"
#		do	
#			for coordination in "${All_coordination[@]}"
#			do
#				for heterogeneity in "${All_heterogeneity[@]}"
#				do
#					for use_policy_pool in "${All_use_policy_pool[@]}"
#					do	
#						for latent_kl in "${All_latent_kl[@]}"
#						do					
#							for Seed in "${Seeds[@]}"
#							do
#								sbatch scripts/OOD_tests/OOD_coordination_test.sh $Env $N_agents $Method $coordination $heterogeneity $use_policy_pool $latent_kl $Seed $ProjectName
#
#							done
#						done
#					done
#				done			
#			done
#		done
#	done
#done
#
#
# OOD HECO test
declare -a All_Envs=("TeamSupportEnv" "TeamTogetherEnv") #"ClutteredCompoundGoalTileCoordinationHeterogeneityEnv"

declare -a All_N_agents=(10)

declare -a All_Methods=("ippo" "mappo")

declare -a All_use_policy_pool=(False)

declare -a All_latent_kl=(False)

declare -a All_coordination=(1)

declare -a All_heterogeneity=(1)

#for Env in "${All_Envs[@]}"
#do
#
#	for N_agents in "${All_N_agents[@]}"
#	do
#		for Method in "${All_Methods[@]}"
#		do
#			for coordination in "${All_coordination[@]}"
#			do
#				for heterogeneity in "${All_heterogeneity[@]}"
#				do
#					for use_policy_pool in "${All_use_policy_pool[@]}"
#					do	
#						for latent_kl in "${All_latent_kl[@]}"
#						do					
#							for Seed in "${Seeds[@]}"
#							do
#								sbatch scripts/OOD_tests/OOD_heco_test.sh $Env $N_agents $Method $coordination $heterogeneity $use_policy_pool $latent_kl $Seed $ProjectName 
#							done
#						done
#					done
#				done			
#			done
#		done
#	done
#done
#
#
#declare -a All_Methods=("saf")
#declare -a All_use_policy_pool=(False True)
#declare -a All_latent_kl=(False True)
#
#for Env in "${All_Envs[@]}"
#do
#	for N_agents in "${All_N_agents[@]}"
#	do
#		for Method in "${All_Methods[@]}"
#		do
#			for coordination in "${All_coordination[@]}"
#			do
#				for heterogeneity in "${All_heterogeneity[@]}"
#				do
#					for use_policy_pool in "${All_use_policy_pool[@]}"
#					do	
#						for latent_kl in "${All_latent_kl[@]}"
#						do					
#							for Seed in "${Seeds[@]}"
#							do
#								sbatch scripts/OOD_tests/OOD_heco_test.sh $Env $N_agents $Method $coordination $heterogeneity $use_policy_pool $latent_kl $Seed $ProjectName 
#							done
#						done
#					done
#				done			
#			done
#		done
#	done
#done
#
## OOD Heterogeneity Tests
#
#declare -a All_Methods=("ippo" "mappo")
#
#declare -a All_use_policy_pool=(False)
#
#declare -a All_latent_kl=(False)
#
#declare -a All_coordination=(3)
#
#declare -a All_heterogeneity=(3)
#
#for Env in "${All_Envs[@]}"
#do
#
#	for N_agents in "${All_N_agents[@]}"
#	do
#		for Method in "${All_Methods[@]}"
#		do
#			for coordination in "${All_coordination[@]}"
#			do
#				for heterogeneity in "${All_heterogeneity[@]}"
#				do
#					for use_policy_pool in "${All_use_policy_pool[@]}"
#					do	
#						for latent_kl in "${All_latent_kl[@]}"
#						do					
#							for Seed in "${Seeds[@]}"
#							do
#								sbatch scripts/OOD_tests/OOD_heco_test.sh $Env $N_agents $Method $coordination $heterogeneity $use_policy_pool $latent_kl $Seed $ProjectName 
#							done
#						done
#					done
#				done			
#			done
#		done
#	done
#done
#
#
#declare -a All_Methods=("saf")
#declare -a All_use_policy_pool=(False True)
#declare -a All_latent_kl=(False True)
#
#for Env in "${All_Envs[@]}"
#do
#	for N_agents in "${All_N_agents[@]}"
#	do
#		for Method in "${All_Methods[@]}"
#		do
#			for coordination in "${All_coordination[@]}"
#			do
#				for heterogeneity in "${All_heterogeneity[@]}"
#				do
#					for use_policy_pool in "${All_use_policy_pool[@]}"
#					do	
#						for latent_kl in "${All_latent_kl[@]}"
#						do					
#							for Seed in "${Seeds[@]}"
#							do
#								sbatch scripts/OOD_tests/OOD_heco_test.sh $Env $N_agents $Method $coordination $heterogeneity $use_policy_pool $latent_kl $Seed $ProjectName 
#							done
#						done
#					done
#				done			
#			done
#		done
#	done
#done
#
## OOD Heterogeneity Tests
#
#declare -a All_Methods=("ippo" "mappo")
#
#declare -a All_use_policy_pool=(False)
#
#declare -a All_latent_kl=(False)
#
#declare -a All_coordination=(2)
#
#declare -a All_heterogeneity=(2)
#
#for Env in "${All_Envs[@]}"
#do
#
#	for N_agents in "${All_N_agents[@]}"
#	do
#		for Method in "${All_Methods[@]}"
#		do
#			for coordination in "${All_coordination[@]}"
#			do
#				for heterogeneity in "${All_heterogeneity[@]}"
#				do
#					for use_policy_pool in "${All_use_policy_pool[@]}"
#					do	
#						for latent_kl in "${All_latent_kl[@]}"
#						do					
#							for Seed in "${Seeds[@]}"
#							do
#								sbatch scripts/OOD_tests/OOD_heco_test.sh $Env $N_agents $Method $coordination $heterogeneity $use_policy_pool $latent_kl $Seed $ProjectName 
#							done
#						done
#					done
#				done			
#			done
#		done
#	done
#done
#
#
#declare -a All_Methods=("saf")
#declare -a All_use_policy_pool=(False True)
#declare -a All_latent_kl=(False True)
#
#for Env in "${All_Envs[@]}"
#do
#	for N_agents in "${All_N_agents[@]}"
#	do
#		for Method in "${All_Methods[@]}"
#		do
#			for coordination in "${All_coordination[@]}"
#			do
#				for heterogeneity in "${All_heterogeneity[@]}"
#				do
#					for use_policy_pool in "${All_use_policy_pool[@]}"
#					do	
#						for latent_kl in "${All_latent_kl[@]}"
#						do					
#							for Seed in "${Seeds[@]}"
#							do
#								sbatch scripts/OOD_tests/OOD_heco_test.sh $Env $N_agents $Method $coordination $heterogeneity $use_policy_pool $latent_kl $Seed $ProjectName 
#							done
#						done
#					done
#				done			
#			done
#		done
#	done
#done
#
# OOD Heterogeneity 
declare -a All_Envs=("keyfortreasure") 

declare -a All_Methods=("ippo" "mappo")

declare -a All_use_policy_pool=(False)

declare -a All_latent_kl=(False)

declare -a All_coordination=(1)

declare -a All_heterogeneity=(1 2 3)

for Env in "${All_Envs[@]}"
do

	for N_agents in "${All_N_agents[@]}"
	do
		for Method in "${All_Methods[@]}"
		do
			for coordination in "${All_coordination[@]}"
			do
				for heterogeneity in "${All_heterogeneity[@]}"
				do
					for use_policy_pool in "${All_use_policy_pool[@]}"
					do	
						for latent_kl in "${All_latent_kl[@]}"
						do					
							for Seed in "${Seeds[@]}"
							do
								sbatch scripts/OOD_tests/OOD_heterogeneity_test.sh $Env $N_agents $Method $coordination $heterogeneity $use_policy_pool $latent_kl $Seed $ProjectName 
							done
						done
					done
				done			
			done
		done
	done
done


declare -a All_Methods=("saf")
declare -a All_use_policy_pool=(False True)
declare -a All_latent_kl=(False True)

for Env in "${All_Envs[@]}"
do
	for N_agents in "${All_N_agents[@]}"
	do
		for Method in "${All_Methods[@]}"
		do
			for coordination in "${All_coordination[@]}"
			do
				for heterogeneity in "${All_heterogeneity[@]}"
				do
					for use_policy_pool in "${All_use_policy_pool[@]}"
					do	
						for latent_kl in "${All_latent_kl[@]}"
						do					
							for Seed in "${Seeds[@]}"
							do
								sbatch scripts/OOD_tests/OOD_heterogeneity_test.sh $Env $N_agents $Method $coordination $heterogeneity $use_policy_pool $latent_kl $Seed $ProjectName 
							done
						done
					done
				done			
			done
		done
	done
done
## OOD Heterogeneity 
#declare -a All_Envs=("CompoundGoalEnv") 
#
#declare -a All_N_agents=(20)
#
#declare -a All_Methods=("ippo" "mappo")
#
#declare -a All_use_policy_pool=(False)
#
#declare -a All_latent_kl=(False)
#
#declare -a All_coordination=(1)
#
#declare -a All_heterogeneity=(1 2 3)
#
#for Env in "${All_Envs[@]}"
#do
#
#	for N_agents in "${All_N_agents[@]}"
#	do
#		for Method in "${All_Methods[@]}"
#		do
#			for coordination in "${All_coordination[@]}"
#			do
#				for heterogeneity in "${All_heterogeneity[@]}"
#				do
#					for use_policy_pool in "${All_use_policy_pool[@]}"
#					do	
#						for latent_kl in "${All_latent_kl[@]}"
#						do					
#							for Seed in "${Seeds[@]}"
#							do
#								sbatch scripts/OOD_tests/OOD_heterogeneity_test_compoundenv.sh $Env $N_agents $Method $coordination $heterogeneity $use_policy_pool $latent_kl $Seed $ProjectName 
#							done
#						done
#					done
#				done			
#			done
#		done
#	done
#done
#
#
#declare -a All_Methods=("saf")
#declare -a All_use_policy_pool=(False True)
#declare -a All_latent_kl=(False True)
#
#for Env in "${All_Envs[@]}"
#do
#	for N_agents in "${All_N_agents[@]}"
#	do
#		for Method in "${All_Methods[@]}"
#		do
#			for coordination in "${All_coordination[@]}"
#			do
#				for heterogeneity in "${All_heterogeneity[@]}"
#				do
#					for use_policy_pool in "${All_use_policy_pool[@]}"
#					do	
#						for latent_kl in "${All_latent_kl[@]}"
#						do					
#							for Seed in "${Seeds[@]}"
#							do
#								sbatch scripts/OOD_tests/OOD_heterogeneity_test_compoundenv.sh $Env $N_agents $Method $coordination $heterogeneity $use_policy_pool $latent_kl $Seed $ProjectName 
#							done
#						done
#					done
#				done			
#			done
#		done
#	done
#done
#
#
#
# OOD Coordination
declare -a All_Envs=("keyfortreasure") 

declare -a All_Methods=("ippo" "mappo")

declare -a All_use_policy_pool=(False)

declare -a All_latent_kl=(False)

declare -a All_coordination=(1 2 3)

declare -a All_heterogeneity=(1)

for Env in "${All_Envs[@]}"
do

	for N_agents in "${All_N_agents[@]}"
	do
		for Method in "${All_Methods[@]}"
		do
			for coordination in "${All_coordination[@]}"
			do
				for heterogeneity in "${All_heterogeneity[@]}"
				do
					for use_policy_pool in "${All_use_policy_pool[@]}"
					do	
						for latent_kl in "${All_latent_kl[@]}"
						do					
							for Seed in "${Seeds[@]}"
							do
								sbatch scripts/OOD_tests/OOD_coordination_test.sh $Env $N_agents $Method $coordination $heterogeneity $use_policy_pool $latent_kl $Seed $ProjectName 
							done
						done
					done
				done			
			done
		done
	done
done


declare -a All_Methods=("saf")
declare -a All_use_policy_pool=(False True)
declare -a All_latent_kl=(False True)

for Env in "${All_Envs[@]}"
do
	for N_agents in "${All_N_agents[@]}"
	do
		for Method in "${All_Methods[@]}"
		do
			for coordination in "${All_coordination[@]}"
			do
				for heterogeneity in "${All_heterogeneity[@]}"
				do
					for use_policy_pool in "${All_use_policy_pool[@]}"
					do	
						for latent_kl in "${All_latent_kl[@]}"
						do					
							for Seed in "${Seeds[@]}"
							do
								sbatch scripts/OOD_tests/OOD_coordination_test.sh $Env $N_agents $Method $coordination $heterogeneity $use_policy_pool $latent_kl $Seed $ProjectName 
							done
						done
					done
				done			
			done
		done
	done
done
#
## OOD Heterogeneity 
#declare -a All_Envs=("CompoundGoalEnv") 
#
#declare -a All_N_agents=(20)
#
#declare -a All_Methods=("ippo" "mappo")
#
#declare -a All_use_policy_pool=(False)
#
#declare -a All_latent_kl=(False)
#
#declare -a All_coordination=(1)
#
#declare -a All_heterogeneity=(1 2 3)
#
#for Env in "${All_Envs[@]}"
#do
#
#	for N_agents in "${All_N_agents[@]}"
#	do
#		for Method in "${All_Methods[@]}"
#		do
#			for coordination in "${All_coordination[@]}"
#			do
#				for heterogeneity in "${All_heterogeneity[@]}"
#				do
#					for use_policy_pool in "${All_use_policy_pool[@]}"
#					do	
#						for latent_kl in "${All_latent_kl[@]}"
#						do					
#							for Seed in "${Seeds[@]}"
#							do
#								sbatch scripts/OOD_tests/OOD_coordination_test_compoundenv.sh $Env $N_agents $Method $coordination $heterogeneity $use_policy_pool $latent_kl $Seed $ProjectName 
#							done
#						done
#					done
#				done			
#			done
#		done
#	done
#done
#
#
#declare -a All_Methods=("saf")
#declare -a All_use_policy_pool=(False True)
#declare -a All_latent_kl=(False True)
#
#for Env in "${All_Envs[@]}"
#do
#	for N_agents in "${All_N_agents[@]}"
#	do
#		for Method in "${All_Methods[@]}"
#		do
#			for coordination in "${All_coordination[@]}"
#			do
#				for heterogeneity in "${All_heterogeneity[@]}"
#				do
#					for use_policy_pool in "${All_use_policy_pool[@]}"
#					do	
#						for latent_kl in "${All_latent_kl[@]}"
#						do					
#							for Seed in "${Seeds[@]}"
#							do
#								sbatch scripts/OOD_tests/OOD_coordination_test_compoundenv.sh $Env $N_agents $Method $coordination $heterogeneity $use_policy_pool $latent_kl $Seed $ProjectName 
#							done
#						done
#					done
#				done			
#			done
#		done
#	done
#done
#

# OOD N agents tests

#declare -a All_Methods=("saf")
#
#declare -a All_coordination=(2)
#
#declare -a All_heterogeneity=(1)
#
#declare -a N_agents=(5 10 15 20 25)
#
#declare -a All_use_policy_pool=(False True)
#
#declare -a All_latent_kl=(False True)
#
#Seeds=(1)
#
#
#
#for Env in "${All_Envs[@]}"
#do
#	for N_agents in "${All_N_agents[@]}"
#	do
#		for Method in "${All_Methods[@]}"
#		do
#			for coordination in "${All_coordination[@]}"
#			do
#				for heterogeneity in "${All_heterogeneity[@]}"
#				do
#					for use_policy_pool in "${All_use_policy_pool[@]}"
#					do	
#						for latent_kl in "${All_latent_kl[@]}"
#						do					
#							for Seed in "${Seeds[@]}"
#							do
#								sbatch scripts/OOD_tests/OOD_num_agents_test.sh $Env $N_agents $Method $coordination $heterogeneity $use_policy_pool $latent_kl $Seed $ProjectName 
#							done
#						done
#					done
#				done			
#			done
#		done
#	done
#done

## OOD Num treasures tests
#
#declare -a All_Envs=("TeamSupportEnv" "TeamTogetherEnv") #"ClutteredCompoundGoalTileCoordinationHeterogeneityEnv"
#
#declare -a All_N_agents=(10)
#
#declare -a All_Methods=("ippo" "mappo")
#
#declare -a All_coordination=(2)
#
#declare -a All_heterogeneity=(1)
#
#declare -a All_use_policy_pool=(False)
#
#declare -a All_latent_kl=(False)
#
#declare -a All_num_treasures=(50 75 100 125 150)
#
#Seeds=(1)
#
#
#
#for Env in "${All_Envs[@]}"
#do
#	for N_agents in "${All_N_agents[@]}"
#	do
#		for Method in "${All_Methods[@]}"
#		do	
#			for coordination in "${All_coordination[@]}"
#			do
#				for heterogeneity in "${All_heterogeneity[@]}"
#				do
#					for use_policy_pool in "${All_use_policy_pool[@]}"
#					do	
#						for latent_kl in "${All_latent_kl[@]}"
#						do					
#							for num_treasures in "${All_num_treasures[@]}"
#							do
#								for Seed in "${Seeds[@]}"
#								do
#									sbatch scripts/OOD_tests/OOD_num_treasures_test.sh $Env $N_agents $Method $coordination $heterogeneity $use_policy_pool $latent_kl $Seed $ProjectName $num_treasures
#
#								done
#							done
#						done
#					done
#				done			
#			done
#		done
#	done
#done
#
#declare -a All_Methods=("saf")
#declare -a All_use_policy_pool=(False True)
#declare -a All_latent_kl=(False True)
#
#
#for Env in "${All_Envs[@]}"
#do
#	for N_agents in "${All_N_agents[@]}"
#	do
#		for Method in "${All_Methods[@]}"
#		do	
#			for coordination in "${All_coordination[@]}"
#			do
#				for heterogeneity in "${All_heterogeneity[@]}"
#				do
#					for use_policy_pool in "${All_use_policy_pool[@]}"
#					do	
#						for latent_kl in "${All_latent_kl[@]}"
#						do					
#							for num_treasures in "${All_num_treasures[@]}"
#							do
#								for Seed in "${Seeds[@]}"
#								do
#									sbatch scripts/OOD_tests/OOD_num_treasures_test.sh $Env $N_agents $Method $coordination $heterogeneity $use_policy_pool $latent_kl $Seed $ProjectName $num_treasures
#
#								done
#							done
#						done
#					done
#				done			
#			done
#		done
#	done
#done

## OOD Grid size tests
#
#declare -a All_Envs=("TeamSupportEnv" "TeamTogetherEnv") #"ClutteredCompoundGoalTileCoordinationHeterogeneityEnv"
#
#declare -a All_N_agents=(10)
#
#declare -a All_Methods=("ippo" "mappo")
#
#declare -a All_coordination=(2)
#
#declare -a All_heterogeneity=(1)
#
#declare -a All_use_policy_pool=(False)
#
#declare -a All_latent_kl=(False)
#
#declare -a All_grid_sizes=(20 30 40 50 60)
#
#Seeds=(1)
#
#
#for Env in "${All_Envs[@]}"
#do
#	for N_agents in "${All_N_agents[@]}"
#	do
#		for Method in "${All_Methods[@]}"
#		do	
#			for coordination in "${All_coordination[@]}"
#			do
#				for heterogeneity in "${All_heterogeneity[@]}"
#				do
#					for use_policy_pool in "${All_use_policy_pool[@]}"
#					do	
#						for latent_kl in "${All_latent_kl[@]}"
#						do					
#							for grid_size in "${All_grid_size[@]}"
#							do
#								for Seed in "${Seeds[@]}"
#								do
#									sbatch scripts/OOD_tests/OOD_size_env_test.sh $Env $N_agents $Method $coordination $heterogeneity $use_policy_pool $latent_kl $Seed $ProjectName $grid_size
#
#								done
#							done
#						done
#					done
#				done			
#			done
#		done
#	done
#done
#
#
#declare -a All_Methods=("saf")
#declare -a All_use_policy_pool=(False True)
#declare -a All_latent_kl=(False True)
#
#
#for Env in "${All_Envs[@]}"
#do
#	for N_agents in "${All_N_agents[@]}"
#	do
#		for Method in "${All_Methods[@]}"
#		do	
#			for coordination in "${All_coordination[@]}"
#			do
#				for heterogeneity in "${All_heterogeneity[@]}"
#				do
#					for use_policy_pool in "${All_use_policy_pool[@]}"
#					do	
#						for latent_kl in "${All_latent_kl[@]}"
#						do					
#							for grid_size in "${All_grid_size[@]}"
#							do
#								for Seed in "${Seeds[@]}"
#								do
#									sbatch scripts/OOD_tests/OOD_size_env_test.sh $Env $N_agents $Method $coordination $heterogeneity $use_policy_pool $latent_kl $Seed $ProjectName $grid_size
#
#								done
#							done
#						done
#					done
#				done			
#			done
#		done
#	done
#done
#

#ProjectName="cg_heterogeneity_saf_new"
##ProjectName="test"
#
#
#declare -a All_Envs=("keyfortreasure") #"TeamSupportEnv" "TeamTogetherEnv" #"ClutteredCompoundGoalTileCoordinationHeterogeneityEnv"
#
#declare -a All_N_agents=(10)
#
#declare -a All_Methods=("saf")
#
#declare -a All_coordination=(1)
#
#declare -a All_heterogeneity=(1 2 3 4 5)
#
#
#
#declare -a All_use_policy_pool=(True False)
#
#
#declare -a All_latent_kl=(True False)
#
#
#Seeds=($(seq 1 1 1))
#
#
#
#for Env in "${All_Envs[@]}"
#do
#
#	for N_agents in "${All_N_agents[@]}"
#	do
#		for Method in "${All_Methods[@]}"
#		do
#			for coordination in "${All_coordination[@]}"
#			do
#				for heterogeneity in "${All_heterogeneity[@]}"
#				do
#
#					for use_policy_pool in "${All_use_policy_pool[@]}"
#					do	
#						for latent_kl in "${All_latent_kl[@]}"
#						do					
#					
#
#							for Seed in "${Seeds[@]}"
#							do
#
#
#								sbatch scripts/saf.sh $Env $N_agents $Method $coordination $heterogeneity $use_policy_pool $latent_kl $Seed $ProjectName $conda_env
#
#							done
#						done
#					done
#				done			
#			done
#		done
#	done
#done
#
#declare -a All_coordination=(2)
#
#declare -a All_heterogeneity=(1)
#
#
#for Env in "${All_Envs[@]}"
#do
#
#	for N_agents in "${All_N_agents[@]}"
#	do
#		for Method in "${All_Methods[@]}"
#		do
#			for coordination in "${All_coordination[@]}"
#			do
#				for heterogeneity in "${All_heterogeneity[@]}"
#				do
#
#					for use_policy_pool in "${All_use_policy_pool[@]}"
#					do	
#						for latent_kl in "${All_latent_kl[@]}"
#						do					
#					
#
#							for Seed in "${Seeds[@]}"
#							do
#								sbatch scripts/saf.sh $Env $N_agents $Method $coordination $heterogeneity $use_policy_pool $latent_kl $Seed $ProjectName $conda_env
#
#							done
#						done
#					done
#				done			
#			done
#		done
#	done
#done

##different baselines at different levels of coordination for marlgrid
#
#ProjectName="team_coordination_baselines_new"
#
#
#declare -a All_Envs=("keyfortreasure") #"TeamSupportEnv" "TeamTogetherEnv" #"ClutteredCompoundGoalTileCoordinationHeterogeneityEnv"
#
#
#declare -a All_N_agents=(10)
#
#declare -a All_Methods=("mappo" "ippo")
#
#
#declare -a All_coordination=(1 2 3 4 5)
#
#declare -a All_heterogeneity=(1)
#
#
#Seeds=($(seq 1 1 1))
#
#
#
#for Env in "${All_Envs[@]}"
#do
#
#	for N_agents in "${All_N_agents[@]}"
#	do
#		for Method in "${All_Methods[@]}"
#		do
#			for coordination in "${All_coordination[@]}"
#			do
#				for heterogeneity in "${All_heterogeneity[@]}"
#				do					
#
#					for Seed in "${Seeds[@]}"
#					do
#
#
#						./scripts/baselines.sh $Env $N_agents $Method $coordination $heterogeneity $Seed $ProjectName $conda_env
#
#					done
#				done			
#			done
#		done
#	done
#done
#
#
#
#
#
#
## # #different baselines at different levels of heterogeneity for marlgrid
#
#ProjectName="team_heterogeneity_baselines_new"
##ProjectName="test"
#
#
#declare -a All_Envs=("keyfortreasure") #"TeamSupportEnv" "TeamTogetherEnv"#"ClutteredCompoundGoalTileCoordinationHeterogeneityEnv"
#
#
#declare -a All_N_agents=(10)
#
#declare -a All_Methods=("ippo" "mappo")
#
#
#declare -a All_coordination=(1)
#
#
#declare -a All_heterogeneity=(1 2 3 4 5)
#
#Seeds=($(seq 1 1 1))
#
#
#
#for Env in "${All_Envs[@]}"
#do
#
#	for N_agents in "${All_N_agents[@]}"
#	do
#		for Method in "${All_Methods[@]}"
#		do
#			for coordination in "${All_coordination[@]}"
#			do
#				for heterogeneity in "${All_heterogeneity[@]}"
#				do					
#
#					for Seed in "${Seeds[@]}"
#					do
#
#
#						sbatch scripts/baselines.sh $Env $N_agents $Method $coordination $heterogeneity $Seed $ProjectName $conda_env
#
#					done
#				done			
#			done
#		done
#	done
#done






