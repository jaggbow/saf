#!/bin/bash






########different version of saf on marlgrid 

ProjectName="team_coordination_saf2"


declare -a All_Envs=("PrisonBreakEnv") #"TeamTogetherEnv" "TeamSupportEnv"
declare -a All_N_agents=(10)

declare -a All_Methods=("saf")

declare -a All_coordination=(1 2 3 4 5)

declare -a All_heterogeneity=(1)

declare -a All_use_policy_pool=(True False)

declare -a All_latent_kl=(True False)


Seeds=($(seq 1 1 1))



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


								sbatch scripts/marlgrid/saf.sh $Env $N_agents $Method $coordination $heterogeneity $use_policy_pool $latent_kl $Seed $ProjectName

							done
						done
					done
				done			
			done
		done
	done
done


ProjectName="team_heterogeneity_saf2"
#ProjectName="test"


declare -a All_Envs=("PrisonBreakEnv") #"TeamSupportEnv" "TeamTogetherEnv" #"ClutteredCompoundGoalTileCoordinationHeterogeneityEnv"

declare -a All_N_agents=(10)

declare -a All_Methods=("saf")

declare -a All_coordination=(1)

declare -a All_heterogeneity=(1 2 3 4 5)



declare -a All_use_policy_pool=(True False)


declare -a All_latent_kl=(True False)


Seeds=($(seq 1 1 1))



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


								sbatch scripts/marlgrid/saf.sh $Env $N_agents $Method $coordination $heterogeneity $use_policy_pool $latent_kl $Seed $ProjectName

							done
						done
					done
				done			
			done
		done
	done
done




# ##different baselines at different levels of coordination for marlgrid

ProjectName="team_coordination_baselines2"
#ProjectName="test"


declare -a All_Envs=("PrisonBreakEnv") #"TeamSupportEnv" "TeamTogetherEnv" #"ClutteredCompoundGoalTileCoordinationHeterogeneityEnv"


declare -a All_N_agents=(10)

declare -a All_Methods=("mappo" "ippo")


declare -a All_coordination=(1 2 3 4 5)

declare -a All_heterogeneity=(1)


Seeds=($(seq 1 1 1))



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

					for Seed in "${Seeds[@]}"
					do


						sbatch scripts/marlgrid/baselines.sh $Env $N_agents $Method $coordination $heterogeneity $Seed $ProjectName

					done
				done			
			done
		done
	done
done






# # #different baselines at different levels of heterogeneity for marlgrid

ProjectName="team_heterogeneity_baselines2"
#ProjectName="test"


declare -a All_Envs=("PrisonBreakEnv") #"TeamSupportEnv" "TeamTogetherEnv"#"ClutteredCompoundGoalTileCoordinationHeterogeneityEnv"


declare -a All_N_agents=(10)

declare -a All_Methods=("ippo" "mappo")


declare -a All_coordination=(1)


declare -a All_heterogeneity=(1 2 3 4 5)

Seeds=($(seq 1 1 1))



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

					for Seed in "${Seeds[@]}"
					do


						sbatch scripts/marlgrid/baselines.sh $Env $N_agents $Method $coordination $heterogeneity $Seed $ProjectName

					done
				done			
			done
		done
	done
done






