#different baselines at different levels of coordination for marlgrid



ProjectName="testing"

declare -a All_Envs=("TeamTogetherEnv") #"keyfortreasure" "TeamTogetherEnv" "TeamSupportEnv"
declare -a All_N_agents=(2)
declare -a All_Methods=("qplex_pp")
declare -a All_coordination=(1)
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
						./scripts/baselines.sh $Env $Method $coordination $heterogeneity $Seed
					done
				done			
			done
		done
	done
done