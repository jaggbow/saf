# saf

This is the version with different coordination levels and different heterogeneity levels in marlgrid envionment

coordination level is defined as the number of agents required to be in the same position to pick up the treasure

heterogeneity level is defined as the number of distinct zones in the envionrment, where the results of the same aciton are different in
different zones ( eg. action 3 means "move forward" in zone 1 and means "turn left" in zone 3). In able to help the agent to learn
we give different color to the agents when they are in different zones.


before running experiment, install the marlgrid environment by
 cd src/envs/marlgrid_env/
 and 
 pip install -e .
 
 change the comet API key to your own key:https://github.com/jaggbow/saf/blob/dfaf9b000a3178d0fac8d3920c691d500da51b44/src/runner.py#L47
 
 
 
 The code to submit job to mila cluster is in Jobs_master.sh file
 
