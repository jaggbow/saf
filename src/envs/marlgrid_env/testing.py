import marlgrid.envs
import gym
from penv import ParallelEnv
import warnings
import random

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

if __name__ == '__main__':
    num_procs = 64              # Running 64 parallel environments
    envs = []

    for i in range(num_procs):
        env = gym.make("CoordinationGoaltile-5Agents-20Goals-2coordination-v1")
        envs.append(env)

    env = ParallelEnv(envs)

    # obs is a list (length = num_procs) containing observations of each environment
    # obs of each environment is a list (length = num_agents) containing the observations of each agent (shape = (56, 56, 3))
    obs = env.reset()
    



    # Taking random step in the environments
    # actions is a list (length = num_procs) containing the actions for each environment
    # action for each environment is a list (length = num_agents) containing the int action for ench agent
    N_agents=5
    r=[[0 for i in range(N_agents)]]
    while r[0][0]==0:
        actions = [[random.randint(0, 6) for _ in range(N_agents)] for _ in range(num_procs)]

        # Taking a step in the environments
        # r is a list (length = num_procs) of rewards for each environment
        # reward for each environment is a list (length = num_agents) contaning reward for each agent
        # d is a list of booleans (length = num_procs) indicating the end of the episode for each environment
        next_obs, r, d, _ = env.step(actions)
        print("r")
        print(r[0].shape)
        #print(type(r))
        if r[0][0]>0:
            print("r[0][0]")
            print(r[0][0])
            import cv2
            cv2.imwrite("images/MARLGridExample.png", next_obs[0][0])
            print("saving image")

            for agent in env.envs[0].agents:
                print(agent.pos)



    print("next_obs")
    print(len(next_obs))