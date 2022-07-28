import marlgrid.envs
import gym

import warnings
import random
import hydra
from omegaconf import DictConfig

from src.envs import get_env
from src.envs import ObstoStateWrapper, pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1, black_death_v3, PermuteObsWrapper, AddStateSpaceActMaskWrapper, ParallelEnv

import numpy as np
import torch
import cv2
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)



@hydra.main(config_path="configs/", config_name="config.yaml")
def main(cfg: DictConfig):

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")

    env_config=cfg.env

    envs = [AddStateSpaceActMaskWrapper(PermuteObsWrapper(get_env(env_config.name, env_config.family, env_config.params))) for _ in range(env_config.rollout_threads)]
    env = ParallelEnv(envs)
    global obs
    obs, state, act_masks=env.reset()

    for _ in range(1000):
        actions =[]
        for _ in range(cfg.rollout_threads):
            actions=actions+[random.randint(0, 6) for _ in range(cfg.n_agents)]   
        actions=np.array(actions)

        # Taking a step in the environments
        # r is a list (length = num_procs) of rewards for each environment
        # reward for each environment is a list (length = num_agents) contaning reward for each agent
        # d is a list of booleans (length = num_procs) indicating the end of the episode for each environment
        obs, state, act_masks, reward, done, info = env.step(actions)
        
        # print("reward")
        # print(reward[0][0])
        # print("obs")
        # print(obs.shape)
        # print("reward")
        # print(reward[0][0])

        if reward[0][0]>0:
            global obs_
            obs_=obs
            for agent in env.envs[0].agents:
                print(agent.pos)
            break



        # for i in range(2):
        #     cv2.imwrite("MARLGridExample"+str(i)+".png", torch.tensor(obs[i]).permute(1,2,0)[:,:,[2,1,0]].cpu().detach().numpy())
        #     print("saving image")


    
if __name__ == '__main__':
    main() 
    # print("obs")
    # print(obs)
    # print(torch.tensor(obs[0]).permute(1,2,0).cpu().detach().numpy())

    for i in range(2):
        cv2.imwrite("MARLGridExample_agent_"+str(i)+"reward.png", torch.tensor(obs_[i]).permute(1,2,0)[:,:,[2,1,0]].cpu().detach().numpy())
    


    print("saving image")
