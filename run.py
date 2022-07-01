import random

import comet_ml
import supersuit as ss
import numpy as np

import torch
from torch.nn import Module

import hydra
from omegaconf import DictConfig

from src.envs import get_env
from src.replay_buffer import ReplayBuffer
from src.runner import PGRunner


def make_env(env_config):
    
    env_class = get_env(env_config.name, env_config.family)
    env = env_class.parallel_env(**env_config.params)
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, env_config.rollout_threads, num_cpus=1, base_class='gym')

    return env

@hydra.main(config_path="configs/", config_name="config.yaml")
def main(cfg: DictConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")

    envs = make_env(cfg.env)

    policy = hydra.utils.instantiate(
        cfg.policy, 
        observation_space=envs.observation_space, 
        action_space=envs.action_space, 
        params=cfg.policy.params)
    policy = policy.to(device)
    buffer = ReplayBuffer(envs.observation_space, envs.action_space, cfg.buffer, device)
    runner = hydra.utils.instantiate(
        cfg.runner,
        env=envs, 
        policy=policy, 
        buffer=buffer, 
        params=cfg.runner.params, 
        device=device)
    runner.run()

    mean_rewards, std_rewards = runner.evaluate()
    print(f"Eval Rewards: {mean_rewards} +- {std_rewards}")
    envs.close()
if __name__ == "__main__":
    
    main()
    