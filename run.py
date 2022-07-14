import random
import os

import comet_ml
import supersuit as ss
import numpy as np

from gym import spaces

import torch
from torch.nn import Module

import hydra
from omegaconf import DictConfig

from src.envs import get_env
from src.envs import ObstoStateWrapper, pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1, black_death_v3
from src.replay_buffer import ReplayBuffer


def make_train_env(env_config):
    
    env_class = get_env(env_config.name, env_config.family)
    env = env_class.parallel_env(**env_config.params)
    
    if env_config.continuous_action:
        env = ss.clip_actions_v0(env)
    if env_config.family != 'starcraft':
        env = ss.pad_observations_v0(env)
        env = ss.pad_action_space_v0(env)
    else:
        env = black_death_v3(env)
    env = ObstoStateWrapper(env)
    if env_config.family == 'starcraft':
        env = pettingzoo_env_to_vec_env_v1(env, black_death=True)
    else:
        env = pettingzoo_env_to_vec_env_v1(env, black_death=False)
    env = concat_vec_envs_v1(env, env_config.rollout_threads, num_cpus=1, base_class='gym')

    return env

def make_eval_env(env_config):
    
    env_class = get_env(env_config.name, env_config.family)
    env = env_class.parallel_env(**env_config.params)
    
    if env_config.continuous_action:
        env = ss.clip_actions_v0(env)
    if env_config.family != 'starcraft':
        env = ss.pad_observations_v0(env)
        env = ss.pad_action_space_v0(env)
    else:
        env = black_death_v3(env)
    env = ObstoStateWrapper(env)

    return env

@hydra.main(config_path="configs/", config_name="config.yaml")
def main(cfg: DictConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")

    train_envs = make_train_env(cfg.env)
    eval_env = make_eval_env(cfg.env)

    if isinstance(train_envs.observation_space, spaces.Dict):
        observation_space = train_envs.observation_space['observation']
    else:
        observation_space = train_envs.observation_space

    policy = hydra.utils.instantiate(
        cfg.policy, 
        observation_space=observation_space, 
        action_space=train_envs.action_space, 
        state_space=train_envs.state_space, 
        params=cfg.policy.params)
    policy = policy.to(device)
    buffer = ReplayBuffer(observation_space, train_envs.action_space, train_envs.state_space, cfg.buffer, device)
    runner = hydra.utils.instantiate(
        cfg.runner,
        train_env=train_envs,
        eval_env=eval_env,
        env_family=cfg.env.family,
        policy=policy, 
        buffer=buffer, 
        params=cfg.runner.params, 
        device=device)
    
    if not cfg.test_mode:
        runner.run()
    mean_rewards, std_rewards, mean_wins, std_wins = runner.evaluate()
    print(f"Eval Rewards: {mean_rewards} +- {std_rewards} | Eval Win Rate: {mean_wins} +- {std_wins}")
    train_envs.close()
    eval_env.close()

if __name__ == "__main__":
    main()
    