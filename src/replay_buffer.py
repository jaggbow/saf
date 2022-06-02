import numpy as np

import torch

from src.utils import *

class ReplayBuffer:
    def __init__(self, observation_space, params, device):
        
        self.n_agents = params.n_agents
        self.rollout_threads = params.rollout_threads
        self.env_steps = params.env_steps
        self.obs_shape = get_obs_shape(observation_space)
        self.device = device

        self.obs = torch.zeros((self.env_steps, self.rollout_threads, self.n_agents)+self.obs_shape).float().to(device)
        self.actions = torch.zeros((self.env_steps, self.rollout_threads, self.n_agents)).float().to(device)
        self.logprobs = torch.zeros((self.env_steps, self.rollout_threads, self.n_agents)).float().to(device)
        self.rewards = torch.zeros((self.env_steps, self.rollout_threads, self.n_agents)).float().to(device)
        self.values = torch.zeros((self.env_steps, self.rollout_threads, self.n_agents)).float().to(device)
        self.dones = torch.zeros((self.env_steps, self.rollout_threads, self.n_agents)).float().to(device)


    def insert(
        self, 
        obs: torch.Tensor,
        actions: torch.Tensor,
        logprobs: torch.Tensor,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        step):

        self.obs[step] = obs
        self.actions[step] = actions
        self.logprobs[step] = logprobs
        self.rewards[step] = rewards
        self.values[step] = values
        self.dones[step] = dones