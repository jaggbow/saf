from functools import partial
import pretrained
from smac.env import MultiAgentEnv, StarCraft2Env
from marlgrid.envs import register_marl_env, get_env_class
import sys
import os
import gym
from gym import ObservationWrapper, spaces
from gym.envs import registry as gym_registry
from gym.spaces import flatdim
import numpy as np
from gym.wrappers import TimeLimit as GymTimeLimit

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

if sys.platform == "linux":
    os.environ.setdefault(
        "SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII")
    )


class TimeLimit(GymTimeLimit):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        # if self.env.spec is not None:
        #     self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert (
            self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            if type(done) == bool:
                info["TimeLimit.truncated"] = not done
            else:
                info["TimeLimit.truncated"] = not all(done)
            done = len(observation) * [True]
        return observation, reward, done, info


class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation of individual agents."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)

        ma_spaces = []

        for sa_obs in env.observation_space:
            flatdim = spaces.flatdim(sa_obs)
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def observation(self, observation):
        return tuple(
            [
                spaces.flatten(obs_space, obs)
                for obs_space, obs in zip(self.env.observation_space, observation)
            ]
        )


class _GymmaWrapper(MultiAgentEnv):
    def __init__(self, key, time_limit, pretrained_wrapper, **kwargs):
        self.episode_limit = time_limit
        if "marlgrid" in key:
            # key written as marlgrid:{env_name}-{n_agents}Agents-{num_goals}Goals-v0
            env_instance_name = key.split(":")[-1]
            env_name, n_agents, num_goals, _ = env_instance_name.split('-')
            env_class = get_env_class(env_name)
            register_marl_env(
                env_instance_name,
                env_class,
                n_agents=int(n_agents[:-6]),
                grid_size=kwargs["grid_size"],
                max_steps=time_limit,
                view_size=kwargs["view_size"],
                view_tile_size=kwargs["view_tile_size"],
                view_offset=1,
                seed=kwargs["seed"],
                env_kwargs={
                    'clutter_density': kwargs["clutter_density"],
                    'n_bonus_tiles': int(num_goals[:-5]),
                    'coordination_level':kwargs["coordination"],
                    'heterogeneity':kwargs["heterogeneity"],
                }
            )
            self._env = TimeLimit(gym.make(env_instance_name), max_episode_steps=time_limit)
        else:
            self._env = TimeLimit(gym.make(f"{key}"), max_episode_steps=time_limit)
        if not kwargs["use_cnn"]:
            self._env = FlattenObservation(self._env)

        self.use_cnn = kwargs["use_cnn"]
        if pretrained_wrapper:
            self._env = getattr(pretrained, pretrained_wrapper)(self._env)

        try:
            self.n_agents = self._env.n_agents
        except AttributeError:
            self.n_agents = self._env.num_agents
        self._obs = None

        self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(
            self._env.observation_space, key=lambda x: x.shape
        )

        self._seed = kwargs["seed"]
        self._env.seed(self._seed)

    def step(self, actions):
        """ Returns reward, terminated, info """
        actions = [int(a) for a in actions]
        self._obs, reward, done, info = self._env.step(actions)
        if not self.use_cnn:
            self._obs = [
                np.pad(
                    o,
                    (0, self.longest_observation_space.shape[0] - len(o)),
                    "constant",
                    constant_values=0,
                )
                for o in self._obs
            ]
        if type(done) == bool:
            return float(sum(reward)), done, {}
        else:
            return float(sum(reward)), all(done), {}

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self._obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        if self.use_cnn:
            return self.longest_observation_space.shape
        return flatdim(self.longest_observation_space)

    def get_state(self):
        if self.use_cnn:
            return np.concatenate(self._obs, axis=-1).astype(np.float32)
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """ Returns the shape of the state"""
        if self.use_cnn:
            shape = self.longest_observation_space.shape
            return shape[:-1]+(shape[-1]*self.n_agents,)
        return self.n_agents * flatdim(self.longest_observation_space)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        valid = flatdim(self._env.action_space[agent_id]) * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))
        return valid + invalid

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return flatdim(self.longest_action_space)

    def reset(self):
        """ Returns initial observations and states"""
        self._obs = self._env.reset()
        if not self.use_cnn:
            self._obs = [
                np.pad(
                    o,
                    (0, self.longest_observation_space.shape[0] - len(o)),
                    "constant",
                    constant_values=0,
                )
                for o in self._obs
            ]
        return self.get_obs(), self.get_state()

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def seed(self):
        return self._env.seed

    def save_replay(self):
        pass

    def get_stats(self):
        return {}


REGISTRY["gymma"] = partial(env_fn, env=_GymmaWrapper)
