"""
This is modified from https://github.com/Farama-Foundation/SuperSuit/blob/767823849865af32050fee5c6a7d2fa7d86c4973/supersuit/vector/markov_vector_wrapper.py
The modification allows the wrapper to pass in information about states in addition to the existing observations 
"""
import numpy as np
import gym.vector
from gym.vector.utils import concatenate, iterate, create_empty_array
from gym import spaces


class MarkovVectorEnv(gym.vector.VectorEnv):
    def __init__(self, par_env, black_death=False):
        """
        parameters:
            - par_env: the pettingzoo Parallel environment that will be converted to a gym vector environment
            - black_death: whether to give zero valued observations and 0 rewards when an agent is done, allowing for environments with multiple numbers of agents.
                            Is equivalent to adding the black death wrapper, but somewhat more efficient.

        The resulting object will be a valid vector environment that has a num_envs
        parameter equal to the max number of agents, will return an array of observations,
        rewards, dones, etc, and will reset environment automatically when it finishes
        """
        self.par_env = par_env
        self.metadata = par_env.metadata
        self.observation_space = par_env.observation_space(par_env.possible_agents[0])
        
        self.state_space = par_env.state_space
        self.action_space = par_env.action_space(par_env.possible_agents[0])
        if type(self.observation_space) == spaces.Dict:
            self.action_mask_space = self.observation_space['action_mask']
            self.observation_space = self.observation_space['observation']
        else:
            assert all(
                self.observation_space == par_env.observation_space(agent)
                for agent in par_env.possible_agents
            ), "observation spaces not consistent. Perhaps you should wrap with `supersuit.aec_wrappers.pad_observations`?"
            assert all(
                self.action_space == par_env.action_space(agent)
                for agent in par_env.possible_agents
            ), "action spaces not consistent. Perhaps you should wrap with `supersuit.aec_wrappers.pad_actions`?"
        self.num_envs = len(par_env.possible_agents)
        self.black_death = black_death

    def concat_obs(self, obs_dict):
        
        obs_list = []
        for i, agent in enumerate(self.par_env.possible_agents):
            if agent not in obs_dict:
                raise AssertionError(
                    "environment has agent death. Not allowed for pettingzoo_env_to_vec_env_v1 unless black_death is True"
                )
            obs_list.append(obs_dict[agent])
        return concatenate(
            self.observation_space,
            obs_list,
            create_empty_array(self.observation_space, self.num_envs),
        )
    
    def concat_state(self, state_dict):
        
        state_list = []
        for i, agent in enumerate(self.par_env.possible_agents):
            if agent not in state_dict:
                raise AssertionError(
                    "environment has agent death. Not allowed for pettingzoo_env_to_vec_env_v1 unless black_death is True"
                )
            state_list.append(state_dict[agent])

        return concatenate(
            self.state_space,
            state_list,
            create_empty_array(self.state_space, self.num_envs),
        )
    
    def concat_act_mask(self, act_mask_dict):
        if type(act_mask_dict) == type(None):
            return None
        act_mask_list = []
        for i, agent in enumerate(self.par_env.possible_agents):
            if agent not in act_mask_dict:
                raise AssertionError(
                    "environment has agent death. Not allowed for pettingzoo_env_to_vec_env_v1 unless black_death is True"
                )
            act_mask_list.append(act_mask_dict[agent])

        return concatenate(
            self.action_mask_space,
            act_mask_list,
            create_empty_array(self.action_mask_space, self.num_envs),
        )
    
    def step_async(self, actions):
        self._saved_actions = actions

    def step_wait(self):
        return self.step(self._saved_actions)

    def reset(self, seed=None):
        obs, state, act_mask = self.par_env.reset(seed=seed)
        return self.concat_obs(obs), self.concat_state(state), self.concat_act_mask(act_mask)

    def step(self, actions):
        actions = list(iterate(self.action_space, actions))

        agent_set = set(self.par_env.agents)
        act_dict = {
            agent: actions[i]
            for i, agent in enumerate(self.par_env.possible_agents)
            if agent in agent_set
        }
        observations, states, act_masks, rewards, dones, infos = self.par_env.step(act_dict)

        # adds last observation to info where user can get it
        #if all(dones.values()):
            #for agent, obs in observations.items():
                #infos[agent]["terminal_observation"] = obs
            #for agent, state in observations.items():
                #infos[agent]["terminal_state"] = state

        rews = np.array(
            [rewards.get(agent, 0) for agent in self.par_env.possible_agents],
            dtype=np.float32,
        )
        dns = np.array(
            [dones.get(agent, False) for agent in self.par_env.possible_agents],
            dtype=np.uint8,
        )
        infs = []
        for agent in self.par_env.possible_agents:
            if agent in infos:
                infs.append(infos[agent])
            else:
                infs.append(infos)

        if all(dones.values()):
            # TODO: need to check how to seed this reset()
            observations, states, act_masks = self.reset()
        else:
            observations = self.concat_obs(observations)
            states = self.concat_state(states)
            if type(act_masks) == type(None):
                act_masks = None
            act_masks = self.concat_act_mask(act_masks)
        assert (
            self.black_death or self.par_env.agents == self.par_env.possible_agents
        ), "MarkovVectorEnv does not support environments with varying numbers of active agents unless black_death is set to True"
        return observations, states, act_masks, rews, dns, infs

    def render(self, mode="human"):
        return self.par_env.render(mode)

    def close(self):
        return self.par_env.close()

    def env_is_wrapped(self, wrapper_class):
        """
        env_is_wrapped only suppors vector and gym environments
        currently, not pettingzoo environments
        """
        return [False] * self.num_envs
