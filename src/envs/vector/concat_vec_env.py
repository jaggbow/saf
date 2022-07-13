from attr import has
import numpy as np
from .single_vec_env import SingleVecEnv
import gym.vector
from gym.vector.utils import concatenate, iterate, create_empty_array
from gym.spaces import Discrete


def transpose(ll):
    return [[ll[i][j] for i in range(len(ll))] for j in range(len(ll[0]))]


@iterate.register(Discrete)
def iterate_discrete(space, items):
    try:
        return iter(items)
    except TypeError:
        raise TypeError(f"Unable to iterate over the following elements: {items}")


class ConcatVecEnv(gym.vector.VectorEnv):
    def __init__(self, vec_env_fns, obs_space=None, state_space=None, act_space=None):
        self.vec_envs = vec_envs = [vec_env_fn() for vec_env_fn in vec_env_fns]
        for i in range(len(vec_envs)):
            if not hasattr(vec_envs[i], "num_envs"):
                vec_envs[i] = SingleVecEnv([lambda: vec_envs[i]])
        self.metadata = self.vec_envs[0].metadata
        self.observation_space = vec_envs[0].observation_space
        if hasattr(vec_envs[0], 'action_mask_space'):
            self.action_mask_space = vec_envs[0].action_mask_space
        self.state_space = vec_envs[0].state_space
        self.action_space = vec_envs[0].action_space
        tot_num_envs = sum(env.num_envs for env in vec_envs)
        self.num_envs = tot_num_envs

    def reset(self, seed=None):
        _res_ls = []
        _res_states = []
        _res_act_mask = []

        # TODO: match the style of seeding of gym.vector.AsyncVectorEnv
        if seed is not None:
            for i in range(len(self.vec_envs)):
                _res, _state, _act_mask = self.vec_envs[i].reset(seed=seed + i)
                _res_ls.append(_res)
                _res_states.append(_state)
                _res_act_mask.append(_act_mask)
        else:
            for vec_env in self.vec_envs:
                _res, _state, _act_mask = vec_env.reset(seed=None)
                _res_ls.append(_res)
                _res_states.append(_state)
                _res_act_mask.append(_act_mask)
        
        if type(_res_act_mask[0]) == type(None):
            _res_act_mask = None
        return self.concat_obs(_res_ls), self.concat_states(_res_states), self.concat_act_masks(_res_act_mask)

    def concat_obs(self, observations):
        return concatenate(
            self.observation_space,
            [
                item
                for obs in observations
                for item in iterate(self.observation_space, obs)
            ],
            create_empty_array(self.observation_space, n=self.num_envs),
        )
    
    def concat_states(self, states):
        return concatenate(
            self.state_space,
            [
                item
                for state in states
                for item in iterate(self.state_space, state)
            ],
            create_empty_array(self.state_space, n=self.num_envs),
        )

    def concat_act_masks(self, act_masks):

        if type(act_masks) == type(None):
            return None
        return concatenate(
            self.action_mask_space,
            [
                item
                for act_mask in act_masks
                for item in iterate(self.action_mask_space, act_mask)
            ],
            create_empty_array(self.action_mask_space, n=self.num_envs),
        )

    def concatenate_actions(self, actions, n_actions):
        return concatenate(
            self.action_space,
            actions,
            create_empty_array(self.action_space, n=n_actions),
        )

    def step_async(self, actions):
        self._saved_actions = actions

    def step_wait(self):
        return self.step(self._saved_actions)

    def step(self, actions):
        data = []
        idx = 0
        actions = list(iterate(self.action_space, actions))
        for venv in self.vec_envs:
            data.append(
                venv.step(
                    self.concatenate_actions(
                        actions[idx : idx + venv.num_envs], venv.num_envs
                    )
                )
            )
            idx += venv.num_envs
        observations, states, act_masks, rewards, dones, infos = transpose(data)
        observations = self.concat_obs(observations)
        states = self.concat_states(states)
        if type(act_masks[0]) == type(None):
            act_masks = None
        act_masks = self.concat_act_masks(act_masks)
        rewards = np.concatenate(rewards, axis=0)
        dones = np.concatenate(dones, axis=0)
        infos = sum(infos, [])
        return observations, states, act_masks, rewards, dones, infos

    def render(self, mode="human"):
        return self.vec_envs[0].render(mode)

    def close(self):
        for vec_env in self.vec_envs:
            vec_env.close()

    def env_is_wrapped(self, wrapper_class):
        return sum(
            [sub_venv.env_is_wrapped(wrapper_class) for sub_venv in self.vec_envs], []
        )
