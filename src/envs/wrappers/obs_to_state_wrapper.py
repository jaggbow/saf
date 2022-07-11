import numpy as np
from pettingzoo.utils import BaseParallelWraper
from gym import spaces

class ObstoStateWrapper(BaseParallelWraper):
    """
    This wrapper creates states for non-state environments by concatenating observations from all agents.
    For environments that not natively support states, this wrapper only works if all agents have the same observation space
    """

    def __init__(self, env):
        super().__init__(env)
        if hasattr(super(), 'state_space'):
            # Environment has state
            self.state_space = super().state_space
        else:
            obs_shape = None
            uneven_obs = False
            for agent in env.observation_spaces:
                if not obs_shape:
                    obs_shape = env.observation_spaces[agent].shape
                else:
                    if env.observation_spaces[agent].shape != obs_shape:
                        uneven_obs = True
                        break
            assert uneven_obs == False, "Not all observation_shapes are the same. We currently do not support this type of environments"
            assert all([env.observation_space(agent).__class__.__name__ == 'Box' for agent in env.possible_agents]), f"We only support Box observations at the moment"
            
            obs_space = env.observation_spaces[env.possible_agents[0]]
            num_agents = len(env.possible_agents)
            self.state_space = spaces.Box(
                                low = np.concatenate([obs_space.low]*num_agents, axis=-1), 
                                high = np.concatenate([obs_space.high]*num_agents, axis=-1),
                                dtype = obs_space.dtype)
    
    def reset(self, seed=None):
        obs = super().reset(seed=seed)
        if hasattr(super(), 'state_space'):
            # Environment has state
            _state = super().state()
        else:
            # Concatenate agents observation to form one state
            _state = []
            for agent in obs:
                _state.append(obs[agent])
            _state = np.concatenate(_state, axis=-1)

        state = {}
        for agent in obs:
            state[agent] = _state
        
        return obs, state

    def step(self, all_actions):
        obs, reward, done, info = super().step(all_actions)
        if hasattr(super(), 'state_space'):
            # Environment has state
            _state = super().state()
        else:
            # Concatenate agents observation to form one state
            _state = []
            for agent in obs:
                _state.append(obs[agent])
            _state = np.concatenate(_state, axis=-1)
        
        state = {}
        for agent in obs:
            state[agent] = _state
        
        return obs, state, reward, done, info

    def __str__(self):
        return str(self.env)