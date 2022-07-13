from .wrappers.obs_to_state_wrapper import ObstoStateWrapper
from .vector.vector_constructors import (
    gym_vec_env_v0,
    concat_vec_envs_v1,
    pettingzoo_env_to_vec_env_v1,
)


def get_env(env_name, family):
    if family == 'mpe':
        from .mpe import ENVS
        
        env = ENVS[env_name]
        return env
    elif family == 'starcraft':
        from src.envs.sc_pettingzoo import StarCraft2PZEnv as sc2

        env = sc2
        return env
    elif family == 'sisl':
        from .sisl import ENVS
        
        env = ENVS[env_name]
        return env
    else:
        raise "Unrecognized family name, please pick a family in [mpe, sisl, starcraft]"