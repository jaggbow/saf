from .wrappers.obs_to_state_wrapper import ObstoStateWrapper
from .wrappers.black_death import black_death_v3
from .vector.vector_constructors import (
    gym_vec_env_v0,
    concat_vec_envs_v1,
    pettingzoo_env_to_vec_env_v1,
)
from marlgrid.utils.wrappers import PermuteObsWrapper, ParallelEnv, AddStateSpaceActMaskWrapper, CooperativeRewardsWrapper

def get_env(env_name, family, params):
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
    elif family == 'marlgrid':
        from marlgrid.envs import register_marl_env, get_env_class
        import gym

        env_instance_name = f'{env_name[9:-3]}-{params.N}Agents-{params.num_goals}Goals-v0'
        env_class = get_env_class(env_name)

        if env_name=="TeamTogetherEnv":
            
            register_marl_env(
                env_instance_name,
                env_class,
                n_agents=params.N,
                grid_size=params.grid_size,
                max_steps=params.max_steps,
                view_size=params.view_size,
                view_tile_size=params.view_tile_size,
                view_offset=1,
                env_kwargs={
                    'clutter_density': params.clutter_density,
                    'n_bonus_tiles': params.num_goals,
                    'coordination_level':params.coordination,
                    'heterogeneity':params.heterogeneity,
                }
            )

        elif env_name=="TeamSupportEnv":
            
            register_marl_env(
                env_instance_name,
                env_class,
                n_agents=params.N,
                grid_size=params.grid_size,
                max_steps=params.max_steps,
                view_size=params.view_size,
                view_tile_size=params.view_tile_size,
                view_offset=1,
                env_kwargs={
                    'clutter_density': params.clutter_density,
                    'n_bonus_tiles': params.num_goals,
                    'coordination_level':params.coordination,
                    'heterogeneity':params.heterogeneity,
                }
            )
        elif env_name == "ClutteredCompoundGoalTileCoordinationHeterogeneityEnv":
            register_marl_env(
                env_instance_name,
                env_class,
                n_agents=params.N,
                grid_size=params.grid_size,
                max_steps=params.max_steps,
                view_size=params.view_size,
                view_tile_size=params.view_tile_size,
                view_offset=1,
                env_kwargs={
                    'clutter_density': params.clutter_density,
                    'n_bonus_tiles': params.num_goals,
                    'heterogeneity': params.heterogeneity,
                }
            )
        else:
                register_marl_env(
                env_instance_name,
                env_class,
                n_agents=params.N,
                grid_size=params.grid_size,
                max_steps=params.max_steps,
                view_size=params.view_size,
                view_tile_size=params.view_tile_size,
                view_offset=1,
                env_kwargs={
                    'clutter_density': params.clutter_density,
                    'n_bonus_tiles': params.num_goals,
                }
            )
        env = gym.make(env_instance_name)

        return env
    else:
        raise "Unrecognized family name, please pick a family in [mpe, sisl, starcraft]"