from numpy import maximum_sctype
from ..base import MultiGridEnv

from .empty import EmptyMultiGrid
from .doorkey import DoorKeyEnv
from .cluttered import ClutteredMultiGrid
from .goalcycle import ClutteredGoalCycleEnv
from .goaltile import ClutteredGoalTileEnv
from .viz_test import VisibilityTestEnv
from .goaltile_teamtogether import TeamTogetherEnv
from .goaltile_teamsupport import TeamSupportEnv
from .goaltile_compound import CompoundGoalEnv
from .keyfortreasure import keyfortreasure
from .prisonbreak import PrisonBreakEnv
from ..agents import GridAgentInterface
from gym.envs.registration import register as gym_register

import sys
import inspect
import random

this_module = sys.modules[__name__]
registered_envs = []


def register_marl_env(
    env_name,
    env_class,
    n_agents,
    max_steps,
    grid_size,
    view_size,
    view_tile_size=8,
    view_offset=0,
    agent_color=None,
    seed=None,
    env_kwargs={},
):
    # colors = ["red", "blue", "purple", "orange", "olive", "pink", "green", "white", "cyan", "custom1", "custom2", "custom3", "custom4", "custom5", "custom6"]
    colors = ["blue" for _ in range(30)]
    assert n_agents <= len(colors)

    class RegEnv(env_class):
        def __new__(cls):
            instance = super(env_class, RegEnv).__new__(env_class)
            instance.__init__(
                agents=[
                    GridAgentInterface(
                        color=c if agent_color is None else agent_color,
                        view_size=view_size,
                        view_tile_size=view_tile_size,
                        view_offset=view_offset,
                        )
                    for c in colors[:n_agents]
                ],
                grid_size=grid_size,
                max_steps=max_steps,
                seed=seed,
                **env_kwargs,
            )
            return instance

    env_class_name = f"env_{len(registered_envs)}"
    setattr(this_module, env_class_name, RegEnv)
    registered_envs.append(env_name)
    gym_register(env_name, entry_point=f"marlgrid.envs:{env_class_name}")


def env_from_config(env_config, randomize_seed=True):
    possible_envs = {k:v for k,v in globals().items() if inspect.isclass(v) and issubclass(v, MultiGridEnv)}
    
    env_class = possible_envs[env_config['env_class']]
    
    env_kwargs = {k:v for k,v in env_config.items() if k != 'env_class'}

    if randomize_seed:
        env_kwargs['seed'] = env_kwargs.get('seed', 0) + random.randint(0, 1337*1337)
    
    return env_class(**env_kwargs)

def get_env_class(env_name):
    classes = {
        'ClutteredGoalTileEnv': ClutteredGoalTileEnv,
        'TeamTogetherEnv': TeamTogetherEnv,
        'TeamSupportEnv':TeamSupportEnv,
        "PrisonBreakEnv":PrisonBreakEnv,
        "keyfortreasure":keyfortreasure,
        'CompoundGoalEnv': CompoundGoalEnv,
    }

    return classes[env_name]


# register_marl_env(
#     "MarlGrid-1AgentCluttered15x15-v0",
#     ClutteredMultiGrid,
#     n_agents=1,
#     grid_size=11,
#     view_size=5,
#     env_kwargs={'n_clutter':30}
# )

# register_marl_env(
#     "MarlGrid-3AgentCluttered11x11-v0",
#     ClutteredMultiGrid,
#     n_agents=3,
#     grid_size=11,
#     view_size=7,
#     max_steps=20,
#     env_kwargs={'clutter_density':0.15}
# )

# register_marl_env(
#     "MarlGrid-3AgentCluttered15x15-v0",
#     ClutteredMultiGrid,
#     n_agents=3,
#     grid_size=15,
#     view_size=7,
#     env_kwargs={'clutter_density':0.15}
# )

# register_marl_env(
#     "MarlGrid-2AgentEmpty9x9-v0", EmptyMultiGrid, n_agents=2, grid_size=9, view_size=7
# )

# register_marl_env(
#     "MarlGrid-3AgentEmpty9x9-v0", EmptyMultiGrid, n_agents=3, grid_size=9, view_size=7
# )

# register_marl_env(
#     "MarlGrid-4AgentEmpty9x9-v0", EmptyMultiGrid, n_agents=4, grid_size=9, view_size=7
# )

# register_marl_env(
#     "Goalcycle-demo-two-v0", 
#     ClutteredGoalCycleEnv, 
#     n_agents=2, 
#     grid_size=7,
#     view_size=7,
#     view_tile_size=5,
#     view_offset=1,
#     max_steps=20,
#     env_kwargs={
#         'clutter_density':0.0,
#         'n_bonus_tiles': 3
#     }
# )

# register_marl_env(
#     "Goaltile-demo-solo-v0", 
#     ClutteredGoalTileEnv, 
#     n_agents=1, 
#     grid_size=13,
#     view_size=7,
#     view_tile_size=5,
#     view_offset=1,
#     env_kwargs={
#         'clutter_density':0.1,
#         'n_bonus_tiles': 3
#     }
# )

# register_marl_env(
#     "Goaltile-5Agents-45Goals-v0",
#     ClutteredGoalTileEnv,
#     n_agents=5,
#     grid_size=20,
#     view_size=7,
#     view_tile_size=8,
#     view_offset=1,
#     env_kwargs={
#         'clutter_density':0.1,
#         'n_bonus_tiles': 45,
#     }
# )

# register_marl_env(
#     "Goaltile-5Agents-45Goals-v1",
#     ClutteredGoalTileEnv,
#     n_agents=5,
#     grid_size=20,
#     view_size=7,
#     view_tile_size=8,
#     view_offset=1,
#     env_kwargs={
#         'clutter_density':0.3,
#         'n_bonus_tiles': 45,
#     }
# )

# register_marl_env(
#     "Goaltile-8Agents-45Goals-v0",
#     ClutteredGoalTileEnv,
#     n_agents=8,
#     grid_size=20,
#     view_size=7,
#     view_tile_size=8,
#     view_offset=1,
#     env_kwargs={
#         'clutter_density':0.05,
#         'n_bonus_tiles': 45,
#     }
# )

# register_marl_env(
#     "Goaltile-10Agents-50Goals-v0",
#     ClutteredGoalTileEnv,
#     n_agents=10,
#     grid_size=20,
#     view_size=7,
#     view_tile_size=8,
#     view_offset=1,
#     env_kwargs={
#         'clutter_density':0.05,
#         'n_bonus_tiles': 50,
#     }
# )

# register_marl_env(
#     "Goaltile-10Agents-50Goals-v1",
#     ClutteredGoalTileEnv,
#     n_agents=10,
#     grid_size=20,
#     view_size=7,
#     view_tile_size=8,
#     view_offset=1,
#     env_kwargs={
#         'clutter_density': 0.15,
#         'n_bonus_tiles': 50,
#     }
# )


# register_marl_env(
#     "Goaltile-20Agents-50Goals-v0",
#     ClutteredGoalTileEnv,
#     n_agents=20,
#     grid_size=20,
#     view_size=7,
#     view_tile_size=8,
#     view_offset=1,
#     env_kwargs={
#         'clutter_density':0.05,
#         'n_bonus_tiles': 50,
#     }
# )

# register_marl_env(
#     "Goaltile-20Agents-50Goals-v1",
#     ClutteredGoalTileEnv,
#     n_agents=20,
#     grid_size=20,
#     view_size=7,
#     view_tile_size=8,
#     view_offset=1,
#     env_kwargs={
#         'clutter_density': 0.15,
#         'n_bonus_tiles': 50,
#     }
# )

# register_marl_env(
#     "Goaltile-30Agents-50Goals-v0",
#     ClutteredGoalTileEnv,
#     n_agents=30,
#     grid_size=20,
#     view_size=7,
#     view_tile_size=8,
#     view_offset=1,
#     env_kwargs={
#         'clutter_density':0.05,
#         'n_bonus_tiles': 50,
#     }
# )

register_marl_env(
    "Goaltile-2Agents-100Goals-v0",
    ClutteredGoalTileEnv,
    n_agents=2,
    grid_size=30,
    max_steps=150,
    view_size=7,
    view_tile_size=8,
    view_offset=1,
    env_kwargs={
        'clutter_density':0.1,
        'n_bonus_tiles': 100,
    }
)

register_marl_env(
    "Goaltile-20Agents-100Goals-v0",
    ClutteredGoalTileEnv,
    n_agents=20,
    grid_size=30,
    max_steps=150,
    view_size=7,
    view_tile_size=8,
    view_offset=1,
    env_kwargs={
        'clutter_density':0.1,
        'n_bonus_tiles': 100,
    }
)

register_marl_env(
    "Goaltile-2Agents-100Goals-v0",
    ClutteredGoalTileEnv,
    n_agents=2,
    grid_size=30,
    max_steps=150,
    view_size=7,
    view_tile_size=1,
    view_offset=1,
    env_kwargs={
        'clutter_density':0.1,
        'n_bonus_tiles': 100,
    }
)

register_marl_env(
    "Goaltile-20Agents-100Goals-v1",
    ClutteredGoalTileEnv,
    n_agents=20,
    grid_size=30,
    max_steps=150,
    view_size=7,
    view_tile_size=8,
    view_offset=1,
    env_kwargs={
        'clutter_density':0.2,
        'n_bonus_tiles': 100,
    }
)

register_marl_env(
    "GoaltileCompound-2Agents-3Goals-v0",
    CompoundGoalEnv,
    n_agents=2,
    grid_size=8,
    max_steps=20,
    view_size=7,
    view_tile_size=8,
    view_offset=1,
    env_kwargs={
        'clutter_density': 0.1,
        'n_bonus_tiles': 3,
        'heterogeneity':1,
    }
)




register_marl_env(
    "PrisonBreakEnv-10Agents-v1",
    PrisonBreakEnv,
    n_agents=10,
    grid_size=30,
    max_steps=150,
    view_size=7,
    view_tile_size=8,
    view_offset=1,
    env_kwargs={
    }
)


register_marl_env(
    "keyfortreasure-10Agents-v1",
    keyfortreasure,
    n_agents=10,
    grid_size=30,
    max_steps=150,
    view_size=7,
    view_tile_size=8,
    view_offset=1,
    env_kwargs={
            'clutter_density':0.2,
            'n_bonus_tiles': 100,
    }
)
