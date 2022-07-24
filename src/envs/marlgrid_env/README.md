# MarlGrid
Gridworld for MARL experiments, based on [MiniGrid](https://github.com/maximecb/gym-minigrid).

Original MARLGrid [here](https://github.com/kandouss/marlgrid)

## Installation

In your python environment, do

```
cd marlgrid/
pip install -e .
```

## Using the GoalTile Environment

```python3
import marlgrid.envs
import gym
from penv import ParallelEnv
import warnings
import random

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

if __name__ == '__main__':
    num_procs = 64              # Running 64 parallel environments
    envs = []

    for i in range(num_procs):
        env = gym.make('Goaltile-20Agents-100Goals-v0')
        envs.append(env)

    env = ParallelEnv(envs)

    # obs is a list (length = num_procs) containing observations of each environment
    # obs of each environment is a list (length = num_agents) containing the observations of each agent (shape = (56, 56, 3))
    obs = env.reset()
    
    # Taking random step in the environments
    # actions is a list (length = num_procs) containing the actions for each environment
    # action for each environment is a list (length = num_agents) containing the int action for ench agent
    actions = [[random.randint(0, 6) for _ in range(20)] for _ in range(num_procs)]

    # Taking a step in the environments
    # r is a list (length = num_procs) of rewards for each environment
    # reward for each environment is a list (length = num_agents) contaning reward for each agent
    # d is a list of booleans (length = num_procs) indicating the end of the episode for each environment
    next_obs, r, d, _ = env.step(actions)
```

## Customizing

New environments can be added in ```marlgrid/envs/__init__.py``` by changing the relevant parameters

