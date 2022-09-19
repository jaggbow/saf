from marlgrid.envs import register_marl_env, get_env_class
import gym
import cv2
import numpy as np

env_class = get_env_class('CompoundGoalEnv')
np.random.seed(4)

register_marl_env(
    'CompoundGoalEnvironment',
    env_class,
    n_agents=10,
    grid_size=7,
    max_steps=100,
    view_size=7,
    view_tile_size=8,
    view_offset=1,
    seed=4,
    env_kwargs={
        'clutter_density': 0.1,
        'n_bonus_tiles': 2,
        'heterogeneity': 1,
        'coordination_level': 3,
    }
)

env = gym.make('CompoundGoalEnvironment')

obs = env.reset()
img = env.grid.render(tile_size=100)
cv2.imwrite('compoundgoal.png', img[:,:,[2,1,0]])
print(env.grid)
print(f'Agents are: {env.agents}')
agent_pos = [agent.pos for agent in env.agents]
print(agent_pos)

actions = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 2, 2, 2], [0, 2, 2, 2], [0, 2, 2, 2], [2, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 1], [1, 1, 1, 2], [2, 3, 3, 2], [2, 2, 2, 3], [2, 2, 2, 3], [2, 2, 2, 3]]

for i, action in enumerate(actions):
    _, r, _, _ = env.step(action)
    print(f'Reward is: {r}')
    img = env.grid.render(tile_size=100)
    cv2.imwrite(f'compoundgoal_{i}.png', img[:,:,[2,1,0]])

print(env.grid)
