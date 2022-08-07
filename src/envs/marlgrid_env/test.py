import marlgrid.envs
import gym
import cv2
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


env = gym.make("GoaltileCompound-2Agents-3Goals-v0")
obs = env.reset()
full_env = env.grid.render(tile_size=10)
cv2.imwrite('full_env_init.png', full_env)


# cv2.imwrite('full_env.png', full_env)
cv2.imwrite('obs0.png', obs[0])
cv2.imwrite('obs1.png', obs[1])

actions = [[0, 0], [0, 0], [2, 2], [2, 2], [1, 2], [2, 1], [2, 2], [3, 2], [2, 2], [2, 2], [1, 1], [2, 2], [2, 0], [2, 2], [3, 1], [2, 2], [2, 2], [1, 1], [2, 2], [2, 2]]

for i, action in enumerate(actions):
    print(f'Step {i}:')
    obs, reward, done, _ = env.step(action)
    print(f'Reward: {reward}')
    # cv2.imwrite(f'obs_{i}0.png', obs[0])
    # cv2.imwrite(f'obs_{i}1.png', obs[1])
    full_env = env.grid.render(tile_size=10)
    cv2.imwrite(f'full_env_{i}.png', full_env)