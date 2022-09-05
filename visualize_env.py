import gym
import marlgrid.envs
import cv2

env = gym.make("PrisonBreakEnv-10Agents-v1")
obs = env.reset()

image = env.grid.render(tile_size=100)
cv2.imwrite('PrisonBreakEnv-10Agents-v1.png', image[:,:,[2,1,0]])