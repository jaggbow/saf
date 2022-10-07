import gym
import marlgrid.envs
import cv2

env = gym.make("keyfortreasure-10Agents-v1")
obs = env.reset()

image = env.grid.render(tile_size=100)
cv2.imwrite('keyfortreasure-10Agents-v1.png', image[:,:,[2,1,0]])