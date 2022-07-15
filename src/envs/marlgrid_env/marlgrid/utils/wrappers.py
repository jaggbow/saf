# Wrappers for MARLGrid Environments

import gym
import numpy as np
import multiprocessing as mp

class PermuteObsWrapper(gym.core.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        obs_shape = self.observation_space[0].shape
        self.observation_space = [gym.spaces.Box(
            low=0,
            high=255,
            shape=(3, obs_shape[0], obs_shape[1]),
            dtype='uint8'
        ) for _ in range(self.env.num_agents)]

        self.observation_space = tuple(self.observation_space)

    def reset(self):
        obs = self.env.reset()
        obs = [obs.transpose(2, 0, 1) for obs in obs]

        return obs

    def step(self, ac):
        obs, reward, done, info = self.env.step(ac)
        obs = [obs.transpose(2, 0, 1) for obs in obs]
        return obs, reward, done, info

def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        # print(f'Received command: {cmd} and action {data}')
        if cmd == "step":
            obs, reward, done, info = env.step(data)
            if done:
                obs = env.reset()
            conn.send((obs, reward, done, info))
        elif cmd == "reset":
            obs = env.reset()
            conn.send(obs)
        else:
            raise NotImplementedError

class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = tuple(self.envs[0].observation_space)
        self.action_space = tuple(self.envs[0].action_space)

        self.locals = []
        self.processes = []
        for env in self.envs[1:]:
            local, remote = mp.Pipe()
            self.locals.append(local)
            p = mp.Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()
            self.processes.append(p)

    def reset(self):
        # print("Resetting")
        i = 0
        for local in self.locals:
            # print(f'Sending the reset command to process {i}')
            local.send(("reset", None))
            i += 1
        results = [self.envs[0].reset()] + [local.recv() for local in self.locals]
        return results

    def step(self, actions):
        # print(f"Taking actions: {actions} of type {actions.dtype}")
        i = 0
        for local, action in zip(self.locals, actions[1:]):
            # print(f"Sending action {action} to process {i}")
            local.send(("step", action))
            i += 1
        obs, reward, done, info = self.envs[0].step(actions[0])
        if done:
            obs = self.envs[0].reset()
        results = zip(*[(obs, reward, done, info)] + [local.recv() for local in self.locals])
        return results

    def render(self):
        raise NotImplementedError

    def __del__(self):
        for p in self.processes:
            p.terminate()

# class ResizeObs(gym.Wrapper):
#     # Assumes square images
#     def __init__(self, env, size=28):
#         gym.Wrapper.__init__(self, env)
#         self.size = size

#     def reset(self):
#         obs = self.env.reset()
#         if obs[0].shape[1] != self.size:
#             if obs[0].shape[1] == obs[0].shape[2]:
#                 obs = [np.resize(obs, (obs.shape[0], self.size, self.size)) for obs in obs]
#             else:
#                 obs = [np.resize(obs, (self.size, self.size, obs.shape[2])) for obs in obs]

#         return obs

#     def step(self, ac):
#         obs, reward, done, info = self.env.step(ac)
#         if obs[0].shape[1] != self.size:
#             if obs[0].shape[1] == obs[0].shape[2]:
#                 obs = [np.resize(obs, (obs.shape[0], self.size, self.size)) for obs in obs]
#             else:
#                 obs = [np.resize(obs, (self.size, self.size, obs.shape[2])) for obs in obs]

#         return obs, reward, done, info