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

class CooperativeRewardsWrapper(gym.core.Wrapper):
    # Wrapper to distribute the total rewards collected by all the agents equally among all the agents
    # This can promote cooperative behaviour among the agents
    def __init__(self, env):
        super().__init__(env)

    def step(self, actions):
        obs, reward, done, info = self.env.step(actions)
        avg_reward = sum(reward) / len(reward)
        reward = [avg_reward for _ in range(self.env.num_agents)]

        return obs, reward, done, info

class AddStateSpaceActMaskWrapper(gym.core.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space[0].shape
        obs_flatten_shape = obs_shape[0] * obs_shape[1] * obs_shape[2]

        self.state_space = [gym.spaces.Box(
            low=0,
            high=255,
            shape=(obs_flatten_shape * self.num_agents,),
            dtype='uint8',
        ) for _ in range(self.env.num_agents)]

        self.state_space = tuple(self.state_space)

    def reset(self):
        obs = self.env.reset()
        state = [agent_obs.flatten() for agent_obs in obs]
        state = np.concatenate(state)
        state = [state for _ in range(self.env.num_agents)]

        return obs, state, None
    
    def step(self, actions):
        obs, reward, done, info = self.env.step(actions)
        state = [agent_obs.flatten() for agent_obs in obs]
        state = np.concatenate(state)
        state = [state for _ in range(self.env.num_agents)]

        return obs, state, None, reward, done, info

def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        # print(f'Received command: {cmd} and action {data}')
        if cmd == "step":
            obs, state, act_mask, reward, done, info = env.step(data)
            if done:
                obs, state, act_mask = env.reset()
            conn.send((obs, state, act_mask, reward, done, info))
        elif cmd == "reset":
            obs, state, act_mask = env.reset()
            conn.send((obs, state, act_mask))
        else:
            raise NotImplementedError

class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.num_agents = self.envs[0].num_agents
        self.observation_space = tuple(self.envs[0].observation_space)
        self.action_space = tuple(self.envs[0].action_space)
        self.state_space = tuple(self.envs[0].state_space)

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
        
        obs, state, act_mask = self.envs[0].reset()
        results = zip(*[(obs, state, act_mask)] + [local.recv() for local in self.locals])

        obs, state, act_mask = results

        obs = np.array(obs).reshape((-1,)+self.envs[0].observation_space[0].shape)
        state = np.array(state).reshape((-1,)+self.envs[0].state_space[0].shape)

        return obs, state, None

    def step(self, actions):
        actions = actions.reshape(-1, self.num_agents).tolist()

        i = 0
        for local, action in zip(self.locals, actions[1:]):
            # print(f'Shape of actions to be communicated: {i, len(action)}')
            # print(f'action: {action}')
            local.send(("step", action))
            i += 1
        obs, state, act_mask, reward, done, info = self.envs[0].step(actions[0])
        if done:
            obs, state, act_mask = self.envs[0].reset()
        results = zip(*[(obs, state, act_mask, reward, done, info)] + [local.recv() for local in self.locals])
        
        obs, state, act_mask, reward, done, info = results

        obs = np.array(obs).reshape((-1,)+self.envs[0].observation_space[0].shape)
        state = np.array(state).reshape((-1,)+self.envs[0].state_space[0].shape)
        reward = np.array(reward)
        done = np.expand_dims(np.array(done), axis=1).repeat(self.envs[0].num_agents, axis=1)   
        
        return obs, state, None, reward, done, info

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