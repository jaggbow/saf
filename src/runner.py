import numpy as np
import os
from os.path import expanduser, expandvars
import time
from datetime import datetime
from pathlib import Path

from gym import spaces

from tqdm import tqdm
import comet_ml

import torch

class PGRunner:
    def __init__(self, train_env, eval_env, env_family, policy, buffer, params, device):

        self.total_timesteps = params.total_timesteps
        self.batch_size = params.rollout_threads * params.env_steps
        self.rollout_threads = params.rollout_threads
        self.n_agents = params.n_agents
        self.env_steps = params.env_steps
        self.lr_decay = params.lr_decay
        self.env_family = env_family
        self.eval_episodes = params.eval_episodes
        self.use_comet = True if params.comet else False
        self.checkpoint_dir = params.checkpoint_dir
        self.save_dir = Path(expandvars(expanduser(str(params.save_dir)))).resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(train_env.observation_space, spaces.Dict):
            self.observation_space = train_env.observation_space['observation']
        elif isinstance(train_env.observation_space, tuple):
            self.observation_space = train_env.observation_space[0]
        else:
            self.observation_space = train_env.observation_space

        if isinstance(train_env.action_space, tuple):
            self.action_space = train_env.action_space[0]
        else:
            self.action_space = train_env.action_space
        
        if self.use_comet:
            self.exp = comet_ml.Experiment(api_key="AIxlnGNX5bfAXGPOMAWbAymIz", project_name=params.comet.project_name)
            self.exp.set_name(f"{policy.__class__.__name__}_{os.environ['SLURM_JOB_ID']}")

        self.train_env = train_env
        self.eval_env = eval_env
        self.buffer = buffer
        self.policy = policy
        self.device = device

        if self.checkpoint_dir:
            print("Resuming training from", self.checkpoint_dir)
            self.load_checkpoints(self.checkpoint_dir)
            
    def env_reset(self):
        '''
        Resets the environment.
        Returns:
        obs: [rollout_threads, n_agents, obs_shape]
        '''
        obs, state, act_masks = self.train_env.reset()
        obs = torch.from_numpy(obs).to(self.device) # [rollout_threads*n_agents, obs_shape]
        obs = obs.reshape((-1, self.n_agents)+obs.shape[1:]) # [rollout_threads, n_agents, obs_shape]

        state = torch.from_numpy(state).to(self.device) # [rollout_threads*n_agents, state_shape]
        state = state.reshape((-1, self.n_agents)+state.shape[1:]) # [rollout_threads, n_agents, state_shape]

        if type(act_masks) != type(None):
            act_masks = torch.from_numpy(act_masks).to(self.device) # [rollout_threads*n_agents, action_shape]
            act_masks = act_masks.reshape((-1, self.n_agents)+act_masks.shape[1:]) # [rollout_threads, n_agents, action_shape]

        return obs, state, act_masks

    def env_step(self, action):
        '''
        Does a step in the defined environment using action.
        Args:
            action: [rollout_threads, n_agents] for Discrete type and [rollout_threads, n_agents, action_dim] for Box type
        '''
        
        print(f'\n-------------------------------------\n')
        print(f'Shape of action into env_step(): {action.shape}')
        if self.action_space.__class__.__name__ == 'Box':
            action_ = action.reshape(-1, action.shape[-1]).cpu().numpy()
        elif self.action_space.__class__.__name__ == 'Discrete':
            action_ = action.reshape(-1).cpu().numpy()
        else:
            raise NotImplementedError

        print(f'Shape of action in env_step() after reshaping: {action_.shape}')
        
        obs, state, act_masks, reward, done, info = self.train_env.step(action_)
        print(f'Shape of obs, state, reward, done out of env.step(): {obs.shape, state.shape, reward.shape, done.shape}')
        
        obs = torch.Tensor(obs).reshape((-1, self.n_agents)+obs.shape[1:]).to(self.device) # [rollout_threads, n_agents, obs_shape]
        state = torch.Tensor(state).reshape((-1, self.n_agents)+state.shape[1:]).to(self.device) # [rollout_threads, n_agents, state_shape]
        if type(act_masks) != type(None):
            act_masks = torch.Tensor(act_masks).reshape((-1, self.n_agents)+act_masks.shape[1:]).to(self.device) # [rollout_threads, n_agents, act_shape]
        done = torch.Tensor(done).reshape((-1, self.n_agents)).to(self.device) # [rollout_threads, n_agents]

        reward = torch.Tensor(reward).reshape((-1, self.n_agents)).to(self.device) # [rollout_threads, n_agent]

        print(f'Shape of obs, state, reward, done outgoing from env_step(): {obs.shape, state.shape, reward.shape, done.shape}')

        return obs, state, act_masks, reward, done, info
    
    def run(self):

        global_step = 0
        start_time = time.time()

        obs, state, act_masks = self.env_reset()
        next_done = torch.zeros((self.rollout_threads, self.n_agents)).to(self.device)

        num_updates = self.total_timesteps // self.batch_size
        best_return = -1e9
        
        for update in range(1, num_updates + 1):
            if self.lr_decay:
                self.policy.update_lr(update, num_updates)

            total_rewards = 0
            
            nb_games = np.ones(self.rollout_threads)
            nb_wins = np.zeros(self.rollout_threads)

            for step in range(self.env_steps):
                print(f'Update Step: {update} | Env Step: {step}')
                global_step += self.rollout_threads

                with torch.no_grad():
                    action, logprob, _, value = self.policy.get_action_and_value(obs, state, act_masks)

                next_obs, next_state, next_act_masks, reward, done, info = self.env_step(action)

                if len(self.observation_space.shape) == 3:
                    self.buffer.insert(
                        obs,
                        act_masks,
                        action,
                        logprob,
                        reward,
                        value,
                        next_done,
                        step) 
                else:
                    self.buffer.insert(
                        obs,
                        state,
                        act_masks,
                        action,
                        logprob,
                        reward,
                        value,
                        next_done,
                        step)
                
                if self.env_family == 'starcraft':
                    total_rewards += reward.max(-1)[0] # (rollout_threads,)
                    # For each rollout, track the number of games player so far and record the wins for finished games
                    for i in range(self.rollout_threads):
                        if torch.isin(1, done[i]):
                            nb_games[i] += 1 
                            for agent_info in info[i*self.n_agents:(i+1)*self.n_agents]:
                                if 'battle_won' in agent_info:
                                    nb_wins[i] += int(agent_info['battle_won'])
                                    break
                else:
                    total_rewards += reward.sum(-1) # (rollout_threads,)
                
                obs = next_obs
                state = next_state
                act_masks = next_act_masks
                next_done = done

            if self.env_family == 'starcraft':
                total_rewards = total_rewards.cpu()/nb_games
                total_rewards = total_rewards.mean().item()
                episodic_wins = (nb_wins/nb_games).mean()
                print(f"global_step={global_step}, episodic_return={total_rewards}, episodic_win_rate={episodic_wins}")
                if self.use_comet:
                    self.exp.log_metric("episodic_return", total_rewards, global_step)
                    self.exp.log_metric("episodic_win_rate", episodic_wins, global_step)
            else:
                total_rewards = total_rewards.mean().item()
                print(f"global_step={global_step}, episodic_return={total_rewards}")
                if self.use_comet:
                    self.exp.log_metric("episodic_return", total_rewards, global_step)
            
            if total_rewards >= best_return:
                self.save_checkpoints(self.save_dir)
                best_return = total_rewards

            with torch.no_grad():
                
                next_value = self.policy.get_value(next_obs, next_state)
                advantages, returns = self.policy.compute_returns(self.buffer, next_value, next_done)
                

            metrics = self.policy.train_step(self.buffer, advantages, returns)
            if self.use_comet:
                for k in metrics:
                    self.exp.log_metric(k, metrics[k], global_step)
                
                self.exp.log_metric("learning_rate", self.policy.optimizer.param_groups[0]["lr"], global_step)
                self.exp.log_metric("SPS", int(global_step / (time.time() - start_time)), global_step)
    
    def evaluate(self):

        agg_rewards = []
        agg_wins = []
        

        for _ in range(self.eval_episodes):
            total_rewards = 0
            obs_, state_, act_mask_ = self.eval_env.reset()
            for step in range(self.env_steps):

                obs, state, act_mask = [], [], []
                for agent in obs_:
                    obs.append(torch.from_numpy(obs_[agent]))
                    state.append(torch.from_numpy(state_[agent]))
                    act_mask.append(torch.from_numpy(act_mask_[agent]))

                obs = torch.stack(obs, dim=0).unsqueeze(0).to(self.device)
                state = torch.stack(state, dim=0).unsqueeze(0).to(self.device)
                act_mask = torch.stack(act_mask, dim=0).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    action_, _, _, _ = self.policy.get_action_and_value(obs, state, act_mask)
                    if self.action_space.__class__.__name__ == 'Box':
                        action_ = action_.reshape(-1, action_.shape[-1]).cpu().numpy()
                    elif self.action_space.__class__.__name__ == 'Discrete':
                        action_ = action_.reshape(-1).cpu().numpy()
                    else:
                        NotImplementedError
                    action = {}
                    for i, agent in enumerate(obs_.keys()):
                        action[agent] = action_[i]
                
                next_obs, next_state, next_act_mask, reward, done_, info = self.eval_env.step(action)
                
                for agent in reward:
                    total_rewards += reward[agent]
                
                done = False
                for agent in done_:
                    if done_[agent]:
                        done = True
                        break
                if done:
                    break
                obs_ = next_obs
                state_ = next_state
                act_mask_ = next_act_mask

            win = 0
            if self.env_family == 'starcraft':
                for agent in info:
                    if 'battle_won' in info[agent]:
                        win = int(info[agent]['battle_won'])
                        break
            
            agg_rewards.append(total_rewards)
            agg_wins.append(win)
            
                        
        mean_rewards = np.mean(agg_rewards)
        std_rewards = np.std(agg_rewards)

        mean_wins = np.mean(agg_wins)
        std_wins = np.std(agg_wins)

        return mean_rewards, std_rewards, mean_wins, std_wins

    def load_checkpoints(self, checkpoint_dir):
        self.policy.load_checkpoints(checkpoint_dir)

    def save_checkpoints(self, checkpoint_dir):
        self.policy.save_checkpoints(checkpoint_dir)
            
