import numpy as np
import os
from os.path import expanduser, expandvars
import time
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
import comet_ml

import torch

class PGRunner:
    def __init__(self, env, env_family, policy, buffer, params, device):
        
        self.total_timesteps = params.total_timesteps
        self.batch_size = params.rollout_threads * params.env_steps
        self.rollout_threads = params.rollout_threads
        self.n_agents = params.n_agents
        self.env_steps = params.env_steps
        self.lr_decay = params.lr_decay
        self.env_family = env_family
        self.eval_episodes = params.eval_episodes
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.use_comet = True if params.comet else False
        self.checkpoint_dir = params.checkpoint_dir
        self.save_dir = Path(expandvars(expanduser(str(params.save_dir)))).resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        if self.use_comet:
            self.exp = comet_ml.Experiment(project_name=params.comet.project_name)
            self.exp.set_name(f"{policy.__class__.__name__}_{os.environ['SLURM_JOB_ID']}")

        self.env = env
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
        obs, state, act_masks = self.env.reset()
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
        if self.action_space.__class__.__name__ == 'Box':
            action_ = action.reshape(-1, action.shape[-1]).cpu().numpy()
        elif self.action_space.__class__.__name__ == 'Discrete':
            action_ = action.reshape(-1).cpu().numpy()
        else:
            NotImplementedError
        
        obs, state, act_masks, reward, done, info = self.env.step(action_)
        
        obs = torch.Tensor(obs).reshape((-1, self.n_agents)+obs.shape[1:]).to(self.device) # [rollout_threads, n_agents, obs_shape]
        state = torch.Tensor(state).reshape((-1, self.n_agents)+state.shape[1:]).to(self.device) # [rollout_threads, n_agents, state_shape]
        if type(act_masks) != type(None):
            act_masks = torch.Tensor(act_masks).reshape((-1, self.n_agents)+act_masks.shape[1:]).to(self.device) # [rollout_threads, n_agents, act_shape]
        done = torch.Tensor(done).reshape((-1, self.n_agents)).to(self.device) # [rollout_threads, n_agents]

        reward = torch.Tensor(reward).reshape((-1, self.n_agents)).to(self.device) # [rollout_threads, n_agent]

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
                global_step += self.rollout_threads

                with torch.no_grad():
                    action, logprob, _, value = self.policy.get_action_and_value(obs, state, act_masks)

                next_obs, next_state, next_act_masks, reward, done, info = self.env_step(action)

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
        obs, state, act_mask = self.env_reset()
        for _ in range(self.eval_episodes):
            total_rewards = 0
            for _ in range(self.env_steps):

                with torch.no_grad():
                    action, _, _, _ = self.policy.get_action_and_value(obs, state, act_mask)

                next_obs, next_state, next_act_mask, reward, _, info = self.env_step(action)
                total_rewards += reward.mean(0).sum()
                
                obs = next_obs
                state = next_state
                act_mask = next_act_mask
            
            episodic_wins = 0
            for i in range(self.rollout_threads):
                win = False
                for agent_info in info[i*self.n_agents:(i+1)*self.n_agents]:
                    if 'battle_won' in agent_info:
                        win = agent_info['battle_won']
                        break
                episodic_wins += int(win)/self.rollout_threads

            agg_rewards.append(total_rewards.item())
            agg_wins.append(episodic_wins)
            
                        
        mean_rewards = np.mean(agg_rewards)
        std_rewards = np.std(agg_rewards)

        mean_wins = np.mean(agg_wins)
        std_wins = np.std(agg_wins)

        return mean_rewards, std_rewards, mean_wins, std_wins

    def load_checkpoints(self, checkpoint_dir):
        self.policy.load_checkpoints(checkpoint_dir)

    def save_checkpoints(self, checkpoint_dir):
        self.policy.save_checkpoints(checkpoint_dir)
            
