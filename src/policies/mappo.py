import os
import numpy as np

import torch
from torch import nn, optim
from torch.distributions.categorical import Categorical

from src.architectures.mlp import MLP
from src.utils import *


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MAPPO(nn.Module):
    def __init__(self, observation_space, action_space, params):
        super(MAPPO, self).__init__()
        
        self.obs_shape = get_obs_shape(observation_space)
        self.action_dim = action_space.n
        self.n_layers = params.n_layers
        self.hidden_dim = params.hidden_dim
        self.activation = params.activation
        self.learning_rate = params.learning_rate
        self.gamma = params.gamma
        self.gae_lambda = params.gae_lambda
        self.gae = params.gae
        self.n_agents = params.n_agents

        self.batch_size = params.rollout_threads * params.env_steps
        self.minibatch_size = self.batch_size // params.num_minibatches
        self.ent_coef = params.ent_coef
        self.vf_coef = params.vf_coef
        self.norm_adv = params.norm_adv
        self.clip_coef = params.clip_coef
        self.clip_vloss = params.clip_vloss
        self.max_grad_norm = params.max_grad_norm
        self.target_kl = params.target_kl
        self.update_epochs = params.update_epochs
        self.shared_critic = params.shared_critic
        self.shared_actor = params.shared_actor

        if self.shared_critic:
            self.critic = MLP(
                self.n_agents * np.array(self.obs_shape).prod(), 
                [self.hidden_dim]*self.n_layers, 
                1, 
                std=1.0,
                activation=self.activation)
        else:
            self.critic = nn.ModuleList([MLP(
                self.n_agents * np.array(self.obs_shape).prod(), 
                [self.hidden_dim]*self.n_layers, 
                1, 
                std=1.0,
                activation=self.activation) for _ in range(self.n_agents)])

        if self.shared_actor:
            self.actor = MLP(
                np.array(self.obs_shape).prod(), 
                [self.hidden_dim]*self.n_layers, 
                self.action_dim, 
                std=0.01,
                activation=self.activation)
        else:
           self.actor = nn.ModuleList([MLP(
                np.array(self.obs_shape).prod(), 
                [self.hidden_dim]*self.n_layers, 
                self.action_dim, 
                std=0.01,
                activation=self.activation) for _ in range(self.n_agents)])
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-5)

    def get_value(self, x):
        """
        Args:
            x: [batch_size, n_agents, obs_shape]
        Returns:
            value: [batch_size, n_agents]
        """
        values = []
        state = x.reshape(x.shape[0], -1)
        for i in range(self.n_agents): 
            if self.shared_critic:
                values.append(self.critic(state))
            else:
                values.append(self.critic[i](state))
        values = torch.stack(values, dim=1).squeeze(-1)
        return values

    def get_action_and_value(self, x, actions=None):
        """
        Args:
            x: [batch_size, n_agents, obs_shape]
            action: [batch_size, n_agents]
        Returns:
            action: [batch_size, n_agents]
            logprob: [batch_size, n_agents]
            entropy: [batch_size, n_agents]
            value: [batch_size, n_agents]
        """
        out_actions = []
        logprobs = []
        entropies = []
        for i in range(self.n_agents):
            if self.shared_actor:
                logits = self.actor(x[:,i])
            else:
                logits = self.actor[i](x[:,i])
            probs = Categorical(logits=logits)
            if actions is None:
                action = probs.sample()
            else:
                action = actions[:,i]
            logprob = probs.log_prob(action)
            entropy = probs.entropy()

            out_actions.append(action)
            logprobs.append(logprob)
            entropies.append(entropy)
        
        out_actions = torch.stack(out_actions, dim=1)
        logprobs = torch.stack(logprobs, dim=1)
        entropies = torch.stack(entropies, dim=1)
        value = self.get_value(x)
        
        return out_actions, logprobs, entropies, value
    
    def update_lr(self, step, total_steps):
        frac = 1.0 - (step - 1.0) / total_steps
        lr = frac * self.learning_rate
        self.optimizer.param_groups[0]["lr"] = lr
        return lr

    def compute_returns(self, buffer, next_value, next_done):
        """
        Args:
            buffer
            next_value: [batch_size, n_agents]
            next_done: [batch_size, n_agents]
        returns:
            advantages: [bach_size, n_agents]
            returns: [bach_size, n_agents]
        """
            
        if self.gae:
            advantages = []
            lastgaelam = 0
            for t in reversed(range(buffer.env_steps)):
                if t == buffer.env_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - buffer.dones[t + 1]
                    nextvalues = buffer.values[t + 1]
                delta = buffer.rewards[t] + self.gamma * nextvalues * nextnonterminal - buffer.values[t]
                adv = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                advantages.insert(0, adv)
            
            advantages = torch.stack(advantages, dim=0)
            returns = advantages + buffer.values
        else:
            returns = []
            for t in reversed(range(buffer.env_steps)):
                if t == buffer.env_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    next_return = next_value
                else:
                    nextnonterminal = 1.0 - buffer.dones[t + 1]
                    next_return = returns[0]

                returns.insert(0, buffer.rewards[t] + self.gamma * nextnonterminal * next_return)
            
            returns = torch.stack(returns, dim=0)
            advantages = returns - buffer.values
        
        advantages = advantages.squeeze(-1)
        returns = returns.squeeze(-1)
        return advantages, returns
    
    def train_step(self, buffer, advantages, returns):
        self.train()
        # flatten the batch
        b_obs = buffer.obs.reshape((-1, self.n_agents) + self.obs_shape)
        b_logprobs = buffer.logprobs.reshape(-1, self.n_agents)
        b_actions = buffer.actions.reshape((-1, self.n_agents))
        b_advantages = advantages.reshape(-1, self.n_agents)
        b_returns = returns.reshape(-1, self.n_agents)
        b_values = buffer.values.reshape(-1, self.n_agents)


        # Optimizing the policy and value network
        b_inds = np.arange(self.batch_size)
        clipfracs = []
        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss

                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()

            if self.target_kl is not None:
                if approx_kl > self.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        metrics = {
            'pg_loss': pg_loss.item(),
            'v_loss': v_loss.item(),
            'entropy_loss': entropy_loss.item(), 
            'old_approx_kl': old_approx_kl.item(), 
            'approx_kl': approx_kl.item(), 
            'clipfracs': np.mean(clipfracs), 
            'explained_var': explained_var
        }
        return metrics
    
    def save_checkpoints(self, checkpoint_dir):
        torch.save(self.actor.state_dict(), os.path.join(checkpoint_dir, 'actor.pth'))
        torch.save(self.critic.state_dict(), os.path.join(checkpoint_dir, 'critic.pth'))
    
    def load_checkpoints(self, checkpoint_dir):
        self.actor.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'actor.pth'), map_location=lambda storage, loc: storage))
        self.critic.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'critic.pth'), map_location=lambda storage, loc: storage))