import os
import numpy as np

import torch
from torch import nn, optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from src.architectures.mlp import MLP
from src.architectures.cnn import CNN

from src.utils import *


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class IPPO(nn.Module):
    def __init__(self, observation_space, action_space, state_space, params):
        super(IPPO, self).__init__()
        # https://ppo-details.cleanrl.dev//2021/11/05/ppo-implementation-details/
        
        self.type = params.type
        self.obs_shape = get_obs_shape(observation_space)
        self.state_shape = get_state_shape(state_space)
        self.action_shape = get_act_shape(action_space)
        self.n_layers = params.n_layers
        self.hidden_dim = params.hidden_dim
        self.activation = params.activation
        self.learning_rate = params.learning_rate
        self.gamma = params.gamma
        self.gae_lambda = params.gae_lambda
        self.gae = params.gae
        self.n_agents = params.n_agents
        self.shared_actor = params.shared_actor
        self.shared_critic = params.shared_critic
        self.continuous_action = params.continuous_action

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

        if self.type == 'conv':
            assert len(self.obs_shape) == 3, 'Convolutional policy cannot be used for non-image observations!'
            self.conv = CNN(out_size=params.conv_out_size)
            self.input_shape = params.conv_out_size
        else:
            self.input_shape = self.obs_shape
        
        if self.shared_critic:
            self.critic = MLP(
                np.array(self.input_shape).prod(), 
                [self.hidden_dim]*self.n_layers, 
                1, 
                std=1.0,
                activation=self.activation)
        else:
            self.critic = nn.ModuleList([MLP(
                np.array(self.input_shape).prod(), 
                [self.hidden_dim]*self.n_layers, 
                1, 
                std=1.0,
                activation=self.activation) for _ in range(self.n_agents)])
        
        if self.shared_actor:
            if self.continuous_action:
                self.actor_mean = MLP(
                np.array(self.input_shape).prod(), 
                [self.hidden_dim]*self.n_layers, 
                np.array(self.action_shape).prod(), 
                std=0.01,
                activation=self.activation)
                self.actor_logstd = nn.ParameterList([nn.Parameter(torch.zeros(1, np.array(self.action_shape).prod()))])
            else:
                self.actor = MLP(
                    np.array(self.input_shape).prod(), 
                    [self.hidden_dim]*self.n_layers, 
                    np.array(self.action_shape).prod(), 
                    std=0.01,
                    activation=self.activation)
        else:
            if self.continuous_action:
                
                self.actor_mean = nn.ModuleList([MLP(
                np.array(self.input_shape).prod(), 
                [self.hidden_dim]*self.n_layers, 
                np.array(self.action_shape).prod(), 
                std=0.01,
                activation=self.activation) for _ in range(self.n_agents)])
                self.actor_logstd = nn.ParameterList([
                    nn.Parameter(torch.zeros(1, np.array(self.action_shape).prod())) for _ in range(self.n_agents)])
            else:
                self.actor = nn.ModuleList([MLP(
                    np.array(self.input_shape).prod(), 
                    [self.hidden_dim]*self.n_layers, 
                    np.array(self.action_shape).prod(), 
                    std=0.01,
                    activation=self.activation) for _ in range(self.n_agents)])
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-5)

    def get_value(self, x, state=None, z=None):
        """
        Args:
            x: [batch_size, n_agents, obs_shape]
            state: [batch_size, n_agents, state_shape]
        Returns:
            value: [batch_size, n_agents]
        """
        
        # if self.type == 'conv':
        #     bs = x.shape[0]
        #     n_ags = x.shape[1]
        #     x = x.reshape((-1,)+self.obs_shape)
        #     x = self.conv(x)
        #     x = x.reshape(bs, n_ags, self.input_shape)

        values = []
        for i in range(self.n_agents): 
            if self.shared_critic:
                values.append(self.critic(x[:,i]))
            else:
                values.append(self.critic[i](x[:,i]))
        values = torch.stack(values, dim=1).squeeze(-1)
        return values

    def get_action_and_value(self, x, state=None, action_mask=None, actions=None, obs_old=None):
        """
        Args:
            x: [batch_size, n_agents, obs_shape]
            action_mask: [batch_size, n_agents, n_actions]
            state: [batch_size, n_agents, state_shape]
            actions: [batch_size, n_agents]
        Returns:
            action: [batch_size, n_agents]
            logprob: [batch_size, n_agents]
            entropy: [batch_size, n_agents]
            value: [batch_size, n_agents]
        """
        

        
        if self.type == 'conv':
            bs = x.shape[0]
            n_ags = x.shape[1]
            x = x.reshape((-1,)+self.obs_shape)
            x = self.conv(x)
            x = x.reshape(bs, n_ags, self.input_shape)
           
        out_actions = []
        logprobs = []
        entropies = []
        
        for i in range(self.n_agents):
            if self.shared_actor:
                if self.continuous_action:
                    action_mean = self.actor_mean[0](x[:,i])
                    action_logstd = self.actor_logstd.expand_as(action_mean)
                    action_std = torch.exp(action_logstd)
                else:
                    logits = self.actor(x[:,i])
                    if type(action_mask) == torch.Tensor:
                        logits[action_mask[:,i]==0] = -1e18
            else:
                if self.continuous_action:
                    action_mean = self.actor_mean[i](x[:,i])
                    action_logstd = self.actor_logstd[i].expand_as(action_mean)
                    action_std = torch.exp(action_logstd)
                else:
                    logits = self.actor[i](x[:,i])
                    if type(action_mask) == torch.Tensor:
                        logits[action_mask[:,i]==0] = -1e18
            
            if self.continuous_action:
                probs = Normal(action_mean, action_std)
            else:
                probs = Categorical(logits=logits)
            
            if actions is None:
                action = probs.sample()
            else:
                action = actions[:,i]
            
            if self.continuous_action:
                logprob = probs.log_prob(action).sum(1)
                entropy = probs.entropy().sum(1)
            else:
                logprob = probs.log_prob(action)
                entropy = probs.entropy()

            out_actions.append(action)
            logprobs.append(logprob)
            entropies.append(entropy)
        
        out_actions = torch.stack(out_actions, dim=1)
        logprobs = torch.stack(logprobs, dim=1)
        entropies = torch.stack(entropies, dim=1)
        value = self.get_value(x)
        
        return out_actions, logprobs, entropies, value, None
    
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
        # Flatten the batch
        b_obs = buffer.obs.reshape((-1, self.n_agents) + self.obs_shape)
        b_logprobs = buffer.logprobs.reshape(-1, self.n_agents)
        if self.continuous_action:
            b_actions = buffer.actions.reshape((-1, self.n_agents)+self.action_shape)
        else:
            b_actions = buffer.actions.reshape((-1, self.n_agents))
        b_action_masks = buffer.action_masks.reshape((-1, self.n_agents)+self.action_shape)
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
                
                if self.continuous_action:
                    _, newlogprob, entropy, newvalue, _ = self.get_action_and_value(b_obs[mb_inds], None, None, b_actions[mb_inds])
                else:
                    _, newlogprob, entropy, newvalue, _ = self.get_action_and_value(b_obs[mb_inds], None, b_action_masks[mb_inds], b_actions.long()[mb_inds])
                
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
        if self.type == "conv":
            torch.save(self.conv.state_dict(), os.path.join(checkpoint_dir, 'conv.pth'))
        if self.continuous_action: 
            torch.save(self.actor_mean.state_dict(), os.path.join(checkpoint_dir, 'actor_mean.pth'))
            torch.save(self.critic.state_dict(), os.path.join(checkpoint_dir, 'critic.pth'))
            if self.shared_actor:
                state = dict(actor_logstd=self.actor_logstd)
                torch.save(state, os.path.join(checkpoint_dir, 'actor_logstd.pth'))
            else:
                torch.save(self.actor_logstd.state_dict(), os.path.join(checkpoint_dir, 'actor_logstd.pth'))
        else:
            torch.save(self.actor.state_dict(), os.path.join(checkpoint_dir, 'actor.pth'))
            torch.save(self.critic.state_dict(), os.path.join(checkpoint_dir, 'critic.pth'))
    
    def load_checkpoints(self, checkpoint_dir):
        if self.type == "conv":
            self.conv.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'conv.pth'), map_location=lambda storage, loc: storage))
        if self.continuous_action: 
            self.actor_mean.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'actor_mean.pth'), map_location=lambda storage, loc: storage))
            self.critic.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'critic.pth'), map_location=lambda storage, loc: storage))
            if self.shared_actor:
                self.actor_logstd = torch.load(os.path.join(checkpoint_dir, 'actor_logstd.pth'))['actor_logstd']
            else:
                self.actor_logstd.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'actor_logstd.pth'), map_location=lambda storage, loc: storage))
        else:
            self.actor.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'actor.pth'), map_location=lambda storage, loc: storage))
            self.critic.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'critic.pth'), map_location=lambda storage, loc: storage))