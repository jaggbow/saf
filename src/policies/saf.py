from einops import rearrange
import math
from perceiver.model.core import InputAdapter
import numpy as np
import sys
import os
import torch
from torch import nn, optim
from torch.distributions.categorical import Categorical
from torch.distributions import MultivariateNormal
from torch.distributions.normal import Normal
from src.architectures.mlp import MLP
from src.utils import *
from src.architectures.mlp import MLP
from src.architectures.cnn import CNN
from perceiver.model.core import PerceiverEncoder, CrossAttention


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class SAF(nn.Module):
    def __init__(self, observation_space, action_space, state_space, params):
        super(SAF, self).__init__()

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

        if self.type == 'conv':
            assert len(self.obs_shape) == 3, 'Convolutional policy cannot be used for non-image observations!'
            self.conv = CNN(out_size=params.conv_out_size)
            self.input_shape = params.conv_out_size
        else:
            self.input_shape = self.obs_shape

       
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.continuous_action = params.continuous_action
        self.action_std_init = params.action_std_init
        if self.continuous_action:
            self.action_var = torch.full(
                (self.action_dim,), self.action_std_init * self.action_std_init)

        # SAF related setting
        self.use_policy_pool = params.use_policy_pool
        self.use_SK = params.use_SK
        self.n_policy = params.n_policy
        self.N_SK_slots = params.N_SK_slots

        # latent kl setting
        self.latent_kl = params.latent_kl
        self.latent_dim = params.latent_dim

        if self.latent_kl:
            input_critic = np.array(self.state_shape).prod() + self.n_agents*self.latent_dim
        else: 
            input_critic = np.array(self.state_shape).prod()
       
        if self.shared_critic:
            self.critic = MLP(
                input_critic, 
                [self.hidden_dim]*self.n_layers, 
                1, 
                std=1.0,
                activation=self.activation)
        else:
            self.critic = nn.ModuleList([MLP(
                input_critic, 
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
                
                self.actor_logstd = nn.Parameter(torch.zeros(1, np.array(self.action_shape).prod()))
            else:
                self.actor = MLP(
                    np.array(self.input_shape).prod(), 
                    [self.hidden_dim]*self.n_layers, 
                    np.array(self.action_shape).prod(), 
                    std=0.01,
                    activation=self.activation)
        elif self.use_policy_pool:
            if self.continuous_action:
                
                self.actor_mean = nn.ModuleList([MLP(
                np.array(self.input_shape).prod(), 
                [self.hidden_dim]*self.n_layers, 
                np.array(self.action_shape).prod(), 
                std=0.01,
                activation=self.activation) for _ in range(self.n_policy)])
                
                self.actor_logstd = nn.ParameterList([
                    nn.Parameter(torch.zeros(1, np.array(self.action_shape).prod())) for _ in range(self.n_policy)])
            else:

                self.actor = nn.ModuleList([MLP(
                    np.array(self.input_shape).prod(), 
                    [self.hidden_dim]*self.n_layers, 
                    np.array(self.action_shape).prod(), 
                    std=0.01,
                    activation=self.activation) for _ in range(self.n_policy)])

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
        
        
        
            
        # SAF module
        self.SAF = Communication_and_policy(input_dim=np.array(self.input_shape).prod(),
                                            key_dim=np.array(self.input_shape).prod(),
                                            N_SK_slots=self.N_SK_slots,
                                            n_agents=self.n_agents, n_policy=self.n_policy,
                                            hidden_dim=self.hidden_dim, n_layers=self.n_layers,
                                            activation=self.activation, latent_kl=self.latent_kl,
                                            latent_dim=self.latent_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-5)

    def get_value(self, x, state, z=None):
        """
        Args:
                x: [batch_size, n_agents, obs_shape]
                state: [batch_size, n_agents, state_shape]
        Returns:
                value: [batch_size, n_agents]
        """
        values = []
        if z is not None:
            for i in range(self.n_agents):
                if self.shared_critic:
                    values.append(self.critic(torch.cat((state[:, 0], z), dim=1)))
                else:
                    values.append(self.critic[i](torch.cat((state[:, 0], z), dim=1)))
        else:
            for i in range(self.n_agents):
                if self.shared_critic:
                    values.append(self.critic(state[:, 0]))
                else:
                    values.append(self.critic[i](state[:, 0]))
        values = torch.stack(values, dim=1).squeeze(-1)
       
        return values  

    def get_action_and_value(self, x, state, action_mask=None, actions=None, x_old=None):
        """
        Args:
            x: [batch_size, n_agents, obs_shape]
            state: [batch_size, n_agents, state_shape]
            action: [batch_size, n_agents]
        Returns:
            action: [batch_size, n_agents]
            logprob: [batch_size, n_agents]
            entropy: [batch_size, n_agents]
            value: [batch_size, n_agents]
        """
        KL = 0
        # print(f'self.type is: {self.type}')
        bs = x.shape[0]
        n_ags = x.shape[1]


        if self.type == 'conv':
            x = x.reshape((-1,)+self.obs_shape)
            x = self.conv(x)
            x = x.reshape(bs, n_ags, self.input_shape)
            state = x.reshape(bs, n_ags * self.input_shape)
            state = state.unsqueeze(1).repeat(1, n_ags, 1)
            if x_old is not None and len(x_old.shape)==5:#use CNN if the x_old is not already processed
      
                bs_, n_ags_=x_old.shape[0],x_old.shape[1]
                x_old = x_old.reshape((-1,)+self.obs_shape)
                x_old = self.conv(x_old)
                x_old = x_old.reshape(bs_, n_ags_, self.input_shape)



        if self.use_SK:
            # communicate among different agents using SK
            x_saf = self.SAF(x)
            if self.type == 'conv':
                state = x_saf.reshape(bs, n_ags * self.input_shape)
            else:
                state = x_saf.reshape(bs, n_ags * self.input_shape[0])
            state = state.unsqueeze(1).repeat(1, n_ags, 1)

        out_actions = []
        logprobs = []
        entropies = []
      
        if self.use_policy_pool:
            # using policy pool

            for j in range(self.n_policy):
                out_actions_ = []
                logprobs_ = []
                entropies_ = []
                for i in range(self.n_agents):
                    if self.shared_actor:
                        if self.continuous_action:
                            action_mean = self.actor_mean[j](x[:, i])
                            action_logstd = self.actor_logstd.expand_as(
                                action_mean)
                            action_std = torch.exp(action_logstd)
                        else:
                            logits = self.actor[j](x[:, i])
                            if type(action_mask) == torch.Tensor:
                                logits[action_mask[:,i]==0] = -1e18
                    else:
                        if self.continuous_action:
                            action_mean = self.actor_mean[j](x[:, i])
                            action_logstd = self.actor_logstd[i].expand_as(
                                action_mean)
                            action_std = torch.exp(action_logstd)
                        else:
                            logits = self.actor[j](x[:, i])
                            if type(action_mask) == torch.Tensor:
                                logits[action_mask[:,i]==0] = -1e18
                            

                    if self.continuous_action:
                        probs = Normal(action_mean, action_std)
                    else:
                        probs = Categorical(logits=logits)

                    if actions is None:
                        action = probs.sample()
                    else:
                        action = actions[:, i]

                    if self.continuous_action:
                        logprob = probs.log_prob(action).sum(1)
                        entropy = probs.entropy().sum(1)
                    else:
                        logprob = probs.log_prob(action)
                        entropy = probs.entropy()

                    out_actions_.append(action)
                    logprobs_.append(logprob)
                    entropies_.append(entropy)

                # [batch_size, n_agents]
                out_actions_ = torch.stack(out_actions_, dim=1)
                logprobs_ = torch.stack(logprobs_, dim=1)
                entropies_ = torch.stack(entropies_, dim=1)

                out_actions.append(out_actions_.unsqueeze(2))
                logprobs.append(logprobs_.unsqueeze(2))
                entropies.append(entropies_.unsqueeze(2))

            # [batch_size, n_agents,n_policy]
            out_actions = torch.cat(out_actions, 2)
            logprobs = torch.cat(logprobs, dim=2)
            entropies = torch.cat(entropies, dim=2)
            # attention for different policy outputs
            attention_score = self.SAF.Run_policy_attention(
                x)  # (bsz,n_agents,n_policy)

            if self.continuous_action:
                out_actions = torch.einsum(
                    'bijk,bij->bik', out_actions.float(), attention_score).long()  # bsz x N_agents
                print('out_actions', out_actions)
            else:
                out_actions = torch.einsum(
                    'bij,bij->bi', out_actions.float(), attention_score).long()  # bsz x N_agents

            logprobs = torch.einsum(
                'bij,bij->bi', logprobs, attention_score)  # bsz x N_agents

            entropies = torch.einsum(
                'bij,bij->bi', entropies, attention_score)  # bsz x N_agents

        else:
            # not using policy pool
            for i in range(self.n_agents):
                if self.shared_actor:
                    if self.continuous_action:
                        action_mean = self.actor_mean(x[:, i])
                        action_logstd = self.actor_logstd.expand_as(
                            action_mean)
                        action_std = torch.exp(action_logstd)
                    else:
                        logits = self.actor(x[:, i])
                        if type(action_mask) == torch.Tensor:
                            logits[action_mask[:,i]==0] = -1e18
                else:
                    if self.continuous_action:
                        action_mean = self.actor_mean[i](x[:, i])
                        action_logstd = self.actor_logstd[i].expand_as(
                            action_mean)
                        action_std = torch.exp(action_logstd)
                    else:
                        logits = self.actor[i](x[:, i])
                        if type(action_mask) == torch.Tensor:
                            logits[action_mask[:,i]==0] = -1e18

                if self.continuous_action:
                    probs = Normal(action_mean, action_std)
                else:
                    probs = Categorical(logits=logits)

                if actions is None:
                    action = probs.sample()
                else:
                    action = actions[:, i]

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

 
        if self.latent_kl:
            z, KL = self.SAF.information_bottleneck(x_saf, x, x_old)
            value = self.get_value(x, state, z)
        else:   
            value = self.get_value(x, state)

        return out_actions, logprobs, entropies, value, KL


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
                delta = buffer.rewards[t] + self.gamma * \
                    nextvalues * nextnonterminal - buffer.values[t]
                adv = lastgaelam = delta + self.gamma * \
                    self.gae_lambda * nextnonterminal * lastgaelam
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

                returns.insert(
                    0, buffer.rewards[t] + self.gamma * nextnonterminal * next_return)

            returns = torch.stack(returns, dim=0)
            advantages = returns - buffer.values

        advantages = advantages.squeeze(-1)
        returns = returns.squeeze(-1)
        return advantages, returns

    def train_step(self, buffer, advantages, returns):
        self.train()
        # flatten the batch
        b_obs = buffer.obs.reshape((-1, self.n_agents) + self.obs_shape)
        if hasattr(buffer, 'state'):
            b_state = buffer.state.reshape((-1, self.n_agents) + self.state_shape)
            #print(f'Hi')
        else:
            #print(f'Hi2')
            b_state = None

        
        if self.latent_kl:
            # old_observation - shifted tensor (the zero-th obs is assumed to be equal to the first one)
            b_obs_old = b_obs.clone()
            b_obs_old[1:] = b_obs_old.clone()[:-1]
        else:   
            b_obs_old = None

        b_logprobs = buffer.logprobs.reshape(-1, self.n_agents)

        if self.continuous_action:
            b_actions = buffer.actions.reshape(
                (-1, self.n_agents)+self.action_shape)
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

                if b_state is not None:
                    if self.continuous_action:
                        _, newlogprob, entropy, newvalue, KL = self.get_action_and_value(b_obs[mb_inds], b_state[mb_inds], None, b_actions[mb_inds], b_obs_old)
                    else:
                        _, newlogprob, entropy, newvalue, KL = self.get_action_and_value(b_obs[mb_inds], b_state[mb_inds], b_action_masks[mb_inds], b_actions.long()[mb_inds], b_obs_old)
                else:
                    if self.continuous_action:
                        _, newlogprob, entropy, newvalue, KL = self.get_action_and_value(b_obs[mb_inds], None, None, b_actions[mb_inds], b_obs_old)
                    else:
                        _, newlogprob, entropy, newvalue, KL= self.get_action_and_value(b_obs[mb_inds], None, b_action_masks[mb_inds], b_actions.long()[mb_inds], b_obs_old)
                

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() >
                                   self.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (
                        mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * \
                    torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
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
                    v_loss = 0.5 * \
                        ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()

                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                
    
                (loss+KL).backward()
                
                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()

            if self.target_kl is not None:
                if approx_kl > self.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - \
            np.var(y_true - y_pred) / var_y

        metrics = {
            'pg_loss': pg_loss.item(),
            'v_loss': v_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'old_approx_kl': old_approx_kl.item(),
            'approx_kl': approx_kl.item(),
            'clipfracs': np.mean(clipfracs),
            'explained_var': explained_var,
            "latent_KL": KL,
        }
        return metrics

    def save_checkpoints(self, checkpoint_dir):
        if self.type == "conv":
            torch.save(self.conv.state_dict(), os.path.join(checkpoint_dir, 'conv.pth'))
        torch.save(self.SAF.state_dict(), os.path.join(checkpoint_dir, 'saf.pth'))
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
        self.SAF.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'saf.pth'), map_location=lambda storage, loc: storage))
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


# Input adapater for perceiver
class agent_input_adapter(InputAdapter):
    def __init__(self, max_seq_len: int, num_input_channels: int):
        super().__init__(num_input_channels=num_input_channels)

        self.pos_encoding = nn.Parameter(
            torch.empty(max_seq_len, num_input_channels))
        self.scale = math.sqrt(num_input_channels)
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.pos_encoding.uniform_(-0.5, 0.5)

    def forward(self, x):
        b, l, dim = x.shape  # noqa: E741
        p_enc = rearrange(self.pos_encoding[:l], "... -> () ...")
        return x * self.scale + p_enc


# ######inter-agent communication


class Communication_and_policy(nn.Module):
    def __init__(self, input_dim, key_dim, N_SK_slots, n_agents, n_policy, hidden_dim, n_layers, activation, latent_kl, latent_dim):
        super(Communication_and_policy, self).__init__()
        self.N_SK_slots = N_SK_slots

        self.n_agents = n_agents

        self.n_policy = n_policy
        self.latent_kl = latent_kl
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.key_dim = key_dim
        self.n_agents = n_agents
        self.n_policy = n_policy

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.policy_keys = torch.nn.Parameter(
            torch.randn(self.n_policy, 1, key_dim)).to(self.device)
        self.policy_attn = nn.MultiheadAttention(
            embed_dim=key_dim, num_heads=1, batch_first=False)

        self.query_projector_s1 = MLP(input_dim,
                                      [self.hidden_dim]*self.n_layers,
                                      key_dim,
                                      std=1.0,
                                      activation=self.activation)  # for sending out message to sk

        self.original_state_projector = MLP(input_dim,
                                            [self.hidden_dim]*self.n_layers,
                                            key_dim,
                                            std=1.0,
                                            activation=self.activation)  # original agent's own state
        self.policy_query_projector = MLP(input_dim,
                                          [self.hidden_dim]*self.n_layers,
                                          key_dim,
                                          std=1.0,
                                          activation=self.activation)  # for query-key attention pick policy form pool

        self.combined_state_projector = MLP(2*key_dim,
                                            [self.hidden_dim]*self.n_layers,
                                            key_dim,
                                            std=1.0,
                                            activation=self.activation).to(self.device)  # responsible for independence of the agent
        # shared knowledge(workspace)

        input_adapter = agent_input_adapter(num_input_channels=key_dim, max_seq_len=n_agents).to(
            self.device)  # position encoding included as well, so we know which agent is which

        self.PerceiverEncoder = PerceiverEncoder(
            input_adapter=input_adapter,
            num_latents=N_SK_slots,  # N
            num_latent_channels=key_dim,  # D
            num_cross_attention_qk_channels=key_dim,  # C
            num_cross_attention_heads=1,
            num_self_attention_heads=1,  # small because observational space is small
            num_self_attention_layers_per_block=self.n_layers,
            num_self_attention_blocks=self.n_layers,
            dropout=0.0,
        ).to(self.device)
        self.SK_attention_read = CrossAttention(
            num_heads=1,
            num_q_input_channels=key_dim,
            num_kv_input_channels=key_dim,
            num_qk_channels=key_dim,
            num_v_channels=key_dim,
            dropout=0.0,
        ).to(self.device)

        if self.latent_kl:
            self.encoder = nn.Sequential(
                nn.Linear(int(2*key_dim), 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, int(2*self.latent_dim)),
            ).to(self.device)

            self.encoder_prior = nn.Sequential(
                nn.Linear(key_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, int(2*self.latent_dim)),
            ).to(self.device)

        self.previous_state = torch.randn(5, 1, key_dim).to(self.device)

    def forward(self, state):
        # sate has shape (bsz,N_agents,embsz)
        # communicate among agents using perceiver
        state = state.to(self.device).permute(1, 0, 2)
        N_agents, bsz, Embsz = state.shape
        state = state.permute(1, 0, 2)
        # message (bsz,N_agent,dim), for communication
        message_to_send = self.query_projector_s1(state)
        state_encoded = self.original_state_projector(
            state)  # state_encoded, for agent's internal uses

        # use perceiver arttecture to collect information from all agents by attention

        SK_slots = self.PerceiverEncoder(message_to_send)
        message = self.SK_attention_read(message_to_send, SK_slots)

        # message plus original state
        # shape (bsz,N_agents,2*dim)
        state_with_message = torch.cat([state_encoded, message], 2)

        state_with_message = state_with_message.permute(
            1, 0, 2)  # (N_agents,bsz,2*dim)

        state_with_message = self.combined_state_projector(
            state_with_message)  # (N_agents,bsz,dim)

        state_with_message = state_with_message.permute(
            1, 0, 2)  # (bsz,N_agents,dim)

        # print(state_with_message.shape)
        return state_with_message

    def forward_NoCommunication(self, state):
        # jsut encoder the original state without communication
        state = state.to(self.device)
        N_agents, bsz, Embsz = state.shape
        state = state.permute(1, 0, 2)
        state_encoded = self.original_state_projector(
            state)  # state_encoded, for agent's internal uses

        state_without_message = torch.cat([state_encoded, torch.zeros(
            state_encoded.shape).to(self.device)], 2)  # without information from other agents

        state_without_message = state_without_message.permute(
            1, 0, 2)  # (N_agents,bsz,2*dim)

        state_without_message = self.combined_state_projector(
            state_without_message)  # (N_agents,bsz,dim)

        return state_without_message

    def Run_policy_attention(self, state):
        '''
        state hasshape (bsz,N_agents,embsz)
        '''
        state = state.permute(1, 0, 2)  # (N_agents,bsz,embsz)
        state = state.to(self.device)
        # how to pick rules and if they are shared across agents
        query = self.policy_query_projector(state)
        N_agents, bsz, Embsz = query.shape

        keys = self.policy_keys.repeat(1, bsz, 1)  # n_ploicies,bsz,Embsz,

        _, attention_score = self.policy_attn(query, keys, keys)

        attention_score = nn.functional.gumbel_softmax(
            attention_score, tau=1, hard=True, dim=2)  # (Bz, N_agents , N_behavior)

        return attention_score

    def information_bottleneck(self, state_with_message, state_without_message, s_agent_previous_t):


        z_ = self.encoder(
            torch.cat((state_with_message, state_without_message), dim=2))
        mu, sigma = z_.chunk(2, dim=2)
        z = (mu + sigma * torch.randn_like(sigma)).reshape(z_.shape[0], -1)
        z_prior = self.encoder_prior(s_agent_previous_t)
        mu_prior, sigma_prior = z_prior.chunk(2, dim=2)
        KL = 0.5 * torch.sum(((mu - mu_prior) ** 2 + sigma ** 2)/(sigma_prior ** 2 + 1e-8) + torch.log(1e-8 + (sigma_prior ** 2)/(
            sigma ** 2 + 1e-8)) - 1) / np.prod(torch.cat((state_with_message, state_without_message), dim=2).shape)

        return z, KL
