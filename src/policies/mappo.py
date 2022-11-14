import os
import numpy as np
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from src.architectures.mlp import MLP
from src.architectures.cnn import CNN
from src.architectures.rnn import GRURNN
from src.utils import *


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MAPPO(nn.Module):
    def __init__(self, observation_space, action_space, state_space, params):
        super(MAPPO, self).__init__()
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
        self.use_rnn = params.use_rnn
        self.tbptt_steps = params.tbptt_steps

        self.batch_size = params.rollout_threads * params.env_steps
        self.env_steps = params.env_steps
        self.rollout_threads = params.rollout_threads
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
        self.continuous_action = params.continuous_action

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.type == "conv":
            assert (
                len(self.obs_shape) == 3
            ), "Convolutional policy cannot be used for non-image observations!"
            self.conv = CNN(out_size=params.conv_out_size)
            self.input_shape = params.conv_out_size
        else:
            self.input_shape = self.obs_shape

        if self.shared_critic:
            self.critic = MLP(
                np.array(self.state_shape).prod(),
                [self.hidden_dim] * self.n_layers,
                1,
                std=1.0,
                activation=self.activation,
            )
        else:
            self.critic = nn.ModuleList(
                [
                    MLP(
                        np.array(self.state_shape).prod(),
                        [self.hidden_dim] * self.n_layers,
                        1,
                        std=1.0,
                        activation=self.activation,
                    )
                    for _ in range(self.n_agents)
                ]
            )

        if self.shared_actor:
            if self.use_rnn:
                self.actor = GRURNN(
                    np.array(self.input_shape).prod(),
                    self.hidden_dim,
                    np.array(self.action_shape).prod(),
                    std=0.01,
                    activation=self.activation,
                )
            else:
                self.actor = MLP(
                    np.array(self.input_shape).prod(),
                    [self.hidden_dim] * self.n_layers,
                    np.array(self.action_shape).prod(),
                    std=0.01,
                    activation=self.activation,
                )
        else:
            if self.use_rnn:
                self.actor = nn.ModuleList(
                    [
                        GRURNN(
                            np.array(self.input_shape).prod(),
                            self.hidden_dim,
                            np.array(self.action_shape).prod(),
                            std=0.01,
                            activation=self.activation,
                        )
                        for _ in range(self.n_agents)
                    ]
                )
            else:
                self.actor = nn.ModuleList(
                    [
                        MLP(
                            np.array(self.input_shape).prod(),
                            [self.hidden_dim] * self.n_layers,
                            np.array(self.action_shape).prod(),
                            std=0.01,
                            activation=self.activation,
                        )
                        for _ in range(self.n_agents)
                    ]
                )

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-5)
        self.init_hidden()

    def init_hidden(self):
        self.hidden_state = [
            torch.zeros(1, self.hidden_dim).to(self.device)
            for _ in range(self.n_agents)
        ]

    def get_value(self, x, state, z=None):
        """
        Args:
            x: [batch_size, n_agents, obs_shape]
            state: [batch_size, n_agents, state_shape]
        Returns:
            value: [batch_size, n_agents]
        """

        # print(f'Shape of x is: {x.shape}')
        # print(f'Shape of state is: {state.shape}')
        values = []
        for i in range(self.n_agents):
            if self.shared_critic:
                values.append(self.critic(state[:, 0]))
            else:
                # print(f'Shape of state: {state.shape}')
                values.append(self.critic[i](state[:, 0]))
        values = torch.stack(values, dim=1).squeeze(-1)
        return values

    def get_action_and_value(
        self, x, state, action_mask=None, actions=None, obs_old=None, hidden_state=None
    ):
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

        # print(f'self.type is: {self.type}')

        if self.type == "conv":
            bs = x.shape[0]
            n_ags = x.shape[1]
            x = x.reshape((-1,) + self.obs_shape)
            x = self.conv(x)
            x = x.reshape(bs, n_ags, self.input_shape)
            state = x.reshape(bs, n_ags * self.input_shape)
            state = state.unsqueeze(1).repeat(1, n_ags, 1)

        out_actions = []
        logprobs = []
        entropies = []

        hidden_state_new = []
        for i in range(self.n_agents):
            if self.shared_actor:
                if self.use_rnn:
                    logits, hidden_state_ = self.actor(x[:, i], hidden_state[i])
                else:
                    logits = self.actor(x[:, i])
                if type(action_mask) == torch.Tensor:
                    logits[action_mask[:, i] == 0] = -1e18
            else:
                if self.use_rnn:
                    logits, hidden_state_ = self.actor[i](x[:, i], hidden_state[i])
                else:
                    logits = self.actor[i](x[:, i])
                if type(action_mask) == torch.Tensor:
                    logits[action_mask[:, i] == 0] = -1e18

            if self.use_rnn:
                hidden_state_new.append(hidden_state_)
            else:
                hidden_state_new.append(None)

            probs = Categorical(logits=logits)

            if actions is None:
                action = probs.sample()
            else:
                action = actions[:, i]

            logprob = probs.log_prob(action)
            entropy = probs.entropy()

            out_actions.append(action)
            logprobs.append(logprob)
            entropies.append(entropy)

        out_actions = torch.stack(out_actions, dim=1)
        logprobs = torch.stack(logprobs, dim=1)
        entropies = torch.stack(entropies, dim=1)
        value = self.get_value(x, state)

        return out_actions, logprobs, entropies, value, None, hidden_state_new

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
                delta = (
                    buffer.rewards[t]
                    + self.gamma * nextvalues * nextnonterminal
                    - buffer.values[t]
                )
                adv = lastgaelam = (
                    delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                )
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
                    0, buffer.rewards[t] + self.gamma * nextnonterminal * next_return
                )

            returns = torch.stack(returns, dim=0)
            advantages = returns - buffer.values

        advantages = advantages.squeeze(-1)
        returns = returns.squeeze(-1)
        return advantages, returns

    def train_step(self, buffer, advantages, returns):
        self.train()
        # flatten the batch
        b_obs = buffer.obs.reshape((-1, self.n_agents) + self.obs_shape)

        if hasattr(buffer, "state"):
            b_state = buffer.state.reshape((-1, self.n_agents) + self.state_shape)
            # print(f'Hi')
        else:
            # print(f'Hi2')
            b_state = None

        b_logprobs = buffer.logprobs.reshape(-1, self.n_agents)
        b_actions = buffer.actions.reshape((-1, self.n_agents))

        b_action_masks = buffer.action_masks.reshape(
            (-1, self.n_agents) + self.action_shape
        )
        b_advantages = advantages.reshape(-1, self.n_agents)
        b_returns = returns.reshape(-1, self.n_agents)
        b_values = buffer.values.reshape(-1, self.n_agents)

        # Optimizing the policy and value network
        bt_inds = np.arange(self.batch_size)
        bt_inds = bt_inds.reshape(self.rollout_threads, -1)
        b_idx = np.arange(self.rollout_threads)

        clipfracs = []
        for epoch in range(self.update_epochs):
            np.random.shuffle(b_idx)
            self.init_hidden()
            for i in range(self.n_agents):
                self.hidden_state[i] = self.hidden_state[i].repeat(
                    self.rollout_threads, 1
                )

            t_logratio = []
            t_ratio = []
            tmb_advantages = []
            t_newvalue = []
            tb_returns = []
            tb_values = []

            for t in range(self.env_steps):
                mb_inds = bt_inds[b_idx, t]
                if self.tbptt_steps > 0:
                    if t == self.env_steps - self.tbptt_steps:
                        for i in range(self.n_agents):
                            self.hidden_state[i] = self.hidden_state[i].detach()

                if b_state is not None:
                    (
                        _,
                        newlogprob,
                        entropy,
                        newvalue,
                        _,
                        self.hidden_state,
                    ) = self.get_action_and_value(
                        b_obs[mb_inds],
                        b_state[mb_inds],
                        b_action_masks[mb_inds],
                        b_actions.long()[mb_inds],
                        hidden_state=self.hidden_state,
                    )
                else:
                    (
                        _,
                        newlogprob,
                        entropy,
                        newvalue,
                        _,
                        self.hidden_state,
                    ) = self.get_action_and_value(
                        b_obs[mb_inds],
                        None,
                        b_action_masks[mb_inds],
                        b_actions.long()[mb_inds],
                        hidden_state=self.hidden_state,
                    )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                mb_advantages = b_advantages[mb_inds]

                t_logratio.append(logratio)
                t_ratio.append(ratio)
                tmb_advantages.append(mb_advantages)
                t_newvalue.append(newvalue)
                tb_returns.append(b_returns[mb_inds])
                tb_values.append(b_values[mb_inds])

            t_logratio = torch.cat(t_logratio, dim=0)
            t_ratio = torch.cat(t_ratio, dim=0)
            tmb_advantages = torch.cat(tmb_advantages, dim=0)
            t_newvalue = torch.cat(t_newvalue, dim=0)
            tb_returns = torch.cat(tb_returns, dim=0)
            tb_values = torch.cat(tb_values, dim=0)

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-t_logratio).mean()
                approx_kl = ((t_ratio - 1) - t_logratio).mean()
                clipfracs += [
                    ((t_ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                ]

            if self.norm_adv:
                tmb_advantages = (tmb_advantages - tmb_advantages.mean()) / (
                    tmb_advantages.std() + 1e-8
                )

            # Policy loss
            pg_loss1 = -tmb_advantages * t_ratio
            pg_loss2 = -tmb_advantages * torch.clamp(
                t_ratio, 1 - self.clip_coef, 1 + self.clip_coef
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            if self.clip_vloss:
                v_loss_unclipped = (t_newvalue - tb_returns) ** 2
                v_clipped = tb_values + torch.clamp(
                    t_newvalue - tb_values,
                    -self.clip_coef,
                    self.clip_coef,
                )
                v_loss_clipped = (v_clipped - tb_returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((t_newvalue - tb_returns) ** 2).mean()

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
            "pg_loss": pg_loss.item(),
            "v_loss": v_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "old_approx_kl": old_approx_kl.item(),
            "approx_kl": approx_kl.item(),
            "clipfracs": np.mean(clipfracs),
            "explained_var": explained_var,
        }
        return metrics

    def save_checkpoints(self, checkpoint_dir):
        if self.type == "conv":
            torch.save(self.conv.state_dict(), os.path.join(checkpoint_dir, "conv.pth"))

        torch.save(self.actor.state_dict(), os.path.join(checkpoint_dir, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(checkpoint_dir, "critic.pth"))

    def load_checkpoints(self, checkpoint_dir):
        if self.type == "conv":
            self.conv.load_state_dict(
                torch.load(
                    os.path.join(checkpoint_dir, "conv.pth"),
                    map_location=lambda storage, loc: storage,
                )
            )
        self.actor.load_state_dict(
            torch.load(
                os.path.join(checkpoint_dir, "actor.pth"),
                map_location=lambda storage, loc: storage,
            )
        )
        self.critic.load_state_dict(
            torch.load(
                os.path.join(checkpoint_dir, "critic.pth"),
                map_location=lambda storage, loc: storage,
            )
        )
