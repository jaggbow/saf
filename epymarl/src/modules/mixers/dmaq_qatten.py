import torch as th
import torch.nn as nn
import numpy as np
from .dmaq_qatten_weight import Qatten_Weight
from .dmaq_si_weight import DMAQ_SI_Weight
from modules.agents.cnn import CNN


class DMAQ_QattenMixer(nn.Module):
    def __init__(self, args):
        super(DMAQ_QattenMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        if self.args.use_cnn:
            self.state_dim = args.mixing_embed_dim * args.n_agents
        else:
            self.state_dim = int(np.prod(args.state_shape))
        self.action_dim = args.n_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim + 1
        if args.use_cnn:
            self.cnn = CNN(in_channels=3, out_size=args.mixing_embed_dim)
        self.attention_weight = Qatten_Weight(args)
        self.si_weight = DMAQ_SI_Weight(args)

    def calc_v(self, agent_qs):
        agent_qs = agent_qs.view(-1, self.n_agents)
        v_tot = th.sum(agent_qs, dim=-1)
        return v_tot

    def calc_adv(self, agent_qs, states, actions, max_q_i):
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        agent_qs = agent_qs.view(-1, self.n_agents)
        max_q_i = max_q_i.view(-1, self.n_agents)

        adv_q = (agent_qs - max_q_i).view(-1, self.n_agents).detach()

        adv_w_final = self.si_weight(states, actions)
        adv_w_final = adv_w_final.view(-1, self.n_agents)

        if self.args.is_minus_one:
            adv_tot = th.sum(adv_q * (adv_w_final - 1.), dim=1)
        else:
            adv_tot = th.sum(adv_q * adv_w_final, dim=1)
        return adv_tot

    def calc(self, agent_qs, states, actions=None, max_q_i=None, is_v=False):
        if is_v:
            v_tot = self.calc_v(agent_qs)
            return v_tot
        else:
            adv_tot = self.calc_adv(agent_qs, states, actions, max_q_i)
            return adv_tot

    def forward(self, agent_qs, states, actions=None, max_q_i=None, is_v=False):
        bs = agent_qs.size(0)
        nb_timesteps = states.shape[1]
        
        if self.args.use_cnn:
            # states has shape [B, T, H, W, N*C]
            states = states.permute(0,1,4,2,3) # [B, T, N*C, H, W]
            states = states.reshape(*states.shape[:2],self.n_agents,-1,*states.shape[-2:]) # [B, T, N, C, H, W]
            states = states.reshape(-1, *states.shape[-3:]) # [BTN, C, H, W]
            states = self.cnn(states) # [BTN, C']
            states = states.reshape(bs, nb_timesteps, self.n_agents, -1) # [B, T, N, C']
            states = states.reshape(bs, nb_timesteps, -1) # [B, T, N*C']
        w_final, v, attend_mag_regs, head_entropies = self.attention_weight(agent_qs, states, actions)
        w_final = w_final.view(-1, self.n_agents)  + 1e-10
        v = v.view(-1, 1).repeat(1, self.n_agents)
        v /= self.n_agents

        agent_qs = agent_qs.view(-1, self.n_agents)
        agent_qs = w_final * agent_qs + v
        if not is_v:
            max_q_i = max_q_i.view(-1, self.n_agents)
            max_q_i = w_final * max_q_i + v

        y = self.calc(agent_qs, states, actions=actions, max_q_i=max_q_i, is_v=is_v)
        v_tot = y.view(bs, -1, 1)

        return v_tot, attend_mag_regs, head_entropies