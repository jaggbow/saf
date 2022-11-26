import torch.nn as nn
from modules.agents.rnn_agent import RNNAgent
import torch as th
from einops import rearrange


import numpy as np

import torch
from torch import nn

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

activations = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'gelu': nn.GELU(),
    'swish': nn.SiLU()
}
class MLP(nn.Module):
    def __init__(
        self, 
        in_dim, 
        hidden_list, 
        out_dim, 
        std=np.sqrt(2),
        bias_const=0.0,
        activation='tanh'):
        
        super().__init__()
        assert activation in ['relu', 'tanh', 'gelu', 'swish']
        
        self.layers = nn.ModuleList()
        self.layers.append(layer_init(nn.Linear(in_dim, hidden_list[0])))
        self.layers.append(activations[activation])
        
        for i in range(len(hidden_list)-1):
            self.layers.append(layer_init(nn.Linear(hidden_list[i], hidden_list[i+1])))
            self.layers.append(activations[activation])
        self.layers.append(layer_init(nn.Linear(hidden_list[-1],out_dim), std=std, bias_const=bias_const))
    
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

######RNN agents with pool of Q functions
class RNNPPAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNPPAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.input_shape = input_shape

        self.n_mechanisms=args.n_mechanisms
        self.agents = th.nn.ModuleList([RNNAgent(input_shape, args) for _ in range(self.n_mechanisms)])

        ###pool of mechanisms related
        key_dim=32 
        hidden_dim=32
        n_layers=2
        activation='relu'

        self.policy_query_projector = MLP(
        128,
        [hidden_dim] * n_layers,
        key_dim,
        std=1.0,
        activation=activation,
        )  # for query-key attention pick policy form pool

        self.mechanism_keys = torch.nn.Parameter(
            torch.randn(self.n_mechanisms, 1, key_dim)
        )

        self.mechanism_attn = nn.MultiheadAttention(
        embed_dim=key_dim, num_heads=1, batch_first=False
            )

        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def init_hidden(self):
        # make hidden states on same device as model
        return th.cat([self.agents[0].init_hidden() for a in range(self.n_agents)])#these initialization are zeros

    def forward(self, inputs, hidden_state):
        hiddens = []
        qs = []

        
        if inputs.size(0) == self.n_agents:
            for i in range(self.n_agents):
                q, h = self.agents[i](inputs[i].unsqueeze(0), hidden_state[:, i])
                hiddens.append(h)
                qs.append(q)
            return th.cat(qs), th.cat(hiddens).unsqueeze(0)
        else:
            inputs = rearrange(inputs, "(b n) ... -> b n ...", n=self.n_agents)
            #input shape [bsz, n_agents, 28, 28, 12]
            #hidden shape [bsz,2,128]
            bsz,n_agents,_=hidden_state.shape
   
            
            for i in range(self.n_agents):
                hs_agent=[]
                qs_agent=[]
                for j in range(self.n_mechanisms):
                    q, h = self.agents[j](inputs[:, i], hidden_state[:, i])
                    hs_agent.append(h.unsqueeze(1).unsqueeze(1))
                    qs_agent.append(q.unsqueeze(1).unsqueeze(1))
                
                hs_agent=torch.cat(hs_agent,dim=1)#(bsz,n_mechanism,1,embsz)
                qs_agent=torch.cat(qs_agent,dim=1)#(bsz,n_mechanism,1,action_space)
                
                hiddens.append(hs_agent)
                qs.append(qs_agent)

            
            hiddens=torch.cat(hiddens,dim=2)#(bsz,n_mechanism,n_agents,embedsize)
            qs=torch.cat(qs,dim=2)#(bsz,n_mechanism,n_agents,action_space)
            #h shape [8, 1, 128] 
            #q shape [8,1,7]

            #calculate the attention score based on hiddent states (query) and key of the mechanism
            attention_score=self.Run_mechanism_attention(hidden_state) #(Bz, N_agents , N_mechanism)
            print("attention score",attention_score.shape)
            #multiply the attention score and qs, hidden

            hiddens = torch.einsum(
            "bjie,bij->bie", hiddens, attention_score
            )  # bsz x N_agentsX embsize
            qs = torch.einsum(
            "bjia,bij->bia", qs, attention_score
            )  # bsz x N_agentsX action space


            qs=qs.view(bsz*self.n_agents,-1)

        
            return qs, hiddens

    def cuda(self, device="cuda:0"):
        # for a in self.agents:
        #     a.cuda(device=device)

        # self.mechanism_attn.cuda(device=device)
        # self.policy_query_projector.cuda(device=device)
        # self.mechanism_keys.cuda(device=device)
        for a in self.agents:
            a.to(self.device)

        self.mechanism_attn.to(self.device)
        self.policy_query_projector.to(self.device)
        self.mechanism_keys.to(self.device)

    def Run_mechanism_attention(self, state):
        """
        state has shape  [bsz, n_agents, embsz]
        """
        state = state.permute(1, 0, 2)  # (N_agents,bsz,embsize)
        
        state = state
        # how to pick rules and if they are shared across agents
        query = self.policy_query_projector(state)
        N_agents, bsz, Embsz = query.shape

        keys = self.mechanism_keys.repeat(1, bsz, 1)  # n_ploicies,bsz,Embsz,

        _, attention_score = self.mechanism_attn(query, keys, keys)

        attention_score = nn.functional.gumbel_softmax(
            attention_score, tau=1, hard=True, dim=2
        )  # (Bz, N_agents , N_mechanism)

        return attention_score
