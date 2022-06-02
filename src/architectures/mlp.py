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