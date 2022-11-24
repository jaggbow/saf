import numpy as np

import torch
from torch import nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


activations = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "gelu": nn.GELU(),
    "swish": nn.SiLU(),
}


class GRURNN(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        std=np.sqrt(2),
        bias_const=0.0,
        activation="tanh",
    ):

        super().__init__()
        assert activation in ["relu", "tanh", "gelu", "swish"]

        self.fc1 = nn.Sequential(
            layer_init(nn.Linear(in_dim, hidden_dim)), activations[activation]
        )
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Sequential(
            activations[activation],
            layer_init(nn.Linear(hidden_dim, out_dim), std=std, bias_const=bias_const),
        )

    def forward(self, x, hidden_state=None):
        out = self.fc1(x)
        hidden_state = self.rnn(out, hidden_state)
        out = self.fc2(hidden_state)
        return out, hidden_state
