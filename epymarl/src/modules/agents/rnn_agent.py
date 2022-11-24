import torch.nn as nn
import torch
import torch.nn.functional as F
from modules.agents.cnn import CNN


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        if self.args.use_cnn:
            self.fc1 = CNN(in_channels=input_shape, out_size=args.hidden_dim)
        else:
            self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return torch.zeros(1, self.args.hidden_dim).to(self.fc2.weight.device)

    def forward(self, inputs, hidden_state):
        if self.args.use_cnn:
            inputs = inputs.permute(0,3,1,2)
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        q = self.fc2(h)
        return q, h

