import torch as th
import torch.nn as nn
import torch.nn.functional as F
from modules.critics.mlp import MLP

from modules.agents.cnn import CNN


class CentralVCriticNS(nn.Module):
    def __init__(self, scheme, args):
        super(CentralVCriticNS, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"

        # Set up network layers
        if self.args.use_cnn:
            self.fc1 = CNN(in_channels=input_shape, out_size=args.hidden_dim)
        else:
            self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.critics = nn.ModuleList([MLP(args.hidden_dim, args.hidden_dim, 1) for _ in range(self.n_agents)])

    def forward(self, batch, t=None):
        inputs, bs, max_t = self._build_inputs(batch, t=t)
        qs = []
        if self.args.use_cnn:
            inputs = inputs.permute(0,3,1,2)
        for i in range(self.n_agents):
            q = self.fc1(inputs)
            q = self.critics[i](q)
            qs.append(q.view(bs, max_t, 1, -1))
        q = th.cat(qs, dim=2)
        return q

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        h, w, c = batch["obs"].shape[-3:]
        inputs = []
        # state
        inputs.append(batch["state"][:, ts])

        # observations
        if self.args.obs_individual_obs:
            if self.args.use_cnn:
                
                obs = batch["obs"][:, ts].view(bs, max_t, self.n_agents, h, w, c)
                obs = obs.permute(0,1,3,4,2,5)
                obs = obs.reshape(0,1,3,4,-1)
                inputs.append(obs)
            else:
                inputs.append(batch["obs"][:, ts].view(bs, max_t, -1))

        if self.args.obs_last_action:
            # last actions
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1))
            elif isinstance(t, int):
                inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1))
            else:
                last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
                last_actions = last_actions.view(bs, max_t, 1, -1)
                inputs.append(last_actions)

        inputs = th.cat([x.reshape(-1, *x.shape[2:]) for x in inputs], dim=-1)
        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        # state
        if self.args.use_cnn:
            input_shape = scheme["state"]["vshape"][-1]
        else:
            input_shape = scheme["state"]["vshape"]
        # observations
        if self.args.obs_individual_obs:
            if self.args.use_cnn:
                input_shape += scheme["obs"]["vshape"][-1]
            else:
                input_shape += scheme["obs"]["vshape"]
        # last actions
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents

        return input_shape

    def parameters(self):
        params = list(self.fc1.parameters())
        for i in range(self.n_agents):
            params += list(self.critics[i].parameters())
        return params

    def state_dict(self):
        return [a.state_dict() for a in self.critics]

    def load_state_dict(self, state_dict):
        for i, a in enumerate(self.critics):
            a.load_state_dict(state_dict[i])

    def cuda(self):
        self.fc1.cuda()
        for c in self.critics:
            c.cuda()
