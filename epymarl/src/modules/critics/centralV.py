import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from modules.agents.cnn import CNN


class CentralVCritic(nn.Module):
    def __init__(self, scheme, args):
        super(CentralVCritic, self).__init__()

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
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

    def forward(self, batch, t=None):
        inputs, _, _ = self._build_inputs(batch, t=t)
        b, t_, n = inputs.shape[:3]
        if self.args.use_cnn:
            inputs = rearrange(inputs, "b t n h w c -> b t n c h w")
            inputs = rearrange(inputs, "... c h w -> (...) c h w")
            x = F.relu(self.fc1(inputs))
            x = rearrange(x, "(b t n) c -> b t n c", b=b, t=t_, n=n)
        else:
            x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

    def _build_inputs(self, batch, t=None):

        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        h, w, c = batch["obs"].shape[-3:]
        inputs = []
        # state
        if self.args.use_cnn:
            inputs.append(repeat(batch["state"][:, ts], "b t h w c -> b t n h w c", n=self.n_agents))
        else:
            inputs.append(batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # observations
        if self.args.obs_individual_obs:
            if self.args.use_cnn:
                
                obs = batch["obs"][:, ts].view(bs, max_t, self.n_agents, h, w, c)
                inputs.append(obs)
            else:
                inputs.append(batch["obs"][:, ts].view(bs, max_t, -1).unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # last actions
        if self.args.obs_last_action:
            
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, :1]).view(bs, max_t, 1, -1))
            elif isinstance(t, int):
                inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1))
            else:
                last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
                last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
                inputs.append(last_actions)
        
        z = th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1) 
        if self.args.use_cnn:
            if self.args.use_cnn:
                z = z.unsqueeze(3).unsqueeze(4)
                z = z.repeat(1,1,1,h,w,1)
            inputs.append(z)

        inputs = th.cat(inputs, dim=-1)
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
                input_shape += scheme["obs"]["vshape"][-1] * self.n_agents
            else:
                input_shape += scheme["obs"]["vshape"] * self.n_agents
        # last actions
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
        input_shape += self.n_agents
        return input_shape