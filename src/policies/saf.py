import numpy as np

import torch
from torch import nn, optim
from torch.distributions.categorical import Categorical

from src.architectures.mlp import MLP
from src.utils import *


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class SAF(nn.Module):
    def __init__(self, observation_space, action_space, params):
        super(SAF, self).__init__()
        
        self.obs_shape = get_obs_shape(observation_space)
        self.action_dim = action_space.n
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
        self.has_continuous_action_space = False
        self.latent_kl = False
        self.pool_policy = False


        #SAF params
        self.N_SK_slots = 3   #attention heads
        self.actor_dims = []
        for i in range(self.n_agents):
            self.actor_dims.append(self.obs_shape[0])
        self.critic_dims = sum(self.actor_dims)
        ###keys for attention mechanisms 
        input_dim=self.critic_dims
        key_dim=self.critic_dims
        self.action_std = 0.06
        self.lr_comm = 0.001
        if self.pool_policy:
            self.n_policy = 10
        else:
            self.n_policy = self.n_agents

        self.critic = nn.ModuleList([MLP(
            np.array(self.obs_shape).prod(), 
            [self.hidden_dim]*self.n_layers, 
            1, 
            std=1.0,
            activation=self.activation) for _ in range(self.n_agents)])

        self.actor = nn.ModuleList([MLP(
            np.array(self.obs_shape).prod(), 
            [self.hidden_dim]*self.n_layers, 
            self.action_dim, 
            std=0.01,
            activation=self.activation) for _ in range(self.n_agents)])
        
        self.comm_policy=Communication_and_policy(input_dim, key_dim, self.N_SK_slots, self.n_agents, self.n_policy, self.action_dim, self.action_std, self.has_continuous_action_space, self.latent_kl, self.pool_policy)
        
        self.optimizer = optim.Adam([
        {'params': self.parameters(), 'lr': self.learning_rate},
        {'params': self.comm_policy.parameters(), 'lr': self.lr_comm}
        ])

    def get_value(self, x):
        """
        Args:
            x: [batch_size, n_agents, obs_shape]
        Returns:
            value: [batch_size, n_agents]
        """
        values = []
        for i in range(self.n_agents): 
            values.append(self.critic[i](x[:,i]))
        values = torch.stack(values, dim=1).squeeze(-1)
        return values

    def get_action_and_value(self, x, actions=None):
        """
        Args:
            x: [batch_size, n_agents, obs_shape]
            action: [batch_size, n_agents]
        Returns:
            action: [batch_size, n_agents]
            logprob: [batch_size, n_agents]
            entropy: [batch_size, n_agents]
            value: [batch_size, n_agents]
        """
        out_actions = []
        logprobs = []
        entropies = []
        for i in range(self.n_agents):
            logits = self.actor[i](x[:,i])
            probs = Categorical(logits=logits)
            if actions is None:
                action = probs.sample()
            else:
                action = actions[:,i]
            logprob = probs.log_prob(action)
            entropy = probs.entropy()

            out_actions.append(action)
            logprobs.append(logprob)
            entropies.append(entropy)
        
        out_actions = torch.stack(out_actions, dim=1)
        logprobs = torch.stack(logprobs, dim=1)
        entropies = torch.stack(entropies, dim=1)
        value = self.get_value(x)
        
        return out_actions, logprobs, entropies, value
    
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
        # flatten the batch
        b_obs = buffer.obs.reshape((-1, self.n_agents) + self.obs_shape)
        b_logprobs = buffer.logprobs.reshape(-1, self.n_agents)
        b_actions = buffer.actions.reshape((-1, self.n_agents))
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

                _, newlogprob, entropy, newvalue = self.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                #_, newlogprob, entropy, newvalue = self.comm_policy.evaluate(b_obs[mb_inds], b_actions.long()[mb_inds],self.PolicyPool, agentID)
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
                # KLLoss computation in Action space
                #_,actions_logprob_commu,_,actions_logprob_nocommu=self.comm_policy.GetActions(old_states_comm.permute(1,0,2),self.PolicyPool,Use_new_policy=True)
                _,actions_logprob_commu,_,actions_logprob_nocommu=self.comm_policy.GetActions(b_obs[mb_inds].permute(1,0,2), self.PolicyPool,Use_new_policy=True)

                KLD=torch.nn.KLDivLoss()
         
                actions_logprob_nocommu_=torch.nn.LogSoftmax()(actions_logprob_nocommu)
                actions_logprob_commu_=F.softmax(actions_logprob_commu)

                KLLoss=KLD(actions_logprob_nocommu_,actions_logprob_commu_)

                self.optimizer.zero_grad()
                (loss+KLLoss).backward()
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



class Communication_and_policy(nn.Module):
    def __init__(self, input_dim, key_dim,N_SK_slots,n_agents,n_policy,action_dim,action_std,has_continuous_action_space, latent_kl, pool_policy):
        super(Communication_and_policy,self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.has_continuous_action_space = has_continuous_action_space
        self.N_SK_slots=N_SK_slots
        #print('action_dim', action_dim)
        self.action_probs_var = torch.full((action_dim,), action_std * action_std).to(self.device)
        self.n_agents=n_agents
        self.latent_kl = latent_kl
        
        self.n_policy=n_policy
        self.pool_policy = pool_policy
        if not self.pool_policy:
            self.n_policy = self.n_agents
        self.input_dim=input_dim
        self.key_dim=key_dim
       
        
        
        ##Vedant's code
        self.num_agents = n_agents
        self.message_projectors = nn.ModuleList([nn.Linear(key_dim, 18) for _ in range(self.num_agents)])
        self.n_saf_slots = N_SK_slots   
        self.n_cycles = 4
        self.saf = nn.Parameter(torch.randn(self.n_saf_slots, 1, 18))
        
        self.write_attention = nn.MultiheadAttention(embed_dim=18, kdim=18, vdim=18, num_heads=2)
        self.TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=18, nhead=2, dim_feedforward=9)
        self.read_attention = nn.MultiheadAttention(embed_dim=key_dim, kdim=18, vdim=18, num_heads=2)

        ###keys for shared knowledge(global workspace)
        self.keys= torch.nn.Parameter(torch.randn(self.n_policy ,1,key_dim))
        self.query_projector_s2=nn.Linear(input_dim, key_dim)###for reading message from sk
    
        self.tanh = nn.Tanh()
        if self.latent_kl:
            self.encoder = nn.Sequential(
                                nn.Linear(int(2*key_dim), 64),
                                nn.Tanh(),
                                nn.Linear(64, 64),
                                nn.Tanh(),
                                nn.Linear(64, 32),
                            ).to(self.device)

            self.encoder_prior = nn.Sequential(
                                nn.Linear(key_dim, 64),
                                nn.Tanh(),
                                nn.Linear(64, 64),
                                nn.Tanh(),
                                nn.Linear(64, 32),
                            ).to(self.device)

        self.previous_state = torch.randn(5, 1, key_dim).to(self.device)

    def write_to(self, encoded_states, evaluate, agentID=0):
        if evaluate:
            message = self.message_projectors[agentID](encoded_states[0].unsqueeze(0)) 
        else:
            message = [self.message_projectors[i](encoded_states[i]) for i in range(self.num_agents)]
            message = torch.stack(message, dim=0)
        #print("message_dim",message.shape)
        batch_size = message.shape[1]

        saf = self.saf.repeat(1, batch_size, 1)

        for _ in range(self.n_cycles):
            saf, _ = self.write_attention(saf, message, message)
            saf = self.TransformerEncoderLayer(saf)

        return saf

    def read_from(self, saf, encoded_states):
        #encoded_states = torch.stack(encoded_states, dim=0)
        #print('encoded_states_shape= ',encoded_states.shape)
        message, _ = self.read_attention(encoded_states, saf, saf)
        encoded_states = encoded_states + message
        #print('message_shape= ',message.shape)
        #print('encoded_states_shape= ',encoded_states.shape)
        return encoded_states

    def forward(self, states, evaluate=False):

        states=states.to(self.device)
        encoded_states=self.query_projector_s2(states)#state_encoded
        saf = self.write_to(encoded_states, evaluate)
        input_to_policy = self.read_from(saf, encoded_states)

        return input_to_policy, encoded_states#also return the original state for KL

    def information_bottleneck(self,s_attention, s_agent, s_agent_previous_t):
        z_ = self.encoder(torch.cat((s_attention,s_agent), dim=2))
        mu, sigma = z_.chunk(2, dim=2)
        z = mu + sigma * torch.randn_like(sigma)
        z_prior = self.encoder_prior(s_agent_previous_t)
        mu_prior, sigma_prior = z_prior.chunk(2, dim=2)
        KL = 0.5 * torch.sum(((mu - mu_prior) ** 2 + sigma ** 2)/(sigma_prior ** 2) + torch.log(1e-8 + (sigma_prior ** 2)/(sigma ** 2)) - 1) / np.prod(torch.cat((s_attention,s_agent), dim=2).shape)
        return z, KL




    def compute_step_single_policy(self, PolicyPool, state, z, Use_new_policy):

        all_actions_logprob_ouput=[]
        all_actions_ouput=[]
        all_values_ouput=[]

        
        for i in range(self.n_agents):
            if self.latent_kl:
                if Use_new_policy:
                    #new policy is used during training 
                    action_vec, action_logprob_vec=PolicyPool[i].policy.act(torch.cat((state[i].unsqueeze(0),z), dim=2))
                    values_output_vec=PolicyPool[i].policy.critic(torch.cat((state[i].unsqueeze(0),z), dim=2)).squeeze(2)
                else:
                    #when picking actions , use old poliy
                    action_vec, action_logprob_vec=PolicyPool[i].policy_old.act(torch.cat((state[i].unsqueeze(0),z), dim=2))
                    values_output_vec=PolicyPool[i].policy.critic(torch.cat((state[i].unsqueeze(0),z), dim=2)).squeeze(2)
            else: 
                if Use_new_policy:
                    #new policy is used during training 
                    action_vec, action_logprob_vec=PolicyPool[i].policy.act(state[i].unsqueeze(0))
                    values_output_vec=PolicyPool[i].policy.critic(state[i].unsqueeze(0)).squeeze(2)
                else:
                    #when picking actions , use old poliy
                    action_vec, action_logprob_vec=PolicyPool[i].policy_old.act(state[i].unsqueeze(0))
                    values_output_vec=PolicyPool[i].policy.critic(state[i].unsqueeze(0)).squeeze(2)
                    #values_output_vec=PolicyPool[i].policy.critic(state[i]).squeeze(2)

            action_vec=action_vec.unsqueeze(2)
            action_logprob_vec=action_logprob_vec.unsqueeze(2)
            values_output_vec=values_output_vec.unsqueeze(2)
            #values_output_vec=values_output_vec.permute(1,0,2)
            all_actions_ouput.append(action_vec)
            all_actions_logprob_ouput.append(action_logprob_vec)
            all_values_ouput.append(values_output_vec)
            
     
        all_actions_logprob_ouput=torch.cat(all_actions_logprob_ouput,0)
        all_actions_ouput=torch.cat(all_actions_ouput,0).float()#(Bz, N_agents , N_behavior)
        all_values_ouput=torch.cat(all_values_ouput,0)# (bz,N_agents,N_behavior)

        return all_actions_ouput.long(), all_actions_logprob_ouput, all_values_ouput



    def compute_step_with_pool_policies(self, PolicyPool, state, z, Use_new_policy):

        
        all_policy_actions_logprob_ouput=[]
        all_policy_actions_ouput=[]
        all_policy_values_ouput=[]
        all_actions_ouput = []
        all_actions_logprob_ouput = []
        all_values_ouput = []
        #print(self.n_policy)
        for i in range(self.n_agents):
            for j in range(self.n_policy):
                if self.latent_kl:
                    if Use_new_policy:
                        #new policy is used during training 
                        action_vec, action_logprob_vec=PolicyPool[i].policy.act(torch.cat((state,z), dim=2))
                        values_output_vec=PolicyPool[i].policy.critic(torch.cat((state,z), dim=2)).squeeze(2)
                    else:
                        #when picking actions , use old poliy
                        action_vec, action_logprob_vec=PolicyPool[i].policy_old.act(torch.cat((state,z), dim=2))
                        values_output_vec=PolicyPool[i].policy.critic(torch.cat((state,z), dim=2)).squeeze(2)
                else: 
                    if Use_new_policy:
                        #new policy is used during training 
                        action_vec, action_logprob_vec=PolicyPool[j].policy.act(state[i].unsqueeze(0))
                        values_output_vec=PolicyPool[j].policy.critic(state[i].unsqueeze(0)).squeeze(2)
                    else:
                        #when picking actions , use old poliy
                        action_vec, action_logprob_vec=PolicyPool[j].policy_old.act(state[i].unsqueeze(0))
                        values_output_vec=PolicyPool[j].policy.critic(state[i].unsqueeze(0)).squeeze(2)
                        #values_output_vec=PolicyPool[i].policy.critic(state[i]).squeeze(2)
          
                #if self.has_continuous_action_space:
                #    action_vec=action_vec.permute(1,0,2).unsqueeze(2)
                #else:
                #    action_vec=action_vec.permute(1,0).unsqueeze(2)
                #action_logprob_vec=action_logprob_vec.permute(1,0).unsqueeze(2)
                action_vec=action_vec.unsqueeze(2)
                action_logprob_vec=action_logprob_vec.unsqueeze(2)
                values_output_vec=values_output_vec.unsqueeze(2)
                #print('action_vec_shape = ', action_vec.shape)
                #print('action_logprob_vec_shapr = ',action_logprob_vec.shape)
                #print('values_vec_shape = ',values_output_vec.shape)
                #values_output_vec=values_output_vec.permute(1,0,2)
                all_policy_actions_ouput.append(action_vec)
                all_policy_actions_logprob_ouput.append(action_logprob_vec)
                all_policy_values_ouput.append(values_output_vec) 
            all_policy_actions_logprob_ouput_=torch.cat(all_policy_actions_logprob_ouput,2)
            all_policy_actions_ouput_=torch.cat(all_policy_actions_ouput,2).float()#(Bz, N_agents , N_behavior)
            all_policy_values_ouput_=torch.cat(all_policy_values_ouput,2)# (bz,N_agents,N_behavior)

            #print('all_policy_actions_ouput_shape = ', all_policy_actions_ouput_.shape)
            #print('all_policy_actions_logprob_ouput_shapr = ',all_policy_actions_logprob_ouput_.shape)
            #print('all_policy_values_ouput__shape = ',all_policy_values_ouput_.shape)

            #if self.has_continuous_action_space:
            #    action_vec=action_vec.permute(1,0,2).unsqueeze(2)
            #else:
            #    action_vec=action_vec.permute(1,0).unsqueeze(2)
            #action_logprob_vec=action_logprob_vec.permute(1,0).unsqueeze(2)
          
            #values_output_vec=values_output_vec.permute(1,0,2)
            all_actions_ouput.append(all_policy_actions_ouput_)
            all_actions_logprob_ouput.append(all_policy_actions_logprob_ouput_)
            all_values_ouput.append(all_policy_values_ouput_)
            all_policy_actions_logprob_ouput=[]
            all_policy_actions_ouput=[]
            all_policy_values_ouput=[]

        
    
        all_actions_logprob_ouput_=torch.cat(all_actions_logprob_ouput,0)
        all_actions_ouput_=torch.cat(all_actions_ouput,0).float()#(Bz, N_agents , N_behavior)
        all_values_ouput_=torch.cat(all_values_ouput,0)# (bz,N_agents,N_behavior)
        #print('all_actions_ouput_shape = ', all_actions_ouput_.shape)
        #print('all_actions_logprob_ouput_shapr = ',all_actions_logprob_ouput_.shape)
        #print('all_values_ouput__shape = ',all_values_ouput_.shape)

        return all_actions_ouput_.permute(1,0,2), all_actions_logprob_ouput_.permute(1,0,2), all_values_ouput_.permute(1,0,2)


    def evaluate_step_with_pool_policies(self, PolicyPool, state, z, Use_new_policy, agentID):

        
        all_policy_actions_logprob_ouput=[]
        all_policy_actions_ouput=[]
        all_policy_values_ouput=[]
       
        for i in range(self.n_policy):
            if self.latent_kl:
                if Use_new_policy:
                    #new policy is used during training 
                    action_vec, action_logprob_vec=PolicyPool[i].policy.act(torch.cat((state,z), dim=2))
                    values_output_vec=PolicyPool[i].policy.critic(torch.cat((state,z), dim=2)).squeeze(2)
                else:
                    #when picking actions , use old poliy
                    action_vec, action_logprob_vec=PolicyPool[i].policy_old.act(torch.cat((state,z), dim=2))
                    values_output_vec=PolicyPool[i].policy.critic(torch.cat((state,z), dim=2)).squeeze(2)
            else: 
                if Use_new_policy:
                    #new policy is used during training 
                    action_vec, action_logprob_vec=PolicyPool[i].policy.act(state[agentID].unsqueeze(0))
                    values_output_vec=PolicyPool[i].policy.critic(state[agentID].unsqueeze(0)).squeeze(2)
                else:
                    #when picking actions , use old poliy
                    action_vec, action_logprob_vec=PolicyPool[i].policy_old.act(state[agentID].unsqueeze(0))
                    values_output_vec=PolicyPool[i].policy.critic(state[agentID].unsqueeze(0)).squeeze(2)
                    #values_output_vec=PolicyPool[i].policy.critic(state[i]).squeeze(2)
        
            #if self.has_continuous_action_space:
            #    action_vec=action_vec.permute(1,0,2).unsqueeze(2)
            #else:
            #    action_vec=action_vec.permute(1,0).unsqueeze(2)
            #action_logprob_vec=action_logprob_vec.permute(1,0).unsqueeze(2)
            action_vec=action_vec.unsqueeze(2)
            action_logprob_vec=action_logprob_vec.unsqueeze(2)
            values_output_vec=values_output_vec.unsqueeze(2)
            #print('action_vec_shape = ', action_vec.shape)
            #print('action_logprob_vec_shapr = ',action_logprob_vec.shape)
            #print('values_vec_shape = ',values_output_vec.shape)
            #values_output_vec=values_output_vec.permute(1,0,2)
            all_policy_actions_ouput.append(action_vec)
            all_policy_actions_logprob_ouput.append(action_logprob_vec)
            all_policy_values_ouput.append(values_output_vec) 
        all_policy_actions_logprob_ouput_=torch.cat(all_policy_actions_logprob_ouput,2)
        all_policy_actions_ouput_=torch.cat(all_policy_actions_ouput,2).float()#(Bz, N_agents , N_behavior)
        all_policy_values_ouput_=torch.cat(all_policy_values_ouput,2)# (bz,N_agents,N_behavior)
        #print('all_policy_actions_ouput_shape = ', all_policy_actions_ouput_.shape)
        #print('all_policy_actions_logprob_ouput_shapr = ',all_policy_actions_logprob_ouput_.shape)
        #print('all_policy_values_ouput__shape = ',all_policy_values_ouput_.shape)
        #if self.has_continuous_action_space:
        #    action_vec=action_vec.permute(1,0,2).unsqueeze(2)
        #else:
        #    action_vec=action_vec.permute(1,0).unsqueeze(2)
        #action_logprob_vec=action_logprob_vec.permute(1,0).unsqueeze(2)
        
        #values_output_vec=values_output_vec.permute(1,0,2)

        return all_policy_actions_ouput_, all_policy_actions_logprob_ouput_, all_policy_values_ouput_

    def attention_step(self, query, all_actions_ouput, all_actions_logprob_ouput, all_values_ouput):
        attention_score=[]
        N_agents, T, Embsz=query.shape
        for j in range(T):
            attention_score_vec=torch.bmm(query[:,j,:].unsqueeze(1).permute(1,0,2), self.keys.permute(1, 2,0))/torch.sqrt(torch.tensor(self.key_dim).float()) 
            attention_score.append(attention_score_vec)
        attention_score=torch.cat(attention_score,0)#(bz,N_agents, Embsz)
    
        attention_score_sm=attention_score.clone()
    
        attention_score=nn.functional.gumbel_softmax(attention_score,hard=True,dim=2)#(Bz, N_agents , N_behavior)
        all_actions_logprob_ouput_all=[]
        all_actions_ouput_all=[]
        all_values_ouput_all=[]

        #print(attention_score.shape)
        #print(attention_score[0,:,:].unsqueeze(1).shape)
        #print(all_actions_logprob_ouput.shape)
        #print(all_actions_logprob_ouput[0,:,:].unsqueeze(2).shape)
        #print(all_actions_ouput.shape)
        #print(all_actions_ouput[0,:,:].unsqueeze(2).shape)
        for j in range(T):
            all_actions_logprob_vec=torch.bmm(attention_score[j,:,:].unsqueeze(1),all_actions_logprob_ouput[j,:,:].unsqueeze(2))

            all_actions_logprob_ouput_all.append(all_actions_logprob_vec)#element shape (N_agent,1,1)
            if self.has_continuous_action_space:
                all_actions_ouput_vec=torch.bmm(all_actions_ouput[j,:,:,:].permute(0,2,1), attention_score[j,:,:].unsqueeze(2))
            else:
                all_actions_ouput_vec=torch.bmm(attention_score[j,:,:].unsqueeze(1),all_actions_ouput[j,:,:].unsqueeze(2))
            all_actions_ouput_all.append(all_actions_ouput_vec)
            all_values_ouput_vec=torch.bmm(attention_score[j,:,:].unsqueeze(1),all_values_ouput[j,:,:].unsqueeze(2))
            all_values_ouput_all.append(all_values_ouput_vec)
        actions_logprob_pred=torch.cat(all_actions_logprob_ouput_all,1).squeeze(2)#shape (N_agent,T)
        if self.has_continuous_action_space:
            actions_pred=0.05*self.tanh(torch.cat(all_actions_ouput_all,2).squeeze(2)) #shape (N_agent,T)
        else:
            actions_pred=torch.cat(all_actions_ouput_all,1).squeeze(2).long() #shape (N_agent,T)
        values_pred=torch.cat(all_values_ouput_all,1).squeeze(2)#shape (N_agent,T)
        return actions_pred, actions_logprob_pred, values_pred


    def Run_policy(self, state, PolicyPool, Use_new_policy=False, z=0, evaluate=False, agentID=0):
 
        query=state
        
        if self.pool_policy:
            all_actions_ouput, all_actions_logprob_ouput, all_values_ouput = self.compute_step_with_pool_policies(PolicyPool, state, z, Use_new_policy)
            actions_pred, actions_logprob_pred, values_pred = self.attention_step(query, all_actions_ouput, all_actions_logprob_ouput, all_values_ouput )
        else:
            actions_pred, actions_logprob_pred, values_pred = self.compute_step_single_policy(PolicyPool, state, z, Use_new_policy)
        

        return actions_pred,actions_logprob_pred,values_pred
       

    def GetActions(self,state,PolicyPool,Use_new_policy=False):

        query,query_internalonly=self.forward(state)#run the communication
        #print('query_shape = ', query.shape)

        if self.latent_kl:
            z, KL = self.information_bottleneck(query, query_internalonly, self.previous_state)
            self.previous_state = query_internalonly.detach()
            #with inter-agent communication
            actions_pred_communicated,actions_logprob_pred_communicated,_=self.Run_policy(query,PolicyPool, Use_new_policy, z)
            #with without inter-agent communicaiton
            actions_pred_nocommunication,actions_logprob_pred_nocommunication,_=self.Run_policy(query_internalonly,PolicyPool, Use_new_policy, z)
            return actions_pred_communicated,actions_logprob_pred_communicated,actions_pred_nocommunication,actions_logprob_pred_nocommunication, KL

        else:
            #with inter-agent communication
            actions_pred_communicated,actions_logprob_pred_communicated,_=self.Run_policy(query, PolicyPool, Use_new_policy)
            #with without inter-agent communicaiton
            actions_pred_nocommunication,actions_logprob_pred_nocommunication,_=self.Run_policy(query_internalonly,PolicyPool, Use_new_policy)
            return actions_pred_communicated,actions_logprob_pred_communicated,actions_pred_nocommunication,actions_logprob_pred_nocommunication

    def evaluate(self, state, action, PolicyPool, agentID):
        evaluate=True
        states=state.to(self.device)
        encoded_states=self.query_projector_s2(states)
        saf = self.write_to(encoded_states, evaluate, agentID)
        query = self.read_from(saf, encoded_states)
 
        if self.latent_kl:
            #new policy is used during training 
            z, _ = self.information_bottleneck(query, encoded_states, self.previous_state)
            if self.pool_policy:
                actions_pred, actions_logprob_pred, values_pred = self.evaluate_step_with_pool_policies(PolicyPool, state, z, True, agentID)
                actions_pred, actions_logprob_pred, values_pred = self.attention_step(actions_pred, actions_logprob_pred, values_pred)
            else:
                actions_pred, actions_logprob_pred=PolicyPool[agentID].policy.act(torch.cat((state,z), dim=2))
                values_pred=PolicyPool[agentID].policy.critic(torch.cat((state,z), dim=2)).squeeze(2)
        
        else: 
            #new policy is used during training 
            actions_pred, actions_logprob_pred=PolicyPool[agentID].policy.act(state)
            values_pred=PolicyPool[agentID].policy.critic(state).squeeze(2)
            #print('actions_logprob_pred_shape = ', actions_logprob_pred.shape)
            #print('actions_pred_shape = ', actions_pred.shape)

        if self.has_continuous_action_space:
            cov_mat = torch.diag(self.action_probs_var).unsqueeze(dim=0)
            dist = MultivariateNormal(actions_pred.permute(2,0,1), cov_mat)
            action_selected_logprobs = dist.log_prob(action.permute(1,0,2))
        else:
            action_probs=torch.exp(actions_logprob_pred) ###remove log
            dist = Categorical(action_probs)
            #print(action_probs.shape)
            #print(action.shape)
            action_selected_logprobs = dist.log_prob(action)
        
        
        dist_entropy = dist.entropy()

        return action_selected_logprobs,values_pred,dist_entropy







