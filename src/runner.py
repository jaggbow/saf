import numpy as np
import os
from os.path import expanduser, expandvars
import time
from datetime import datetime
from pathlib import Path

from gym import spaces

from tqdm import tqdm
import comet_ml

import torch


class PGRunner:
    def __init__(self, train_env, eval_env, env_family, policy, buffer, params, device):

        self.policy_type = policy.type
        self.total_timesteps = params.total_timesteps
        self.batch_size = params.rollout_threads * params.env_steps
        self.latent_kl = params.latent_kl
        self.rollout_threads = params.rollout_threads
        self.n_agents = params.n_agents
        self.env_steps = params.env_steps
        self.lr_decay = params.lr_decay
        self.env_family = env_family
        self.eval_episodes = params.eval_episodes
        self.use_comet = True if params.comet else False
        self.checkpoint_dir = params.checkpoint_dir
        self.save_dir = (
            Path(expandvars(expanduser(str(os.getcwd())))).resolve()
            / os.environ["SLURM_JOB_ID"]
        )
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(train_env.observation_space, spaces.Dict):
            self.observation_space = train_env.observation_space["observation"]
        elif isinstance(train_env.observation_space, tuple):
            self.observation_space = train_env.observation_space[0]
        else:
            self.observation_space = train_env.observation_space

        if isinstance(train_env.action_space, tuple):
            self.action_space = train_env.action_space[0]
        else:
            self.action_space = train_env.action_space

        test_mode = params.test_mode

        self.train_env = train_env
        self.eval_env = eval_env
        self.buffer = buffer
        self.policy = policy
        self.device = device
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        print(os.getcwd())

        if self.checkpoint_dir is not None and not test_mode:
            print("Resuming training from", self.checkpoint_dir)
            self.load_checkpoints(self.checkpoint_dir)
            if self.use_comet:
                api_key_path = Path(self.checkpoint_dir) / Path("apy_key.txt")

                with open(api_key_path) as f:
                    self.exp_api_key = f.readlines()
                # Check to see if there is a key in environment:
                EXPERIMENT_KEY = os.environ.get(
                    "COMET_EXPERIMENT_KEY", self.exp_api_key[0]
                )

                # First, let's see if we continue or start fresh:
                if EXPERIMENT_KEY is not None:
                    # There is one, but the experiment might not exist yet:
                    api = comet_ml.API()  # Assumes API key is set in config/env
                    try:
                        api_experiment = api.get_experiment_by_key(EXPERIMENT_KEY)
                    except Exception:
                        api_experiment = None
                    if api_experiment is not None:
                        CONTINUE_RUN = True
                        # We can get the last details logged here, if logged:
                        self.step = int(
                            api_experiment.get_parameters_summary("curr_step")[
                                "valueCurrent"
                            ]
                        )

        else:
            if test_mode:
                print("Loading model from", self.checkpoint_dir)
                self.load_checkpoints(self.checkpoint_dir)
            if self.use_comet:
                self.exp = comet_ml.Experiment(project_name=params.comet.project_name)
                self.exp.set_name(params.comet.experiment_name)
                self.exp_key = self.exp.get_key()

    def env_reset(self, mode="train"):
        """
        Resets the environment.
        Returns:
        obs: [rollout_threads, n_agents, obs_shape]
        """
        if mode == "train":
            obs, state, act_masks = self.train_env.reset()
        elif mode == "eval":
            obs, state, act_masks = self.eval_env.reset()
        obs = torch.from_numpy(obs).to(
            self.device
        )  # [rollout_threads*n_agents, obs_shape]
        obs = obs.reshape(
            (-1, self.n_agents) + obs.shape[1:]
        )  # [rollout_threads, n_agents, obs_shape]

        state = torch.from_numpy(state).to(
            self.device
        )  # [rollout_threads*n_agents, state_shape]
        state = state.reshape(
            (-1, self.n_agents) + state.shape[1:]
        )  # [rollout_threads, n_agents, state_shape]

        if type(act_masks) != type(None):
            act_masks = torch.from_numpy(act_masks).to(
                self.device
            )  # [rollout_threads*n_agents, action_shape]
            act_masks = act_masks.reshape(
                (-1, self.n_agents) + act_masks.shape[1:]
            )  # [rollout_threads, n_agents, action_shape]

        return obs, state, act_masks

    def env_step(self, action, mode="train"):
        """
        Does a step in the defined environment using action.
        Args:
            action: [rollout_threads, n_agents] for Discrete type and [rollout_threads, n_agents, action_dim] for Box type
        """

        if self.action_space.__class__.__name__ == "Box":
            action_ = action.reshape(-1, action.shape[-1]).cpu().numpy()
        elif self.action_space.__class__.__name__ == "Discrete":
            action_ = action.reshape(-1).cpu().numpy()
        else:
            raise NotImplementedError

        if mode == "train":
            obs, state, act_masks, reward, done, info = self.train_env.step(action_)
        elif mode == "eval":
            obs, state, act_masks, reward, done, info = self.eval_env.step(action_)

        obs = (
            torch.Tensor(obs)
            .reshape((-1, self.n_agents) + obs.shape[1:])
            .to(self.device)
        )  # [rollout_threads, n_agents, obs_shape]
        state = (
            torch.Tensor(state)
            .reshape((-1, self.n_agents) + state.shape[1:])
            .to(self.device)
        )  # [rollout_threads, n_agents, state_shape]
        if type(act_masks) != type(None):
            act_masks = (
                torch.Tensor(act_masks)
                .reshape((-1, self.n_agents) + act_masks.shape[1:])
                .to(self.device)
            )  # [rollout_threads, n_agents, act_shape]
        done = (
            torch.Tensor(done).reshape((-1, self.n_agents)).to(self.device)
        )  # [rollout_threads, n_agents]

        reward = (
            torch.Tensor(reward).reshape((-1, self.n_agents)).to(self.device)
        )  # [rollout_threads, n_agent]

        return obs, state, act_masks, reward, done, info

    def run(self):

        global_step = 0
        start_time = time.time()

        obs, state, act_masks = self.env_reset()
        next_done = torch.zeros((self.rollout_threads, self.n_agents)).to(self.device)

        num_updates = self.total_timesteps // self.batch_size
        best_return = -1e9

        if self.latent_kl:
            ## old_observation - shifted tensor (the zero-th obs is assumed to be equal to the first one)
            obs_old = obs.clone()
            obs_old[1:] = obs_old.clone()[:-1]

            if self.policy_type == "conv":
                bs = obs_old.shape[0]
                n_ags = obs_old.shape[1]

                obs_old = obs_old.reshape((-1,) + self.policy.obs_shape)
                obs_old = self.policy.conv(obs_old)
                obs_old = obs_old.reshape(bs, n_ags, self.policy.input_shape)
        else:
            obs_old = None

        if not os.path.exists(self.save_dir):

            os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))
            os.makedirs(self.save_dir, exist_ok=True)
            api_key_path = Path(self.save_dir) / Path("apy_key.txt")

            with open(api_key_path, "w") as f:
                f.write(self.exp_key)

        for update in range(1, num_updates + 1):
            if self.lr_decay:
                self.policy.update_lr(update, num_updates)

            total_rewards = 0

            nb_games = np.ones(self.rollout_threads)
            nb_wins = np.zeros(self.rollout_threads)

            if self.policy.use_rnn:
                self.policy.init_hidden()
                hidden_state = self.policy.hidden_state
                if self.policy.use_policy_pool:
                    for j in range(self.policy.n_policy):
                        for i in range(self.n_agents):
                            hidden_state[j][i] = hidden_state[j][i].repeat(
                                obs.shape[0], 1
                            )
                else:
                    for i in range(self.n_agents):
                        hidden_state[i] = hidden_state[i].repeat(obs.shape[0], 1)
            for step in range(self.env_steps):
                global_step += self.rollout_threads

                with torch.no_grad():
                    if self.policy.use_rnn:
                        (
                            action,
                            logprob,
                            _,
                            value,
                            _,
                            hidden_state,
                        ) = self.policy.get_action_and_value(
                            obs, state, act_masks, None, obs_old, hidden_state
                        )
                    else:
                        (
                            action,
                            logprob,
                            _,
                            value,
                            _,
                        ) = self.policy.get_action_and_value(
                            obs, state, act_masks, None, obs_old
                        )

                (
                    next_obs,
                    next_state,
                    next_act_masks,
                    reward,
                    done,
                    info,
                ) = self.env_step(action)

                if len(self.observation_space.shape) == 3:
                    self.buffer.insert(
                        obs, act_masks, action, logprob, reward, value, next_done, step
                    )
                else:
                    self.buffer.insert(
                        obs,
                        state,
                        act_masks,
                        action,
                        logprob,
                        reward,
                        value,
                        next_done,
                        step,
                    )

                if self.env_family == "starcraft":
                    total_rewards += reward.max(-1)[0]  # (rollout_threads,)
                    # For each rollout, track the number of games player so far and record the wins for finished games
                    for i in range(self.rollout_threads):
                        if torch.isin(1, done[i]):
                            nb_games[i] += 1
                            for agent_info in info[
                                i * self.n_agents : (i + 1) * self.n_agents
                            ]:
                                if "battle_won" in agent_info:
                                    nb_wins[i] += int(agent_info["battle_won"])
                                    break
                else:
                    total_rewards += reward.sum(-1)  # (rollout_threads,)

                obs = next_obs
                state = next_state
                act_masks = next_act_masks
                next_done = done

            if self.env_family == "starcraft":
                total_rewards = total_rewards.cpu() / nb_games
                total_rewards = total_rewards.mean().item()
                episodic_wins = (nb_wins / nb_games).mean()
                print(
                    f"global_step={global_step}, episodic_return={total_rewards}, episodic_win_rate={episodic_wins}"
                )
                if self.use_comet:
                    self.exp.log_metric("episodic_return", total_rewards, global_step)
                    self.exp.log_metric("episodic_win_rate", episodic_wins, global_step)
            else:
                total_rewards = total_rewards.mean().item()
                print(f"global_step={global_step}, episodic_return={total_rewards}")
                if self.use_comet:
                    self.exp.log_metric("episodic_return", total_rewards, global_step)

            if total_rewards >= best_return:
                print(f"global_step={global_step}: Checkpoint saved!")
                self.save_checkpoints(self.save_dir)
                best_return = total_rewards

            with torch.no_grad():

                if self.policy_type == "conv":
                    bs = next_obs.shape[0]
                    n_ags = next_obs.shape[1]
                    next_obs = next_obs.reshape((-1,) + self.policy.obs_shape)
                    next_obs = self.policy.conv(next_obs)
                    next_obs = next_obs.reshape(bs, n_ags, self.policy.input_shape)
                    next_state = next_obs.reshape(bs, n_ags * self.policy.input_shape)
                    next_state = next_state.unsqueeze(1).repeat(1, n_ags, 1)

                if self.latent_kl:
                    next_obs_saf = self.policy.SAF(next_obs)
                    next_z, _ = self.policy.SAF.information_bottleneck(
                        next_obs_saf, next_obs, obs_old
                    )
                else:
                    next_z = None

                next_value = self.policy.get_value(next_obs, next_state, next_z)
                advantages, returns = self.policy.compute_returns(
                    self.buffer, next_value, next_done
                )

            metrics = self.policy.train_step(self.buffer, advantages, returns)
            if self.use_comet:
                for k in metrics:
                    self.exp.log_metric(k, metrics[k], global_step)

                self.exp.log_metric(
                    "learning_rate",
                    self.policy.optimizer.param_groups[0]["lr"],
                    global_step,
                )
                self.exp.log_metric(
                    "SPS", int(global_step / (time.time() - start_time)), global_step
                )

    def evaluate(self):

        global_step = 0

        obs, state, act_masks = self.env_reset(mode="eval")
        next_done = torch.zeros((1, self.n_agents)).to(self.device)

        if self.latent_kl:
            ## old_observation - shifted tensor (the zero-th obs is assumed to be equal to the first one)
            obs_old = obs.clone()
            obs_old[1:] = obs_old.clone()[:-1]

            if self.policy_type == "conv":
                bs = obs_old.shape[0]
                n_ags = obs_old.shape[1]

                obs_old = obs_old.reshape((-1,) + self.policy.obs_shape)
                obs_old = self.policy.conv(obs_old)
                obs_old = obs_old.reshape(bs, n_ags, self.policy.input_shape)
        else:
            obs_old = None

        agg_returns = []
        agg_winrate = []

        for global_step in range(self.eval_episodes):

            total_rewards = 0

            nb_games = np.ones(1)
            nb_wins = np.zeros(1)

            if self.policy.use_rnn:
                self.policy.init_hidden()
                hidden_state = self.policy.hidden_state
                if self.policy.use_policy_pool:
                    for j in range(self.policy.n_policy):
                        for i in range(self.n_agents):
                            hidden_state[j][i] = hidden_state[j][i].repeat(
                                obs.shape[0], 1
                            )
                else:
                    for i in range(self.n_agents):
                        hidden_state[i] = hidden_state[i].repeat(obs.shape[0], 1)
            for step in range(self.env_steps):

                with torch.no_grad():
                    if self.policy.use_rnn:
                        (
                            action,
                            logprob,
                            _,
                            value,
                            _,
                            hidden_state,
                        ) = self.policy.get_action_and_value(
                            obs, state, act_masks, None, obs_old, hidden_state
                        )
                    else:
                        (
                            action,
                            logprob,
                            _,
                            value,
                            _,
                        ) = self.policy.get_action_and_value(
                            obs, state, act_masks, None, obs_old
                        )

                (
                    next_obs,
                    next_state,
                    next_act_masks,
                    reward,
                    done,
                    info,
                ) = self.env_step(action, mode="eval")

                if self.env_family == "starcraft":
                    total_rewards += reward.max(-1)[0]  # (rollout_threads,)
                    # For each rollout, track the number of games player so far and record the wins for finished games
                    for i in range(1):
                        if torch.isin(1, done[i]):
                            nb_games[i] += 1
                            for agent_info in info[
                                i * self.n_agents : (i + 1) * self.n_agents
                            ]:
                                if "battle_won" in agent_info:
                                    nb_wins[i] += int(agent_info["battle_won"])
                                    break
                else:
                    total_rewards += reward.sum(-1)  # (rollout_threads,)

                obs = next_obs
                state = next_state
                act_masks = next_act_masks
                next_done = done
                if torch.any(done[0]):
                    break

            if self.env_family == "starcraft":
                total_rewards = total_rewards.cpu() / nb_games
                total_rewards = total_rewards.mean().item()
                episodic_wins = (nb_wins / nb_games).mean()
                print(
                    f"global_step={global_step}, episodic_return={total_rewards}, episodic_win_rate={episodic_wins}"
                )
            else:
                total_rewards = total_rewards.mean().item()
                print(f"global_step={global_step}, episodic_return={total_rewards}")
                if self.use_comet:
                    self.exp.log_metric("episodic_return", total_rewards, global_step)

            agg_returns.append(total_rewards)
            if self.env_family == "starcraft":
                agg_winrate.append(episodic_wins)
            else:
                agg_winrate.append(0)

        mean_rewards = np.mean(agg_returns)
        std_rewards = np.std(agg_returns)

        mean_wins = np.mean(agg_winrate)
        std_wins = np.std(agg_winrate)

        return mean_rewards, std_rewards, mean_wins, std_wins

    def load_checkpoints(self, checkpoint_dir):
        self.policy.load_checkpoints(checkpoint_dir)

    def save_checkpoints(self, checkpoint_dir):
        self.policy.save_checkpoints(checkpoint_dir)
