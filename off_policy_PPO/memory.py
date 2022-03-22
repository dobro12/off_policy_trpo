from collections import deque
from copy import deepcopy
import numpy as np
import torch

EPS = 1e-8

@torch.jit.script
def normalize(a, maximum, minimum):
    temp_a = 1.0/(maximum - minimum)
    temp_b = minimum/(minimum - maximum)
    temp_a = torch.ones_like(a)*temp_a
    temp_b = torch.ones_like(a)*temp_b
    return temp_a*a + temp_b

@torch.jit.script
def unnormalize(a, maximum, minimum):
    temp_a = maximum - minimum
    temp_b = minimum
    temp_a = torch.ones_like(a)*temp_a
    temp_b = torch.ones_like(a)*temp_b
    return temp_a*a + temp_b

class ReplayBuffer:
    def __init__(self, args):
        self.n_envs = args.n_envs
        self.n_steps = args.n_steps
        self.n_steps_per_env = int(self.n_steps/self.n_envs)
        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.gae_coeff = args.gae_coeff
        self.discount_factor = args.discount_factor
        self.max_episode_steps = args.max_episode_steps

        self.device = args.device
        self.action_bound_min = torch.tensor(args.action_bound_min, device=args.device, requires_grad=False)
        self.action_bound_max = torch.tensor(args.action_bound_max, device=args.device, requires_grad=False)

        self.trajectories = [deque(maxlen=int(args.len_replay_buffer/args.n_envs)) for _ in range(args.n_envs)]
        self.states_buffer = None
        self.actions_buffer = None
        self.targets_buffer = None
        self.gaes_buffer = None
        self.old_log_probs_buffer = None
        self.mu_log_probs_buffer = None


    def add(self, observation, action, reward, done, fail, next_observation, mean, std, env_idx):
        self.trajectories[env_idx].append([
            observation, action, reward, float(done), float(fail), next_observation, mean, std, 
        ])

    def getGaesTargets(self, rewards, values, dones, fails, next_values, rhos):
        delta = 0.0
        targets = np.zeros_like(rewards)
        for t in reversed(range(len(targets))):
            targets[t] = rewards[t] + self.discount_factor*(1.0 - fails[t])*next_values[t] \
                            + self.discount_factor*(1.0 - dones[t])*delta
            delta = self.gae_coeff*rhos[t]*(targets[t] - values[t])
        gaes = targets - values
        return gaes, targets

    def update(self, policy, value):
        # update buffers
        self.states_buffer = []
        self.actions_buffer = []
        self.targets_buffer = []
        self.gaes_buffer = []
        self.old_log_probs_buffer = []
        self.mu_log_probs_buffer = []

        for env_idx in range(self.n_envs):
            env_traj = list(self.trajectories[env_idx])
            states = np.array([traj[0] for traj in env_traj])
            actions = np.array([traj[1] for traj in env_traj])
            rewards = np.array([traj[2] for traj in env_traj])
            dones = np.array([traj[3] for traj in env_traj])
            fails = np.array([traj[4] for traj in env_traj])
            next_states = np.array([traj[5] for traj in env_traj])
            mu_means = np.array([traj[6] for traj in env_traj])
            mu_stds = np.array([traj[7] for traj in env_traj])

            with torch.no_grad():
                states_tensor = torch.tensor(states, device=self.device, dtype=torch.float32)
                actions_tensor = normalize(torch.tensor(actions, device=self.device, dtype=torch.float), self.action_bound_max, self.action_bound_min)
                next_states_tensor = torch.tensor(next_states, device=self.device, dtype=torch.float32)
                mu_means_tensor = torch.tensor(mu_means, device=self.device, dtype=torch.float)
                mu_stds_tensor = torch.tensor(mu_stds, device=self.device, dtype=torch.float)

                # calculate rhos
                means_tensor, _, stds_tensor = policy(states_tensor)
                old_dists_tensor = torch.distributions.Normal(means_tensor, stds_tensor + EPS)
                old_log_probs_tensor = torch.sum(old_dists_tensor.log_prob(actions_tensor), dim=1)
                mu_dists_tensor = torch.distributions.Normal(mu_means_tensor, mu_stds_tensor + EPS)
                mu_log_probs_tensor = torch.sum(mu_dists_tensor.log_prob(actions_tensor), dim=1)
                rhos = torch.clamp(torch.exp(old_log_probs_tensor - mu_log_probs_tensor), 0.0, 1.0).detach().cpu().numpy()
                old_log_probs = old_log_probs_tensor.detach().cpu().numpy()
                mu_log_probs = mu_log_probs_tensor.detach().cpu().numpy()

                # get values
                values_tensor = value(states_tensor)
                next_values_tensor = value(next_states_tensor)

            values = values_tensor.detach().cpu().numpy()
            next_values = next_values_tensor.detach().cpu().numpy()
            gaes, targets = self.getGaesTargets(rewards, values, dones, fails, next_values, rhos)

            self.states_buffer.append(states)
            self.actions_buffer.append(actions)
            self.gaes_buffer.append(gaes)
            self.targets_buffer.append(targets)
            self.old_log_probs_buffer.append(old_log_probs)
            self.mu_log_probs_buffer.append(mu_log_probs)
            
        self.states_buffer = np.concatenate(self.states_buffer, axis=0)
        self.actions_buffer = np.concatenate(self.actions_buffer, axis=0)
        self.gaes_buffer = np.concatenate(self.gaes_buffer, axis=0)
        self.targets_buffer = np.concatenate(self.targets_buffer, axis=0)
        self.old_log_probs_buffer = np.concatenate(self.old_log_probs_buffer, axis=0)
        self.mu_log_probs_buffer = np.concatenate(self.mu_log_probs_buffer, axis=0)

    def sample(self, batch_indices):
        states = self.states_buffer[batch_indices]
        actions = self.actions_buffer[batch_indices]
        targets = self.targets_buffer[batch_indices]
        gaes = self.gaes_buffer[batch_indices]
        old_log_probs = self.old_log_probs_buffer[batch_indices]
        mu_log_probs = self.mu_log_probs_buffer[batch_indices]
        gaes = (gaes - np.mean(gaes))/(np.std(gaes) + 1e-8)
        return states, actions, targets, gaes, old_log_probs, mu_log_probs
