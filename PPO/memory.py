from copy import deepcopy
import numpy as np
import torch


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

class RolloutBuffer:
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

        self.trajectories = [[] for _ in range(self.n_envs)]
        self.states_buffer = None
        self.actions_buffer = None
        self.targets_buffer = None
        self.gaes_buffer = None
        self.log_probs_buffer = None


    def add(self, observation, action, reward, done, fail, next_observation, env_idx):
        self.trajectories[env_idx].append([
            observation, action, reward, float(done), float(fail), next_observation,
        ])

    def getGaesTargets(self, rewards, values, dones, fails, next_values):
        delta = 0.0
        targets = np.zeros_like(rewards)
        for t in reversed(range(len(targets))):
            targets[t] = rewards[t] + self.discount_factor*(1.0 - fails[t])*next_values[t] \
                            + self.discount_factor*(1.0 - dones[t])*delta
            delta = self.gae_coeff*(targets[t] - values[t])
        gaes = targets - values
        return gaes, targets

    def update(self, policy, value):
        # update buffers
        self.states_buffer = []
        self.actions_buffer = []
        self.targets_buffer = []
        self.gaes_buffer = []
        self.log_probs_buffer = []

        for env_idx in range(self.n_envs):
            env_traj = self.trajectories[env_idx]
            states = np.array([traj[0] for traj in env_traj])
            actions = np.array([traj[1] for traj in env_traj])
            rewards = np.array([traj[2] for traj in env_traj])
            dones = np.array([traj[3] for traj in env_traj])
            fails = np.array([traj[4] for traj in env_traj])
            next_states = np.array([traj[5] for traj in env_traj])

            with torch.no_grad():
                states_tensor = torch.tensor(states, device=self.device, dtype=torch.float32)
                actions_tensor = normalize(torch.tensor(actions, device=self.device, dtype=torch.float), self.action_bound_max, self.action_bound_min)
                next_states_tensor = torch.tensor(next_states, device=self.device, dtype=torch.float32)

                # get log probs
                means_tensor, _, stds_tensor = policy(states_tensor)
                dists = torch.distributions.Normal(means_tensor, stds_tensor)
                log_probs_tensor = torch.sum(dists.log_prob(actions_tensor), dim=1)
                log_probs = log_probs_tensor.detach().cpu().numpy()

                # get values
                values_tensor = value(states_tensor)
                next_values_tensor = value(next_states_tensor)

            values = values_tensor.detach().cpu().numpy()
            next_values = next_values_tensor.detach().cpu().numpy()
            gaes, targets = self.getGaesTargets(rewards, values, dones, fails, next_values)

            self.states_buffer.append(states)
            self.actions_buffer.append(actions)
            self.gaes_buffer.append(gaes)
            self.targets_buffer.append(targets)
            self.log_probs_buffer.append(log_probs)
            
        self.states_buffer = np.concatenate(self.states_buffer, axis=0)
        self.actions_buffer = np.concatenate(self.actions_buffer, axis=0)
        self.gaes_buffer = np.concatenate(self.gaes_buffer, axis=0)
        self.targets_buffer = np.concatenate(self.targets_buffer, axis=0)
        self.log_probs_buffer = np.concatenate(self.log_probs_buffer, axis=0)

        # reset the trajectories
        self.trajectories = [[] for _ in range(self.n_envs)]

    def sample(self, batch_indices):
        states = self.states_buffer[batch_indices]
        actions = self.actions_buffer[batch_indices]
        targets = self.targets_buffer[batch_indices]
        gaes = self.gaes_buffer[batch_indices]
        log_probs = self.log_probs_buffer[batch_indices]
        gaes = (gaes - np.mean(gaes))/(np.std(gaes) + 1e-8)
        return states, actions, targets, gaes, log_probs
