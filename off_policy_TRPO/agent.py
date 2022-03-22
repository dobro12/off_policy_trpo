from typing import Optional, List

from models import Policy
from models import Value

from collections import deque
from copy import deepcopy
import numpy as np
import pickle
import random
import torch
import copy
import time
import os

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

@torch.jit.script
def clip(a, maximum, minimum):
    clipped = torch.where(a > maximum, maximum, a)
    clipped = torch.where(clipped < minimum, minimum, clipped)
    return clipped

def flatGrad(y, x, retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True
    g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
    g = torch.cat([t.view(-1) for t in g])
    return g

class Agent:
    def __init__(self, args):
        # base
        self.device = args.device
        self.name = args.name
        self.checkpoint_dir=f'{args.save_dir}/checkpoint'
        # for env
        self.discount_factor = args.discount_factor
        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.action_bound_min = torch.tensor(args.action_bound_min, device=args.device)
        self.action_bound_max = torch.tensor(args.action_bound_max, device=args.device)
        self.n_envs = args.n_envs
        self.n_steps = args.n_steps
        self.n_steps_per_env = int(self.n_steps/self.n_envs)
        # for training
        self.gae_coeff = args.gae_coeff
        self.ent_coeff = args.ent_coeff
        self.lr = args.lr
        self.value_epochs = args.n_epochs
        self.num_replays = args.num_replays
        self.max_grad_norm = args.max_grad_norm
        # for trust region
        self.damping_coeff = args.damping_coeff
        self.num_conjugate = args.num_conjugate
        self.max_kl = args.max_kl
        self.line_decay = args.line_decay
        self.clip_value = args.clip_value
        self.improve_ratio = args.improve_ratio
        # for networks
        self.policy = Policy(args).to(args.device)
        self.value = Value(args).to(args.device)
        self.optimizer = torch.optim.Adam(self.value.parameters(), lr=self.lr)
        # replay_buffer
        self.replay_buffer = [deque(maxlen=int(args.len_replay_buffer/args.n_envs)) for _ in range(args.n_envs)]
        # load model
        self.load()


    def normalizeAction(self, a):
        return normalize(a, self.action_bound_max, self.action_bound_min)

    def unnormalizeAction(self, a):
        return unnormalize(a, self.action_bound_max, self.action_bound_min)

    def getAction(self, state, is_train):
        mean, log_std, std = self.policy(state)
        if is_train:
            noise = torch.randn(*mean.size(), device=self.device)
            action = self.unnormalizeAction(mean + noise*std)
        else:
            action = self.unnormalizeAction(mean)
        clipped_action = clip(action, self.action_bound_max, self.action_bound_min)
        return action, clipped_action, mean, std

    def getGaesTargets(self, rewards, values, dones, fails, next_values, rhos):
        delta = 0.0
        targets = np.zeros_like(rewards)
        for t in reversed(range(len(targets))):
            targets[t] = rewards[t] + self.discount_factor*(1.0 - fails[t])*next_values[t] \
                            + self.discount_factor*(1.0 - dones[t])*delta
            delta = self.gae_coeff*rhos[t]*(targets[t] - values[t])
        gaes = targets - values
        return gaes, targets

    def getEntropy(self, states):
        means, log_stds, stds = self.policy(states)
        normal = torch.distributions.Normal(means, stds)
        entropy = torch.mean(torch.sum(normal.entropy(), dim=1))
        return entropy

    def train(self, trajs):
        for env_idx in range(self.n_envs):
            self.replay_buffer[env_idx] += deque(trajs[env_idx])

        value_losses = []
        objectives = []
        kls = []
        entropies = []

        value_loss, objective, kl, entropy = self.update(is_last=True)
        value_losses.append(value_loss)
        objectives.append(objective)
        kls.append(kl)
        entropies.append(entropy)

        for replay_idx in range(self.num_replays):
            value_loss, objective, kl, entropy = self.update(is_last=False)
            value_losses.append(value_loss)
            objectives.append(objective)
            kls.append(kl)
            entropies.append(entropy)

        return np.mean(value_losses), np.mean(objectives), np.mean(kls), np.mean(entropies)

    def update(self, is_last=False):
        # convert to numpy array
        states_list = []
        actions_list = []
        gaes_list = []
        targets_list = []
        mu_means_list = []
        mu_stds_list = []

        for env_idx in range(self.n_envs):
            n_steps = min(len(self.replay_buffer[env_idx]), self.n_steps_per_env)
            if is_last:
                start_idx = len(self.replay_buffer[env_idx]) - n_steps
            else:
                start_idx = np.random.randint(0, len(self.replay_buffer[env_idx]) - n_steps + 1)
            end_idx = start_idx + n_steps

            env_traj = list(self.replay_buffer[env_idx])[start_idx:end_idx]
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
                next_states_tensor = torch.tensor(next_states, device=self.device, dtype=torch.float32)
                actions_tensor = self.normalizeAction(torch.tensor(actions, device=self.device, dtype=torch.float32))
                mu_means_tensor = torch.tensor(mu_means, device=self.device, dtype=torch.float)
                mu_stds_tensor = torch.tensor(mu_stds, device=self.device, dtype=torch.float)

                # calculate rhos
                means_tensor, _, stds_tensor = self.policy(states_tensor)
                old_dist = torch.distributions.Normal(means_tensor, stds_tensor + EPS)
                old_log_probs = torch.sum(old_dist.log_prob(actions_tensor), dim=1)
                mu_dist = torch.distributions.Normal(mu_means_tensor, mu_stds_tensor + EPS)
                mu_log_probs = torch.sum(mu_dist.log_prob(actions_tensor), dim=1)
                rhos = torch.clamp(torch.exp(old_log_probs - mu_log_probs), 0.0, 1.0).detach().cpu().numpy()

                # calculate values
                values_tensor = self.value(states_tensor)
                next_values_tensor = self.value(next_states_tensor)

            values = values_tensor.detach().cpu().numpy()
            next_values = next_values_tensor.detach().cpu().numpy()
            gaes, targets = self.getGaesTargets(rewards, values, dones, fails, next_values, rhos)

            states_list.append(states)
            actions_list.append(actions)
            gaes_list.append(gaes)
            targets_list.append(targets)
            mu_means_list.append(mu_means)
            mu_stds_list.append(mu_stds)

        states = np.concatenate(states_list, axis=0)
        actions = np.concatenate(actions_list, axis=0)
        gaes = np.concatenate(gaes_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)
        mu_means = np.concatenate(mu_means_list, axis=0)
        mu_stds = np.concatenate(mu_stds_list, axis=0)

        # convert to tensor
        states_tensor = torch.tensor(states, device=self.device, dtype=torch.float32)
        actions_tensor = self.normalizeAction(torch.tensor(actions, device=self.device, dtype=torch.float32))
        mu_means_tensor = torch.tensor(mu_means, device=self.device, dtype=torch.float32)
        mu_stds_tensor = torch.tensor(mu_stds, device=self.device, dtype=torch.float32)
        gaes_tensor = torch.tensor(gaes, device=self.device, dtype=torch.float32)
        targets_tensor = torch.tensor(targets, device=self.device, dtype=torch.float32)

        # ========== for policy update ========== #
        # backup old policy
        with torch.no_grad():
            means, log_stds, stds = self.policy(states_tensor)
            old_means = means.clone().detach()
            old_stds = stds.clone().detach()

        # get objective & KL
        objective, entropy = self.getObjective(states_tensor, actions_tensor, gaes_tensor, old_means, old_stds, mu_means_tensor, mu_stds_tensor)
        cur_means, _, cur_stds = self.policy(states_tensor)
        kl_original = self.getKL(old_means, old_stds, cur_means, cur_stds)
        kl_old = self.getKL(mu_means_tensor, mu_stds_tensor, old_means, old_stds)
        kl_old = torch.sqrt(kl_old*(self.max_kl + 0.25*kl_old)) - 0.5*kl_old
        kl = kl_original + kl_old

        # find search direction
        grad_g = flatGrad(objective, self.policy.parameters(), retain_graph=True)
        x_value = self.conjugateGradient(kl, grad_g)

        # line search
        Ax = self.Hx(kl, x_value)
        xAx = torch.dot(x_value, Ax)
        beta = torch.sqrt(2.0*self.max_kl / xAx)
        init_theta = torch.cat([t.view(-1) for t in self.policy.parameters()]).clone().detach()
        init_objective = objective.clone().detach()
        expected_improve = torch.dot(grad_g, x_value)
        while True:
            theta = beta*x_value + init_theta
            self.applyParams(theta)
            objective, entropy = self.getObjective(states_tensor, actions_tensor, gaes_tensor, old_means, old_stds, mu_means_tensor, mu_stds_tensor)
            improve = objective - init_objective
            improve_ratio = improve/(expected_improve*beta)
            cur_means, _, cur_stds = self.policy(states_tensor)
            kl_original = self.getKL(old_means, old_stds, cur_means, cur_stds)
            kl = kl_original + kl_old
            if kl <= self.max_kl and (improve_ratio > self.improve_ratio and improve > 0.0):
                break
            beta *= self.line_decay
        # ======================================= #

        # =========== for value update =========== #
        for _ in range(self.value_epochs):
            value_loss = torch.mean(torch.square(self.value(states_tensor) - targets_tensor))
            self.optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
            self.optimizer.step()
        # ======================================== #

        scalar = lambda x:x.detach().cpu().numpy()
        np_value_loss = scalar(value_loss)
        np_objective = scalar(objective)
        np_kl = scalar(kl)
        np_entropy = scalar(entropy)
        return np_value_loss, np_objective, np_kl, np_entropy

    def getObjective(self, states, actions, gaes, old_means, old_stds, mu_means, mu_stds):
        means, log_stds, stds = self.policy(states)
        dist = torch.distributions.Normal(means, stds + EPS)
        log_probs = torch.sum(dist.log_prob(actions), dim=1)
        entropy = torch.mean(torch.sum(dist.entropy(), dim=1))

        old_dist = torch.distributions.Normal(old_means, old_stds + EPS)
        old_log_probs = torch.sum(old_dist.log_prob(actions), dim=1)

        mu_dist = torch.distributions.Normal(mu_means, mu_stds + EPS)
        mu_log_probs = torch.sum(mu_dist.log_prob(actions), dim=1)

        if self.clip_value > 0:
            prob_ratios = torch.clamp(torch.exp(log_probs - mu_log_probs), 1.0 - self.clip_value, 1.0 + self.clip_value)
            old_prob_ratios = torch.clamp(torch.exp(old_log_probs - mu_log_probs), 1.0 - self.clip_value, 1.0 + self.clip_value)
        else:
            prob_ratios = torch.exp(log_probs - mu_log_probs)
            old_prob_ratios = torch.exp(old_log_probs - mu_log_probs)

        with torch.no_grad():
            gaes_mean = torch.mean(gaes*old_prob_ratios)/(torch.mean(old_prob_ratios) + EPS)
            gaes_std = torch.std(gaes)

        objective = torch.mean(prob_ratios*(gaes - gaes_mean)/(gaes_std + EPS))
        objective += self.ent_coeff*(entropy/self.action_dim)
        return objective, entropy

    def getKL(self, old_means, old_stds, means, stds):
        dist = torch.distributions.Normal(means, stds)
        old_dist = torch.distributions.Normal(old_means, old_stds)
        kl = torch.distributions.kl.kl_divergence(old_dist, dist)
        kl = torch.mean(torch.sum(kl, dim=1))
        return kl

    def applyParams(self, params):
        n = 0
        for p in self.policy.parameters():
            numel = p.numel()
            g = params[n:n + numel].view(p.shape)
            p.data = g
            n += numel

    def Hx(self, kl:torch.Tensor, x:torch.Tensor) -> torch.Tensor:
        '''
        get (Hessian of KL * x).
        input:
            kl: tensor(,)
            x: tensor(dim,)
        output:
            Hx: tensor(dim,)
        '''
        flat_grad_kl = flatGrad(kl, self.policy.parameters(), create_graph=True)
        kl_x = torch.dot(flat_grad_kl, x)
        H_x = flatGrad(kl_x, self.policy.parameters(), retain_graph=True)
        return H_x + x*self.damping_coeff

    def conjugateGradient(self, kl:torch.Tensor, g:torch.Tensor) -> torch.Tensor:
        '''
        get (H^{-1} * g).
        input:
            kl: tensor(,)
            g: tensor(dim,)
        output:
            H^{-1}g: tensor(dim,)
        '''
        x = torch.zeros_like(g, device=self.device)
        r = g.clone()
        p = g.clone()
        rs_old = torch.sum(r*r)
        for i in range(self.num_conjugate):
            Ap = self.Hx(kl, p)
            pAp = torch.sum(p*Ap)
            alpha = rs_old/(pAp + EPS)
            x += alpha*p
            r -= alpha*Ap
            rs_new = torch.sum(r*r)
            p = r + (rs_new/rs_old)*p
            rs_old = rs_new
        return x

    def save(self):
        torch.save({
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'optim': self.optimizer.state_dict(),
            }, f"{self.checkpoint_dir}/model.pt")
        print('[save] success.')

    def load(self):
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        checkpoint_file = f"{self.checkpoint_dir}/model.pt"
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            self.policy.load_state_dict(checkpoint['policy'])
            self.value.load_state_dict(checkpoint['value'])
            self.optimizer.load_state_dict(checkpoint['optim'])
            print('[load] success.')
        else:
            self.policy.initialize()
            self.value.initialize()
            print('[load] fail.')
