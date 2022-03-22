from typing import Optional, List

from models import Policy
from models import Value

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
        self.n_steps = args.n_steps
        self.n_envs = args.n_envs
        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.action_bound_min = torch.tensor(args.action_bound_min, device=args.device)
        self.action_bound_max = torch.tensor(args.action_bound_max, device=args.device)
        # for training
        self.gae_coeff = args.gae_coeff
        self.lr = args.lr
        self.value_epochs = args.n_epochs
        # for trust region
        self.damping_coeff = args.damping_coeff
        self.num_conjugate = args.num_conjugate
        self.max_kl = args.max_kl
        self.line_decay = args.line_decay
        # for networks
        self.policy = Policy(args).to(args.device)
        self.value = Value(args).to(args.device)
        self.optimizer = torch.optim.Adam(self.value.parameters(), lr=self.lr)
        self.load()


    def normalizeAction(self, a:torch.Tensor) -> torch.Tensor:
        return normalize(a, self.action_bound_max, self.action_bound_min)

    def unnormalizeAction(self, a:torch.Tensor) -> torch.Tensor:
        return unnormalize(a, self.action_bound_max, self.action_bound_min)

    def getAction(self, state:torch.Tensor, is_train:bool) -> List[torch.Tensor]:
        '''
        input:
            states:     Tensor(state_dim,)
            is_train:   boolean
        output:
            action:         Tensor(action_dim,)
            cliped_action:  Tensor(action_dim,)
        '''
        mean, log_std, std = self.policy(state)
        if is_train:
            noise = torch.randn(*mean.size(), device=self.device)
            action = self.unnormalizeAction(mean + noise*std)
        else:
            action = self.unnormalizeAction(mean)
        clipped_action = clip(action, self.action_bound_max, self.action_bound_min)
        return action, clipped_action

    def getGaesTargets(self, rewards:np.ndarray, values:np.ndarray, dones:np.ndarray, fails:np.ndarray, next_values:np.ndarray) -> List[np.ndarray]:
        '''
        input:
            rewards:        np.array(n_steps,)
            values:         np.array(n_steps,)
            dones:          np.array(n_steps,)
            fails:          np.array(n_steps,)
            next_values:    np.array(n_steps,)
        output:
            gaes:       np.array(n_steps,)
            targets:    np.array(n_steps,)
        '''
        delta = 0.0
        targets = np.zeros_like(rewards)
        for t in reversed(range(len(targets))):
            targets[t] = rewards[t] + self.discount_factor*(1.0 - fails[t])*next_values[t] \
                            + self.discount_factor*(1.0 - dones[t])*delta
            delta = self.gae_coeff*(targets[t] - values[t])
        gaes = targets - values
        return gaes, targets

    def getEntropy(self, states:torch.Tensor) -> torch.Tensor:
        '''
        return scalar tensor for entropy value.
        input:
            states:     Tensor(n_steps, state_dim)
        output:
            entropy:    Tensor(,)
        '''
        means, log_stds, stds = self.policy(states)
        normal = torch.distributions.Normal(means, stds)
        entropy = torch.mean(torch.sum(normal.entropy(), dim=1))
        return entropy

    def train(self, trajs):
        # convert to numpy array
        states_list = []
        actions_list = []
        gaes_list = []
        targets_list = []

        for env_idx in range(self.n_envs):
            env_traj = trajs[env_idx]
            states = np.array([traj[0] for traj in env_traj])
            actions = np.array([traj[1] for traj in env_traj])
            rewards = np.array([traj[2] for traj in env_traj])
            dones = np.array([traj[3] for traj in env_traj])
            fails = np.array([traj[4] for traj in env_traj])
            next_states = np.array([traj[5] for traj in env_traj])

            with torch.no_grad():
                states_tensor = torch.tensor(states, device=self.device, dtype=torch.float32)
                next_states_tensor = torch.tensor(next_states, device=self.device, dtype=torch.float32)
                values_tensor = self.value(states_tensor)
                next_values_tensor = self.value(next_states_tensor)
            values = values_tensor.detach().cpu().numpy()
            next_values = next_values_tensor.detach().cpu().numpy()
            gaes, targets = self.getGaesTargets(rewards, values, dones, fails, next_values)

            states_list.append(states)
            actions_list.append(actions)
            gaes_list.append(gaes)
            targets_list.append(targets)        

        states = np.concatenate(states_list, axis=0)
        actions = np.concatenate(actions_list, axis=0)
        gaes = np.concatenate(gaes_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)
        gaes = (gaes - np.mean(gaes))/(np.std(gaes) + EPS)

        # convert to tensor
        states_tensor = torch.tensor(states, device=self.device, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, device=self.device, dtype=torch.float32)
        norm_actions_tensor = self.normalizeAction(actions_tensor)
        gaes_tensor = torch.tensor(gaes, device=self.device, dtype=torch.float32)
        targets_tensor = torch.tensor(targets, device=self.device, dtype=torch.float32)

        # get entropy
        entropy = self.getEntropy(states_tensor)

        # ========== for policy update ========== #
        # backup old policy
        with torch.no_grad():
            means, log_stds, stds = self.policy(states_tensor)
            old_means = means.clone().detach()
            old_stds = stds.clone().detach()

        # get objective & KL
        objective = self.getObjective(states_tensor, norm_actions_tensor, gaes_tensor, old_means, old_stds)
        kl = self.getKL(states_tensor, old_means, old_stds)

        # find search direction
        grad_g = flatGrad(objective, self.policy.parameters(), retain_graph=True)
        x_value = self.conjugateGradient(kl, grad_g)

        # line search
        Ax = self.Hx(kl, x_value)
        xAx = torch.dot(x_value, Ax)
        beta = torch.sqrt(2.0*self.max_kl / xAx)
        init_theta = torch.cat([t.view(-1) for t in self.policy.parameters()]).clone().detach()
        init_objective = objective.clone().detach()
        while True:
            theta = beta*x_value + init_theta
            self.applyParams(theta)
            objective = self.getObjective(states_tensor, norm_actions_tensor, gaes_tensor, old_means, old_stds)
            kl = self.getKL(states_tensor, old_means, old_stds)
            if kl <= self.max_kl and objective > init_objective:
                break
            beta *= self.line_decay
        # ======================================= #

        # =========== for value update =========== #
        for _ in range(self.value_epochs):
            value_loss = torch.mean(torch.square(self.value(states_tensor) - targets_tensor))
            self.optimizer.zero_grad()
            value_loss.backward()
            self.optimizer.step()
        # ======================================== #

        scalar = lambda x:x.detach().cpu().numpy()
        np_value_loss = scalar(value_loss)
        np_objective = scalar(objective)
        np_kl = scalar(kl)
        np_entropy = scalar(entropy)
        return np_value_loss, np_objective, np_kl, np_entropy

    def getObjective(self, states, norm_actions, gaes, old_means, old_stds):
        means, log_stds, stds = self.policy(states)
        dist = torch.distributions.Normal(means, stds)
        old_dist = torch.distributions.Normal(old_means, old_stds)
        log_probs = torch.sum(dist.log_prob(norm_actions), dim=1)
        old_log_probs = torch.sum(old_dist.log_prob(norm_actions), dim=1)
        objective = torch.mean(torch.exp(log_probs - old_log_probs)*gaes)
        return objective

    def getKL(self, states, old_means, old_stds):
        means, log_stds, stds = self.policy(states)
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
