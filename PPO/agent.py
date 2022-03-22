from typing import Optional, List

from memory import RolloutBuffer
from models import Policy
from models import Value

from sklearn.utils import shuffle
from collections import deque
from scipy.stats import norm
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
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.n_epochs = args.n_epochs
        self.clip_value = args.clip_value
        self.gae_coeff = args.gae_coeff
        self.ent_coeff = args.ent_coeff
        self.vf_coeff = args.vf_coeff
        self.max_kl = args.max_kl
        self.max_grad_norm = args.max_grad_norm
        # for networks
        self.policy = Policy(args).to(args.device)
        self.value = Value(args).to(args.device)
        # optimizer
        self.param_list = list(self.policy.parameters()) + list(self.value.parameters())
        self.optimizer = torch.optim.Adam(self.param_list, lr=self.lr)
        # memory
        self.rollout_buffer = RolloutBuffer(args)
        # load model
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

    def add(self, obs, action, reward, done, fail, next_obs, env_idx=0):
        self.rollout_buffer.add(obs, action, reward, done, fail, next_obs, env_idx)

    def train(self):
        self.rollout_buffer.update(self.policy, self.value)

        value_losses = []
        policy_losses = []
        kls = []
        entropies = []
        continue_training = True

        for _ in range(self.n_epochs):
            indices = np.random.permutation(self.n_envs*self.n_steps_per_env)
            start_idx = 0
            end_idx = self.batch_size

            while end_idx <= len(indices):
                batch_indices = indices[start_idx:end_idx]
                trajs = self.rollout_buffer.sample(batch_indices)
                value_loss, policy_loss, kl, entropy = self.update(trajs)

                value_losses.append(value_loss)
                policy_losses.append(policy_loss)
                kls.append(kl)
                entropies.append(entropy)

                if kl > self.max_kl*1.5:
                    continue_training = False
                    break

                start_idx += self.batch_size
                end_idx += self.batch_size

            if not continue_training:
                break

        return np.mean(value_losses), np.mean(policy_losses), np.mean(kls), np.mean(entropies)

    def update(self, trajs):
        states, actions, targets, gaes, old_log_probs = trajs

        # convert to tensor
        states_tensor = torch.tensor(states, device=self.device, dtype=torch.float32)
        actions_tensor = self.normalizeAction(torch.tensor(actions, device=self.device, dtype=torch.float32))
        gaes_tensor = torch.tensor(gaes, device=self.device, dtype=torch.float32)
        targets_tensor = torch.tensor(targets, device=self.device, dtype=torch.float32)
        old_log_probs_tensor = torch.tensor(old_log_probs, device=self.device, dtype=torch.float32)

        # ========== for policy update ========== #
        means, log_stds, stds = self.policy(states_tensor)
        dists = torch.distributions.Normal(means, stds)
        log_probs = torch.sum(dists.log_prob(actions_tensor), dim=1)
        entropy = torch.mean(torch.sum(dists.entropy(), dim=1))

        ratios = torch.exp(log_probs - old_log_probs_tensor)
        clipped_ratios = torch.clamp(ratios, min=1.0 - self.clip_value, max=1.0 + self.clip_value)

        policy_loss = -(torch.mean(torch.minimum(gaes_tensor*ratios, gaes_tensor*clipped_ratios)))
        value_loss = torch.mean(torch.square(self.value(states_tensor) - targets_tensor))
        entropy_loss = -entropy
        loss = policy_loss + self.vf_coeff*value_loss + self.ent_coeff*entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.param_list, self.max_grad_norm)
        self.optimizer.step()

        # calculate KL
        means, _, stds = self.policy(states_tensor)
        dists = torch.distributions.Normal(means, stds)
        log_ratio = torch.sum(dists.log_prob(actions_tensor), dim=1) - old_log_probs_tensor
        approx_kl = torch.mean((torch.exp(log_ratio) - 1.0) - log_ratio)
        # ======================================= #

        scalar = lambda x:x.detach().cpu().numpy()
        np_value_loss = scalar(value_loss)
        np_policy_loss = scalar(policy_loss)
        np_kl = scalar(approx_kl)
        np_entropy = scalar(entropy)
        return np_value_loss, np_policy_loss, np_kl, np_entropy

    def getKL(self, states, old_means, old_stds):
        means, log_stds, stds = self.policy(states)
        dist = torch.distributions.Normal(means, stds)
        old_dist = torch.distributions.Normal(old_means, old_stds)
        kl = torch.distributions.kl.kl_divergence(old_dist, dist)
        kl = torch.mean(torch.sum(kl, dim=1))
        return kl

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

    def save(self):
        torch.save({
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'optim': self.optimizer.state_dict(),
            }, f"{self.checkpoint_dir}/checkpoint")
        print('[save] success.')

    def load(self):
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        checkpoint_file = f"{self.checkpoint_dir}/checkpoint"
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
