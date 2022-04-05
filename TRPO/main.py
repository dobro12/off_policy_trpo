# ===== add python path ===== #
import glob
import sys
import os
PATH = os.getcwd()
for dir_idx, dir_name in enumerate(PATH.split('/')):
    dir_path = '/'.join(PATH.split('/')[:(dir_idx+1)])
    file_list = [os.path.basename(sub_dir) for sub_dir in glob.glob(f"{dir_path}/.*")]
    if '.dobro_package' in file_list:
        PATH = dir_path
        break
if not PATH in sys.path:
    sys.path.append(PATH)
# =========================== #

from utils.vectorize import DobroSubprocVecEnv
from utils.normalize import RunningMeanStd
from utils.slackbot import Slackbot
from utils.logger import Logger
from agent import Agent

from stable_baselines3.common.env_util import make_vec_env
from collections import deque
from copy import deepcopy
import numpy as np
import argparse
import pickle
import random
import torch
import wandb
import time
import gym

def getPaser():
    parser = argparse.ArgumentParser(description='legged_robot')
    # common
    parser.add_argument('--wandb',  action='store_true', help='use wandb?')
    parser.add_argument('--slack',  action='store_true', help='use slack?')
    parser.add_argument('--test',  action='store_true', help='test or train?')
    parser.add_argument('--name', type=str, default='offpolicy_TRPO', help='save name.')
    parser.add_argument('--save_freq', type=int, default=int(1e6), help='# of time steps for save.')
    parser.add_argument('--slack_freq', type=int, default=int(2.5e6), help='# of time steps for slack message.')
    parser.add_argument('--total_steps', type=int, default=int(1e7), help='total training steps.')
    parser.add_argument('--seed', type=int, default=1, help='seed number.')
    parser.add_argument('--device', type=str, default='gpu', help='gpu or cpu.')
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index.')
    # for env
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v2', help='gym environment name.')
    parser.add_argument('--max_episode_steps', type=int, default=1000, help='# of maximum episode steps.')
    parser.add_argument('--n_envs', type=int, default=4, help='gym environment name.')
    parser.add_argument('--n_steps', type=int, default=5000, help='# of steps for each environment per update.')
    # for networks
    parser.add_argument('--activation', type=str, default='ReLU', help='activation function. ReLU, Tanh, Sigmoid...')
    parser.add_argument('--hidden_dim', type=int, default=256, help='the number of hidden layer\'s node.')
    parser.add_argument('--log_std_init', type=float, default=0.0, help='log of initial std.')
    # for RL
    parser.add_argument('--discount_factor', type=float, default=0.99, help='discount factor.')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate.')
    parser.add_argument('--n_epochs', type=int, default=10, help='update epochs.')
    parser.add_argument('--gae_coeff', type=float, default=0.97, help='gae coefficient.')
    parser.add_argument('--ent_coeff', type=float, default=0.0, help='entropy coefficient.')
    # trust region
    parser.add_argument('--damping_coeff', type=float, default=0.01, help='damping coefficient.')
    parser.add_argument('--num_conjugate', type=int, default=10, help='# of maximum conjugate step.')
    parser.add_argument('--line_decay', type=float, default=0.8, help='line decay.')
    parser.add_argument('--max_kl', type=float, default=0.01, help='maximum kl divergence.')
    return parser

def train(args):
    # wandb
    if args.wandb:
        project_name = '[off-policy-TRPO] mujoco'
        wandb.init(
            project=project_name, 
            config=args,
        )
        run_idx = wandb.run.name.split('-')[-1]
        wandb.run.name = f"{args.name}-{run_idx}"

    # slackbot
    if args.slack:
        slackbot = Slackbot()

    # define env
    vec_env = make_vec_env(
        env_id=lambda: gym.make(args.env_name), n_envs=args.n_envs, seed=args.seed,
        vec_env_cls=DobroSubprocVecEnv,
        vec_env_kwargs={'args':args, 'start_method':'spawn'},
    )

    # set args value for env
    args.obs_dim = vec_env.observation_space.shape[0]
    args.action_dim = vec_env.action_space.shape[0]
    args.action_bound_min = vec_env.action_space.low
    args.action_bound_max = vec_env.action_space.high

    # define agent
    agent = Agent(args)

    # for log
    objective_logger = Logger(args.save_dir, 'objective')
    v_loss_logger = Logger(args.save_dir, 'v_loss')
    kl_logger = Logger(args.save_dir, 'kl')
    entropy_logger = Logger(args.save_dir, 'entropy')
    score_logger = Logger(args.save_dir, 'score')
    eplen_logger = Logger(args.save_dir, 'eplen')

    # train
    observations = vec_env.reset()
    reward_history = [[] for _ in range(args.n_envs)]
    env_cnts = np.zeros(args.n_envs)
    total_step = 0
    save_step = 0
    slack_step = 0
    while total_step < args.total_steps:

        # ======= collect trajectories ======= #
        trajectories = [[] for _ in range(args.n_envs)]
        step = 0
        while step < args.n_steps:
            env_cnts += 1
            step += args.n_envs
            total_step += args.n_envs

            with torch.no_grad():
                obs_tensor = torch.tensor(observations, device=args.device, dtype=torch.float32)
                action_tensor, clipped_action_tensor = agent.getAction(obs_tensor, True)
                actions = action_tensor.detach().cpu().numpy()
                clipped_actions = clipped_action_tensor.detach().cpu().numpy()
            next_observations, rewards, dones, infos = vec_env.step(clipped_actions)

            for env_idx in range(args.n_envs):
                reward_history[env_idx].append(rewards[env_idx])
                dones[env_idx] = True if env_cnts[env_idx] >= args.max_episode_steps else dones[env_idx]
                fail = env_cnts[env_idx] < args.max_episode_steps if dones[env_idx] else False
                next_observation = infos[env_idx]['terminal_observation'] if dones[env_idx] else next_observations[env_idx]
                trajectories[env_idx].append([observations[env_idx], actions[env_idx], rewards[env_idx], dones[env_idx], fail, next_observation])

                if dones[env_idx]:
                    ep_len = len(reward_history[env_idx])
                    score = np.sum(reward_history[env_idx])

                    score_logger.write([ep_len, score])
                    eplen_logger.write([ep_len, ep_len])

                    reward_history[env_idx] = []
                    env_cnts[env_idx] = 0

            observations = next_observations
        # ==================================== #

        v_loss, objective, kl, entropy = agent.train(trajs=trajectories)

        objective_logger.write([step, objective])
        v_loss_logger.write([step, v_loss])
        kl_logger.write([step, kl])
        entropy_logger.write([step, entropy])
        log_data = {
            "rollout/score": score_logger.get_avg(args.n_envs), 
            "rollout/ep_len": eplen_logger.get_avg(args.n_envs),
            "train/value_loss":v_loss_logger.get_avg(), 
            "train/objective":objective_logger.get_avg(), 
            "train/kl":kl_logger.get_avg(), 
            "train/entropy":entropy_logger.get_avg(),
        }
        if args.wandb:
            wandb.log(log_data)
        print(log_data)

        if args.slack and total_step - slack_step >= args.slack_freq:
            slackbot.sendMsg(f"{project_name}\nname: {wandb.run.name}\nsteps: {total_step}\nlog: {log_data}")
            slack_step += args.slack_freq

        if total_step - save_step >= args.save_freq:
            save_step += args.save_freq
            agent.save()
            objective_logger.save()
            v_loss_logger.save()
            entropy_logger.save()
            kl_logger.save()
            score_logger.save()
            eplen_logger.save()

    vec_env.close()


def test(args):
    # define env
    env = gym.make(args.env_name)
    obs_rms = RunningMeanStd(args.save_dir, env.observation_space.shape[0])

    # set args value for env
    args.obs_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.action_bound_min = env.action_space.low
    args.action_bound_max = env.action_space.high

    # define agent
    agent = Agent(args)

    # episodes = int(1e6)
    episodes = int(10)

    for episode in range(episodes):
        obs = env.reset()
        obs = obs_rms.normalize(obs)
        env.render()

        done = False
        score = 0
        while not done:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, device=args.device, dtype=torch.float32)
                action_tensor, clipped_action_tensor = agent.getAction(obs_tensor, False)
                action = action_tensor.detach().cpu().numpy()
                clipped_action = clipped_action_tensor.detach().cpu().numpy()
            obs, reward, done, info = env.step(clipped_action)
            obs = obs_rms.normalize(obs)

            score += reward
            env.render()
            time.sleep(0.001)

        print("score :",score)

if __name__ == "__main__":
    parser = getPaser()
    args = parser.parse_args()

    # ==== processing args ==== #
    # save_dir
    args.save_dir = f"results/{args.name}_s{args.seed}"
    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{args.gpu_idx}"
    # device
    if torch.cuda.is_available() and args.device == 'gpu':
        device = torch.device('cuda:0')
        print('[torch] cuda is used.')
    else:
        device = torch.device('cpu')
        print('[torch] cpu is used.')
    args.device = device
    # ========================= #

    if args.test:
        test(args)
    else:
        train(args)
