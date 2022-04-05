# Off-Policy TRPO

This is a simple implemenation of off-policy TRPO ([link](https://ieeexplore.ieee.org/document/9334437)).

## requirement

- python 3.7 or greater
- gym
- mujoco-py (https://github.com/openai/mujoco-py)
- stable-baselines3
- torch==1.10.0 or greater
- requests
- wandb

## results

### HalfCheetah-v2
![img](./imgs/Half-Cheetah_score%26v_loss%26entropy.png)
- obtained by training with three seeds.
- {algo_name}-Norm: training with state normalization.
