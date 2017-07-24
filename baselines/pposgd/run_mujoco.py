#!/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
from baselines.pposgd.pposgd_args import base_args, pposgd_args, ex
import os.path as osp
import gym, logging
from baselines import logger
import sys

@ex.automain
def train(env_id, num_timesteps, seed, render, timesteps_per_batch, \
          clip_param, entcoeff, optim_epochs, optim_stepsize, optim_batchsize, \
          gamma, lam):
    from baselines.pposgd import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    logger.session().__enter__()
    set_global_seeds(seed)
    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env = bench.Monitor(env, osp.join(logger.get_dir(), "monitor.json"))
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    pposgd_simple.learn(env, policy_fn, \
            max_timesteps=num_timesteps, \
            timesteps_per_batch=timesteps_per_batch, \
            clip_param=clip_param,\
            entcoeff=entcoeff, \
            optim_epochs=optim_epochs,\
            optim_stepsize=optim_stepsize, \
            optim_batchsize=optim_batchsize, \
            gamma=gamma,\
            lam=lam, \
            render=render
        )
    env.close()

def main():
    train()


if __name__ == '__main__':
    base_args()
    pposgd_args()
    main()
