import argparse
from sacred import Experiment

ex = Experiment('PPOSGD_MPI')

@ex.config
def base_args():
    env_id = 'Hopper-v1'
    num_timesteps = 1000000
    seed = 0
    render = False


@ex.config
def pposgd_args():
    timesteps_per_batch=2048
    clip_param=0.2
    entcoeff=0.0
    optim_epochs=10
    optim_stepsize=3e-4
    optim_batchsize=64
    gamma=0.99
    lam=0.95
