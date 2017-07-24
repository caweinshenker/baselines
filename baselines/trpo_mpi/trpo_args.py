import argparse
from sacred import Experiment

ex = Experiment('TRPO_MPI')

@ex.config
def base_args():
    env_id = 'Hopper-v1'
    num_timesteps = 1000000
    seed = 0
    render = False


@ex.config
def trpo_args():
    timesteps_per_batch = 1024
    max_kl = 0.01
    cg_iters = 10
    cg_damping = 0.1
    gamma = 0.99
    lam = 0.98
    vf_iters = 5
    vf_stepsize = 0.01
