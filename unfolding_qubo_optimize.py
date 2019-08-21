#!/usr/bin/env python3

import os
import sys
import argparse
import datetime as dt
import numpy as np
import pandas as pd
import hyperopt as hp
from scipy import stats
from scipy.spatial.distance import hamming

from decimal2binary import *
from input_data import *

# DWave stuff
import dimod
from dwave.system import EmbeddingComposite, FixedEmbeddingComposite, TilingComposite, DWaveSampler
from dwave_tools import get_embedding_with_short_chain, get_energy, anneal_sched_custom, merge_substates

np.set_printoptions(precision=1, linewidth=200, suppress=True)

parser = argparse.ArgumentParser("Quantum unfolding optimizer")
parser.add_argument('-n', '--max_evals', default=20)
parser.add_argument('-d', '--dry-run', action='store_true', default=False)
args = parser.parse_args()

hardware_sampler = DWaveSampler()

# parameters to be optimized
lmbd = 0.  # regularization strength
nreads = 10  # number of reads
n = 4  # number of bits

# constants
N = x.shape[0]
D = laplacian(N)


def objective(args):
    lmdb = args['lmbd']
    num_reads = args['num_reads']
    n = args['num_bits']
    annealing_time = args['annealing_time']

    x_b = discretize_vector(x, n)
    z_b = discretize_vector(z, n)
    d_b = discretize_vector(d, n)
    R_b = discretize_matrix(R0, n)
    D_b = discretize_matrix(D, n)

    # Create QUBO operator
    #Q = np.zeros([n*N, n*N])
    S = {}
    h = {}
    J = {}

    # linear constraints
    for j in range(n * N):
        h[(j)] = 0
        for i in range(N):
            h[(j)] += (R_b[i][j] * R_b[i][j] - 2 * R_b[i][j] * d[i] +
                       lmbd * D_b[i][j] * D_b[i][j])
        S[(j, j)] = h[(j)]

    # quadratic constraints
    for j in range(n * N):
        for k in range(j + 1, n * N):
            J[(j, k)] = 0
            for i in range(N):
                J[(j, k)] += 2 * (R_b[i][j] * R_b[i][k] +
                                  lmbd * D_b[i][j] * D_b[i][k])
            S[(j, k)] = J[(j, k)]

    bqm = dimod.BinaryQuadraticModel(linear=h,
                                     quadratic=J,
                                     offset=0.0,
                                     vartype=dimod.BINARY)

    embedding = get_embedding_with_short_chain(
        S, tries=5, processor=hardware_sampler.edgelist, verbose=False)

    sampler = FixedEmbeddingComposite(hardware_sampler, embedding)
    solver_parameters = {
        'num_reads': num_reads,
        'auto_scale': True,
        'annealing_time': annealing_time,  # default: 20 us
        #'anneal_schedule': anneal_sched_custom(id=3),
        'num_spin_reversal_transforms': 2,  # default: 2
    }
    results = sampler.sample(bqm, **solver_parameters).aggregate()
    best_fit = results.first
    energy_bestfit = best_fit.energy
    q = np.array(list(best_fit.sample.values()))
    y = compact_vector(q, n)

    dof = N - 1
    chi2, p = stats.chisquare(y, z, dof)
    chi2dof = chi2 / float(dof)

    hamm = hamming(z_b, q)

    return {
        'loss': hamm,  # chi2dof,
        'status': hp.STATUS_OK,
        'diffxs': y,
        'q': q,
        'hamming': hamm,
        'lmbd': lmbd,
        'num_reads': num_reads,
        'num_bits': n,
        'annealing_time': annealing_time,
    }


max_evals = int(args.max_evals)

space = hp.hp.choice(
    'unfolder',
    [{
        'lmbd': hp.hp.choice('lmbd', [0.0, 0.5, 1.0]),
        'num_reads': hp.hp.choice('num_reads', [100, 500, 1000]),
        'num_bits': hp.hp.choice('num_bits', [4, 8]),
        'annealing_time': hp.hp.choice('annealing_time', [20, 50, 100]),
    }])

tpe_algo = hp.tpe.suggest
tpe_trials = hp.Trials()
bestfit = hp.fmin(fn=objective,
                  space=space,
                  algo=tpe_algo,
                  trials=tpe_trials,
                  max_evals=max_evals)
print(bestfit)

for trial in tpe_trials:
    print("Trial:")
    print(trial)
    print(" --- ")

# results = pd.DataFrame({'loss': [r['loss'] for r in tpe_trials.results],
#                        'iteration': tpe_trials.idxs_vals[0]['x'],
#                        'x': tpe_trials.idxs_vals[1]['x']}
#                      )
# print(results.head())
