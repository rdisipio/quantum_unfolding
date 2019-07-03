#!/usr/bin/env python3

import os
import sys
import argparse
import datetime as dt
import numpy as np
from scipy import stats
from scipy.spatial.distance import hamming

from decimal2binary import *
from input_data import *

# DWave stuff
import dimod
import neal

from dwave.system import EmbeddingComposite, FixedEmbeddingComposite, TilingComposite, DWaveSampler
from dwave_tools import get_embedding_with_short_chain, get_energy, anneal_sched_custom, make_reverse_anneal_schedule

np.set_printoptions(precision=1, linewidth=200, suppress=True)

parser = argparse.ArgumentParser("Quantum unfolding optimizer")
parser.add_argument('-n', '--max_evals', default=10)
parser.add_argument('-d', '--dry-run', action='store_true', default=False)
args = parser.parse_args()

hardware_sampler = DWaveSampler()

# constants
N = x.shape[0]
D = laplacian(N)

# parameters to be optimized
lmbd = 0.  # regularization strength
num_reads = 1000  # number of reads
n = 4  # number of bits
annealing_time = 20.
max_evals = int(args.max_evals)


#x_b = discretize_vector(x, n)
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
for j in range(n*N):
    h[(j)] = 0
    for i in range(N):
        h[(j)] += (
            R_b[i][j]*R_b[i][j] -
            2*R_b[i][j] * d[i] +
            lmbd*D_b[i][j]*D_b[i][j]
        )
    S[(j, j)] = h[(j)]

# quadratic constraints
for j in range(n*N):
    for k in range(j+1, n*N):
        J[(j, k)] = 0
        for i in range(N):
            J[(j, k)] += 2*(R_b[i][j]*R_b[i][k] + lmbd*D_b[i][j]*D_b[i][k])
        S[(j, k)] = J[(j, k)]

bqm = dimod.BinaryQuadraticModel(linear=h,
                                 quadratic=J,
                                 offset=0.0,
                                 vartype=dimod.BINARY)

energy_true_z = get_energy(bqm, z_b)

embedding = get_embedding_with_short_chain(S,
                                           tries=5,
                                           processor=hardware_sampler.edgelist,
                                           verbose=False)

sampler = FixedEmbeddingComposite(hardware_sampler, embedding)

solver_parameters = {'num_reads': num_reads,
                     'auto_scale': True,
                     #'annealing_time': annealing_time,  # default: 20 us
                     #'anneal_schedule': anneal_sched_custom(id=3),
                     #'num_spin_reversal_transforms': 2,  # default: 2
                     }

schedule = make_reverse_anneal_schedule(s_target=0.99,
                                        hold_time=1,
                                        ramp_up_slope=0.2)

neal_sampler = neal.SimulatedAnnealingSampler()


print("INFO: solving initial state")
best_fit = sampler.sample(bqm, **solver_parameters).aggregate().first
energy_bestfit = best_fit.energy
q = np.array(list(best_fit.sample.values()))
y = compact_vector(q, n)
min_hamming = hamming(z_b, q)

neal_solution = neal_sampler.sample(bqm, num_reads=num_reads).aggregate().first
neal_energy = neal_solution.energy
neal_q = np.array(list(neal_solution.sample.values()))
neal_y = compact_vector(neal_q, n)
neal_hamm = hamming(z_b, neal_q)

print("INFO: initial solution:", q,
      "::", y, ":: E=", energy_bestfit)
print("INFO: neal solution:", neal_q, "::", neal_y, ":: E =", neal_energy)
print("INFO: truth value:  ", z_b, "::", z, ":: E =", energy_true_z)
print("INFO: hamming distance:", min_hamming)
print(" --- ")

for itrial in range(max_evals):
    reverse_anneal_params = dict(anneal_schedule=schedule,
                                 initial_state=best_fit.sample,
                                 reinitialize_state=True)

    solver_parameters = {'num_reads': num_reads,
                         'auto_scale': True,
                         **reverse_anneal_params,
                         }

    print("INFO: refine solution: attempt %i/%i" % (itrial+1, max_evals))
    this_result = sampler.sample(
        bqm, **solver_parameters).aggregate().first
    this_energy = this_result.energy
    this_q = np.array(list(this_result.sample.values()))
    this_y = compact_vector(this_q, n)
    this_hamm = hamming(z_b, this_q)

    print("INFO: this solution:", this_q,
          "::", this_y, ":: E=", this_energy)
    print("INFO: truth value:  ", z_b, "::", z, ":: E =", energy_true_z)
    print("INFO: hamming distance:", this_hamm)

    if this_hamm < min_hamming:
        best_fit = this_result
        energy_bestfit = this_energy
        q = this_q
        y = this_y
        min_hamming = this_hamm
        print("INFO: improved!")

    print(" --- ")

print("INFO: Final results:")

from scipy import stats
dof = N - 1
chi2, p = stats.chisquare(y, z, dof)
chi2dof = chi2 / float(dof)

from sklearn.metrics import accuracy_score
score = accuracy_score(z_b, q)

print("INFO: best-fit:     ", q, "::", y, ":: E =", energy_bestfit)
print("INFO: neal solution:", neal_q, "::", neal_y, ":: E =", neal_energy)
print("INFO: truth value:  ", z_b, "::", z, ":: E =", energy_true_z)
print("INFO: accuracy:     ", score)
print("INFO: chi2/dof = %.2f" % chi2dof)
print("INFO: hamming distance:", min_hamming)
