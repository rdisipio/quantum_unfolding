#!/usr/bin/env python3

import os
import sys
import argparse
import datetime as dt
import numpy as np
from scipy import stats
from scipy.spatial.distance import hamming
from scipy import stats
import dimod
import neal
from dwave.system import EmbeddingComposite, FixedEmbeddingComposite, TilingComposite, DWaveSampler

from quantum_unfolding import anneal_sched_custom, get_embedding_with_short_chain, get_energy, make_reverse_anneal_schedule

np.set_printoptions(precision=1, linewidth=200, suppress=True)

parser = argparse.ArgumentParser("Quantum unfolding optimizer")
parser.add_argument('-n', '--max_evals', default=10)
parser.add_argument('-d', '--dry-run', action='store_true', default=False)
args = parser.parse_args()

hardware_sampler = DWaveSampler()

# constants
N = x.shape[0]
D = laplacian(N)
dof = N - 1

# parameters to be optimized
lmbd = 0.  # regularization strength
num_reads = 100  # number of reads
n = 4  # number of bits
annealing_time = 100.  # us
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

energy_true_z = get_energy(bqm, z_b)

embedding = get_embedding_with_short_chain(S,
                                           tries=5,
                                           processor=hardware_sampler.edgelist,
                                           verbose=False)

sampler = FixedEmbeddingComposite(hardware_sampler, embedding)

solver_parameters = {
    'num_reads': num_reads,
    'auto_scale': True,
    'annealing_time': annealing_time,  # default: 20 us
    #'anneal_schedule': anneal_sched_custom(id=3),
    #'num_spin_reversal_transforms': 2,  # default: 2
}

# schedule = make_reverse_anneal_schedule(s_target=0.99,
#                                        hold_time=1,
#                                        ramp_up_slope=0.2)
#schedule = [(0.0, 1.0), (1.0, 0.7), (7.0, 0.7), (10.0, 1.0)]
#schedule = [(0.0, 1.0), (2.0, 0.7), (8.0, 0.7), (10.0, 1.0)]
schedule = [(0.0, 1.0), (2.0, 0.7), (98.0, 0.7), (100.0, 1.0)]
print("INFO: reverse annealing schedule:")
print(schedule)

neal_sampler = neal.SimulatedAnnealingSampler()

print("INFO: solving initial state")
best_fit = sampler.sample(bqm, **solver_parameters).aggregate().first
energy_bestfit = best_fit.energy
q = np.array(list(best_fit.sample.values()))
y = compact_vector(q, n)
min_hamming = hamming(z_b, q)
best_chi2, best_p = stats.chisquare(y, z, dof)

neal_solution = neal_sampler.sample(bqm, num_reads=num_reads).aggregate().first
neal_energy = neal_solution.energy
neal_q = np.array(list(neal_solution.sample.values()))
neal_y = compact_vector(neal_q, n)
neal_hamm = hamming(z_b, neal_q)

print("INFO: initial solution:", q, "::", y, ":: E=", energy_bestfit)
print("INFO: neal solution:", neal_q, "::", neal_y, ":: E =", neal_energy)
print("INFO: truth value:  ", z_b, "::", z, ":: E =", energy_true_z)
print("INFO: hamming distance:", min_hamming)
print(" --- ")

for itrial in range(max_evals):
    #s = 1.00 - 1.5 * min_hamming
    #schedule = [(0, 1.0), (2, s), (8, s), (10, 1.0)]

    reverse_anneal_params = dict(anneal_schedule=schedule,
                                 initial_state=best_fit.sample,
                                 reinitialize_state=True)

    solver_parameters = {
        'num_reads': num_reads,
        'auto_scale': True,
        **reverse_anneal_params,
    }

    print("INFO: refine solution: attempt %i/%i" % (itrial + 1, max_evals))
    print("INFO: schedule:", schedule)
    this_result = sampler.sample(bqm, **solver_parameters).aggregate().first
    this_energy = this_result.energy
    this_q = np.array(list(this_result.sample.values()))
    this_y = compact_vector(this_q, n)
    this_hamm = hamming(z_b, this_q)

    this_chi2, this_p = stats.chisquare(this_y, z, dof)

    print("INFO: this solution:", this_q, "::", this_y, ":: E =", this_energy,
          ":: hamm =", this_hamm, "chi2/dof = %.3f" % (this_chi2 / float(dof)))
    print("INFO: best solution:", q, "::", y, ":: E =", energy_bestfit,
          ":: hamm =", min_hamming,
          "chi2/dof = %.3f" % (best_chi2 / float(dof)))
    print("INFO: truth value:  ", z_b, "::", z, ":: E =", energy_true_z)

    # if this_hamm < min_hamming: # makes no sense in real life where truth is unknown!
    if this_energy < energy_bestfit:
        best_fit = this_result
        energy_bestfit = this_energy
        q = this_q
        y = this_y
        min_hamming = this_hamm
        best_chi2 = this_chi2
        best_p = this_p
        print("INFO: improved!")

    print(" --- ")

print("INFO: Final results:")

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
