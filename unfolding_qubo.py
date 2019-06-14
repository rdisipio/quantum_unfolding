#!/usr/bin/env python3

import os
import sys
import argparse
import datetime as dt
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from decimal2binary import *

# DWave stuff
import dimod
from dwave.system import EmbeddingComposite, FixedEmbeddingComposite, DWaveSampler
from dwave_tools import get_embedding_with_short_chain, get_energy, anneal_sched_custom
import neal

np.set_printoptions(precision=1, linewidth=200, suppress=True)

parser = argparse.ArgumentParser("Quantum unfolding")
parser.add_argument('-l', '--lmbd', default=0)
parser.add_argument('-n', '--nreads', default=5000)
parser.add_argument('-b', '--backend', default='sim')  # [cpu, qpu, sim]
parser.add_argument('-d', '--dry-run', action='store_true', default=False)
args = parser.parse_args()

num_reads = int(args.nreads)
dry_run = bool(args.dry_run)
if dry_run:
    print("WARNING: dry run. There will be no results at the end.")

from input_data import *

n = 4
N = x.shape[0]

print("INFO: N bins:", N)
print("INFO: n-bits encoding:", n)

lmbd = int(args.lmbd)
lmbd = np.uint8(lmbd)  # regularization strength
D = laplacian(N)

# convert to bits
x_b = discretize_vector(x, n)
z_b = discretize_vector(z, n)
d_b = discretize_vector(d, n)
R_b = discretize_matrix(R0, n)
D_b = discretize_matrix(D, n)

print("INFO: Signal truth-level x:")
print(x, x_b)
print("INFO: Pseudo-data truth-level x:")
print(z, z_b)
print("INFO: pseudo-data b:")
print(d, d_b)
print("INFO: Response matrix:")
print(R0)
print(R_b)
print("INFO: Laplacian operator:")
print(D)
print(D_b)
print("INFO: regularization strength:", lmbd)

# Create QUBO operator

# linear constraints
h = {}
for j in range(n*N):
    idx = (j)
    h[idx] = 0
    for i in range(N):
        h[idx] += (R_b[i][j]*R_b[i][j] -
                   2*R_b[i][j] * d[i] +
                   lmbd*D_b[i][j]*D_b[i][j])
    #print("h", idx, ":", h[idx])

# quadratic constraints
J = {}
for j in range(n*N):
    for k in range(j+1, n*N):
        idx = (j, k)
        J[idx] = 0
        for i in range(N):
            J[idx] += 2*(R_b[i][j]*R_b[i][k] + lmbd*D_b[i][j]*D_b[i][k])
        #print("J", idx, ":", J[idx])

# QUBO
bqm = dimod.BinaryQuadraticModel(linear=h,
                                 quadratic=J,
                                 offset=0.0,
                                 vartype=dimod.BINARY)
print("INFO: solving the QUBO model...")

result = None
if args.backend == 'cpu':
    print("INFO: running on CPU...")
    if not dry_run:
        result = dimod.ExactSolver().sample(bqm)

elif args.backend == 'qpu':
    print("INFO: running on QPU")

    hardware_sampler = DWaveSampler()

    print("INFO: finding optimal minor embedding...")
    embedding = get_embedding_with_short_chain(J,
                                               tries=5,
                                               processor=hardware_sampler.edgelist,
                                               verbose=True)
    if embedding == None:
        raise("ERROR: could not find embedding")
        exit(0)

    print("INFO: creating DWave sampler...")
    #sampler = FixedEmbeddingComposite(hardware_sampler, embedding)
    sampler = EmbeddingComposite(hardware_sampler)  # default

    solver_parameters = {'num_reads': num_reads,
                         #'postprocess':   'sampling',
                         #'postprocess':  'optimization',
                         'auto_scale': True,
                         #'annealing_time': 20,  # default: 20 us
                         'anneal_schedule': anneal_sched_custom(),
                         'num_spin_reversal_transforms': 2}  # default: 2

    print("INFO: annealing (n_reads=%i) ..." % num_reads)
    if not dry_run:
        result = sampler.sample(bqm, **solver_parameters).aggregate()
        # result = sampler.sample(
        #    bqm, num_reads=num_reads).aggregate()  # default

elif args.backend == 'sim':
    print("INFO: running on simulated annealer (neal)")

    sampler = neal.SimulatedAnnealingSampler()

    print("INFO: annealing (n_reads=%i) ..." % num_reads)
    if not dry_run:
        result = sampler.sample(bqm, num_reads=num_reads).aggregate()

print("INFO: ...done.")

if dry_run:
    print("INFO: dry runn.")
    exit(0)

best_fit = result.first
energy_bestfit = best_fit.energy
q = np.array(list(best_fit.sample.values()))
y = compact_vector(q, n)
energy_true_x = get_energy(bqm, x_b)
energy_true_z = get_energy(bqm, z_b)

print("INFO: best-fit:   ", q, "::", y, ":: E =", energy_bestfit)
print("INFO: truth value:", z_b, "::", z, ":: E =", energy_true_z)

from sklearn.metrics import accuracy_score
score = accuracy_score(x_b, q)
print("INFO: accuracy:", score)

print("INFO: add the following line to the list of unfolded results")
print(list(y), end='')
print(', # E =', energy_bestfit)
