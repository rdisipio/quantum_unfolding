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
import dwave_networkx as dnx
import neal
from dwave_qbsolv import QBSolv

from dwave.system import EmbeddingComposite, FixedEmbeddingComposite, TilingComposite, DWaveSampler
from dwave_tools import get_embedding_with_short_chain, get_energy, anneal_sched_custom, merge_substates

np.set_printoptions(precision=1, linewidth=400, suppress=True)

parser = argparse.ArgumentParser("Quantum unfolding")
parser.add_argument('-l', '--lmbd', default=0)
parser.add_argument('-n', '--nreads', default=100)
parser.add_argument('-b', '--backend', default='sim')  # [cpu, sim, qpu, hyb, qbs]
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

lmbd = float(args.lmbd)  # regularization strength
# lmbd = np.uint8(lmbd)
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
Q = np.zeros([n*N, n*N])
S = {}

# linear constraints
h = {}
for j in range(n*N):
    idx = (j)
    h[idx] = 0
    for i in range(N):
        h[idx] += (
            R_b[i][j]*R_b[i][j] -
            2*R_b[i][j] * d[i] +
            lmbd*D_b[i][j]*D_b[i][j]
        )
    Q[j][j] = h[idx]
    S[(j, j)] = h[idx]

# quadratic constraints
J = {}
for j in range(n*N):
    for k in range(j+1, n*N):
        idx = (j, k)
        J[idx] = 0
        for i in range(N):
            J[idx] += 2*(R_b[i][j]*R_b[i][k] + lmbd*D_b[i][j]*D_b[i][k])
        Q[j][k] = J[idx]
        S[idx] = J[idx]
print("INFO: QUBO coefficients:")
print(Q)

# QUBO
# bqm = dimod.BinaryQuadraticModel(linear=h,
#                                 quadratic=J,
#                                 offset=0.0,
#                                 vartype=dimod.BINARY)

bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(Q)

print("INFO: solving the QUBO model (size=%i)..." % len(bqm))

result = None
if args.backend == 'cpu':
    print("INFO: running on CPU...")
    if not dry_run:
        result = dimod.ExactSolver().sample(bqm)

elif args.backend in ['qpu', 'hyb', 'qbs']:
    print("INFO: running on QPU")

    hardware_sampler = DWaveSampler()
    #C16 = dnx.chimera_graph(16)
    #P16 = dnx.pegasus_graph(16) # experimental

    print("INFO: finding optimal minor embedding...")
    embedding = get_embedding_with_short_chain(S,
                                               tries=5,
                                               processor=hardware_sampler.edgelist,
                                               verbose=True)
    if embedding == None:
        print("ERROR: could not find embedding")
        exit(0)

    print("INFO: creating DWave sampler...")
    sampler = FixedEmbeddingComposite(hardware_sampler, embedding)

    solver_parameters = {'num_reads': num_reads,
                         #'postprocess':   'sampling', # seems very bad!
                         #'postprocess':  'optimization',
                         'auto_scale': True,
                         'annealing_time': 10,  # default: 20 us
                         #'anneal_schedule': anneal_sched_custom(id=3),
                         'num_spin_reversal_transforms': 2,  # default: 2
                         #'chain_strength' : 50000,
                         #'chain_break_fraction':0.33,
                         #'chain_break_method':None,
                         }

    print("INFO: annealing (n_reads=%i) ..." % num_reads)
    if not dry_run:
        if args.backend == 'qpu':
            print("INFO: Using QPU")
            #results = sampler.sample(bqm, **solver_parameters).aggregate()
            results = sampler.sample_qubo(S, **solver_parameters).aggregate()
        elif args.backend == 'qbs':
            print("INFO: using QBsolve with FixedEmbeddingComposite")
            results = QBSolv().sample_qubo(S, solver=sampler, solver_limit=5)
        elif args.backend == 'hyb':
            print("INFO: hybrid")
            import hybrid

            # Define the workflow
            # hybrid.EnergyImpactDecomposer(size=len(bqm), rolling_history=0.15)
            iteration = hybrid.RacingBranches(
                #hybrid.Identity(),
                hybrid.InterruptableTabuSampler(),
                hybrid.EnergyImpactDecomposer(size=len(bqm)//2, rolling=True)
                | hybrid.QPUSubproblemAutoEmbeddingSampler(num_reads=num_reads)
                | hybrid.SplatComposer()
            ) | hybrid.ArgMin()
            workflow = hybrid.LoopUntilNoImprovement(iteration, convergence=3)
            #workflow = hybrid.Loop(iteration, max_iter=20, convergence=3)

            # show execution profile
            init_state = hybrid.State.from_problem(bqm)
            results = workflow.run(init_state).result().samples
            print("INFO: timing:")
            workflow.timers
            hybrid.print_structure(workflow)
            hybrid.profiling.print_counters(workflow)

elif args.backend == 'sim':
    print("INFO: running on simulated annealer (neal)")

    sampler = neal.SimulatedAnnealingSampler()

    print("INFO: annealing (n_reads=%i) ..." % num_reads)
    if not dry_run:
        results = sampler.sample(bqm, num_reads=num_reads).aggregate()

print("INFO: ...done.")

if dry_run:
    print("INFO: dry run.")
    exit(0)

#r_energy = np.array([0.]*len(result))
#w_energy = np.array([0.]*len(result))
#i = 0
# for res in result.record:
#    r_energy[i] = res.energy
#    w_energy[i] = res.num_occurrences
# h_energy = np.histogram(
#    r_energy, bins=np.linspace(-5500, -3500, num=11), weights=w_energy)
#print("INFO: energy histogram:")
# print(h_energy)

print(results)
#for res in results.record:
#    #if res.energy > -5400: continue
#    print(res.energy, res.num_occurrences)

best_fit = results.first

energy_bestfit = best_fit.energy
q = np.array(list(best_fit.sample.values()))
y = compact_vector(q, n)
energy_true_x = get_energy(bqm, x_b)
energy_true_z = get_energy(bqm, z_b)

from scipy import stats
dof = N - 1
chi2, p = stats.chisquare(y, z, dof)
chi2dof = chi2 / float(dof)

print("INFO: best-fit:   ", q, "::", y, ":: E =",
      energy_bestfit, ":: chi2/dof = %.2f" % chi2dof)
print("INFO: truth value:", z_b, "::", z, ":: E =", energy_true_z)

from sklearn.metrics import accuracy_score
score = accuracy_score(z_b, q)
print("INFO: accuracy:", score)

from scipy.spatial.distance import hamming
hamm = hamming(z_b, q)
print("Hamming:", hamm)

print("INFO: add the following line to the list of unfolded results")
print(list(y), end='')
print(', # E =', energy_bestfit, "chi2/dof = %.2f" % chi2dof)
