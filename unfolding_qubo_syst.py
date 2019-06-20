#!/usr/bin/env python3

import os
import sys
import argparse
import datetime as dt
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from decimal2binary import *
from input_data import *

# DWave stuff
import dimod
from dwave.system import EmbeddingComposite, FixedEmbeddingComposite, DWaveSampler
from dwave_tools import get_embedding_with_short_chain, get_energy, anneal_sched_custom
import neal

np.set_printoptions(precision=1, linewidth=200, suppress=True)

parser = argparse.ArgumentParser("Quantum unfolding")
parser.add_argument('-l', '--lmbd', default=0.00)
parser.add_argument('-n', '--nreads', default=5000)
parser.add_argument('-b', '--backend', default='sim')  # [qpu, sim, hyb]
parser.add_argument('-d', '--dry-run', action='store_true', default=False)
args = parser.parse_args()

num_reads = int(args.nreads)
dry_run = bool(args.dry_run)
if dry_run:
    print("WARNING: dry run. There will be no results at the end.")

n = 4
N = x.shape[0]

print("INFO: N bins:", Nbins)
print("INFO: N syst:", Nsyst)
print("INFO: Nparams = Nbins + Nsyst:", Nparams)
print("INFO: n-bits encoding:", n)

# lmbd = np.uint8(args.lmbd)  # regularization strength
lmbd = float(args.lmbd)
D = laplacian(Nbins)

# in case Nsyst>0, extend vectors and laplacian
D = np.block([[D, np.zeros([Nbins, Nsyst])]])  # ,
#              [np.zeros([Nsyst, Nbins]), np.zeros([Nsyst, Nsyst])]])
x = np.hstack((x, np.zeros(Nsyst)))
z = np.hstack((z, s))
d = np.dot(R, z)
# y = np.dot(R, x)

# convert to bits
x_b = discretize_vector(x, n)
z_b = discretize_vector(z, n)
d_b = discretize_vector(d, n)
R_b = discretize_matrix(R, n)
D_b = discretize_matrix(D, n)

print("INFO: Signal truth-level x:")
print(x, x_b)
print("INFO: Pseudo-data truth-level z:")
print(z, z_b)
print("INFO: Response matrix R (incl syst):")
print(R)
print(R_b)
print("INFO: pseudo-data d = R.z:")
print(d, d_b)
print("INFO: Laplacian operator:")
print(D)
print(D_b)
print("INFO: regularization strength:", lmbd)

# Create QUBO operator
Q = np.zeros([n*Nparams, n*Nparams])

# linear constraints
h = {}
for j in range(n*Nparams):
    idx = (j)
    h[idx] = 0
    for i in range(Nbins):
        h[idx] += (R_b[i][j]*R_b[i][j] -
                   2*R_b[i][j] * d[i] +
                   lmbd*D_b[i][j]*D_b[i][j])
    Q[j][j] = h[idx]

# quadratic constraints
J = {}
for j in range(n*Nparams):
    for k in range(j+1, n*Nparams):
        idx = (j, k)
        J[idx] = 0
        for i in range(Nbins):
            J[idx] += 2*(R_b[i][j]*R_b[i][k] + lmbd*D_b[i][j]*D_b[i][k])
        Q[j][k] = J[idx]
print("INFO: QUBO coefficients:")
print(Q)

# QUBO
# bqm = dimod.BinaryQuadraticModel(linear=h,
#                                 quadratic=J,
#                                 offset=0.0,
#                                 vartype=dimod.BINARY)
bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(Q)


print("INFO: solving the QUBO model...")

result = None
if args.backend == 'cpu':
    print("INFO: running on CPU...")
    if not dry_run:
        result = dimod.ExactSolver().sample(bqm)

elif args.backend in [ 'qpu', 'hyb' ]:
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
    sampler = FixedEmbeddingComposite(hardware_sampler, embedding)
    # sampler = EmbeddingComposite(hardware_sampler)  # default

    solver_parameters = {'num_reads': num_reads,
                         #'postprocess':   'sampling',
                         #'postprocess':  'optimization',
                         'auto_scale': True,
                         #'annealing_time': 20,  # default: 20 us
                         #'anneal_schedule': anneal_sched_custom(),
                         'num_spin_reversal_transforms': 2}  # default: 2

    print("INFO: annealing (n_reads=%i) ..." % num_reads)
    if not dry_run:
      if args.backend == 'qpu':
        results = sampler.sample(bqm, **solver_parameters).aggregate()
      elif args.backend == 'hyb':
        import hybrid

        iteration = hybrid.RacingBranches(
            hybrid.Identity(),
            hybrid.InterruptableTabuSampler(),
            hybrid.EnergyImpactDecomposer(size=2)
            | hybrid.QPUSubproblemAutoEmbeddingSampler()
            | hybrid.SplatComposer()
        ) | hybrid.ArgMin()
        workflow = hybrid.LoopUntilNoImprovement(iteration, convergence=3)

        init_state = hybrid.State.from_problem(bqm)
        results = workflow.run(init_state).result().samples

elif args.backend == 'sim':
    print("INFO: running on simulated annealer (neal)")

    sampler = neal.SimulatedAnnealingSampler()

    print("INFO: annealing (n_reads=%i) ..." % num_reads)
    if not dry_run:
        results = sampler.sample(bqm, num_reads=num_reads).aggregate()

print("INFO: ...done.")

if dry_run:
    print("INFO: dry runn.")
    exit(0)

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
score = accuracy_score(x_b, q)
print("INFO: accuracy:", score)

print("INFO: add the following line to the list of unfolded results")
print(list(y), end='')
print(', # E =', energy_bestfit, "chi2/dof = %.2f" % chi2dof)
