#!/usr/bin/env python3

import os
import sys
import argparse
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats, optimize
from scipy.spatial.distance import hamming
from sklearn.metrics import accuracy_score

from quantum_unfolding import Backends, get_energy, input_data, QUBOUnfolder, R0, sigma_syst, StatusCode

np.set_printoptions(precision=2, linewidth=500, suppress=True)

parser = argparse.ArgumentParser("Quantum unfolding")
parser.add_argument('-o', '--observable', default='peak')
parser.add_argument('-l', '--lmbd', default=0.00)
parser.add_argument('-g', '--gamma', default=1)
parser.add_argument('-n', '--nreads', default=5000)
parser.add_argument('-b', '--backend', default='sim')  # [qpu, sim, hyb]
parser.add_argument('-e', '--encoding', default=4)
parser.add_argument('-f', '--file', default=None)
parser.add_argument('-d', '--dry-run', action='store_true', default=False)
args = parser.parse_args()

obs = args.observable
num_reads = int(args.nreads)
backend = Backends[args.backend]
lmbd = float(args.lmbd)
gamma = float(args.gamma)
dry_run = bool(args.dry_run)
if dry_run:
    print("WARNING: dry run. There will be no results at the end.")

unfolder = QUBOUnfolder()

# Signal (reference MC)
x = input_data[obs]['truth']
y = np.dot(R0, x)  # signal @ reco-level

# Pseudo-data (to be unfolded)
#z = input_data[obs]['pdata']
z = input_data[obs]['truth']  # closure test
d = np.dot(R0, z)  # pseduo-data @ reco-level

print("INFO: pseudo-data (before systs):")
print(d)

n = int(args.encoding)
N = x.shape[0]

print("INFO: N bins:", N)
print("INFO: n-bits encoding:", n)

# Systematic uncertainties:
#dy1 = np.array([3., 3., 3., 3., 3.])  # overall shift
#dy2 = np.array([1., 2., 3., 2., 1.])  # shape change
dy1 = np.array([5., 4., 3., 2., 1.])  # shape change

# strength of systematics in pseudo-data
#sigma_syst = np.array([1.0, -1.0])
#sigma_syst = np.array( [-0.75] )

d = np.add(d, sigma_syst[0] * dy1)
#d = np.add(d, sigma_syst[1] * dy2)

print("INFO: pseudo-data (incl effect of systs):")
print(d)

unfolder.get_data().set_truth(x)
unfolder.get_data().set_response(R0)
unfolder.get_data().set_data(d)
unfolder.set_regularization(lmbd)
unfolder.set_syst_penalty(gamma)
unfolder.set_encoding(n)

unfolder.syst_range = 2.  # +- 2sigma
unfolder.add_syst_1sigma(dy1, n_bits=4)
#unfolder.add_syst_1sigma(dy2, n_bits=4)

unfolder.backend = backend
unfolder.solver_parameters['num_reads'] = num_reads
unfolder.solver_parameters['annealing_time'] = 20  #us

print("INFO: Pseudo-data truth-level x:")
print(z)

print("INFO: solving QUBO problem...")
status = unfolder.solve()
print("INFO: ...done.")
if not status == StatusCode.success:
    print("ERROR: something went wrong during execution.")

if dry_run:
    print("INFO: dry run.")
    exit(0)

y = unfolder.get_unfolded()
z = np.append(z, sigma_syst)

z_b = unfolder._encoder.encode(z)
x_b = unfolder._encoder.encode(x)
bqm = unfolder._bqm
best_fit = unfolder.best_fit
energy_bestfit = best_fit.energy
q = np.array(list(best_fit.sample.values()))
y = unfolder._encoder.decode(q)
energy_true_x = get_energy(bqm, x_b)
energy_true_z = get_energy(bqm, z_b)

dof = N - 1
print("y =", y)
print("z =", z)
chi2, p = stats.chisquare(y, z, dof)
chi2dof = chi2 / float(dof)

print("INFO: best-fit:   ", q, "::", y, ":: E =", energy_bestfit,
      ":: chi2/dof = %.2f" % chi2dof)
print("INFO: truth value:", z_b, "::", z, ":: E =", energy_true_z)

score = accuracy_score(z_b, q)
print("INFO: accuracy:", score)

hamm = hamming(z_b, q)
print("Hamming:", hamm)

print("INFO: add the following line to the list of unfolded results")
print(list(y), end='')
print(', # E =', energy_bestfit, "chi2/dof = %.2f" % chi2dof)

if not args.file == None:
    f = open(args.file, 'a')
    np.savetxt(f, y.reshape(1, y.shape[0]), fmt="%f", delimiter=",")
