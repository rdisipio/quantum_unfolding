#!/usr/bin/env python3
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from decimal2binary import *
import dimod

np.set_printoptions(precision=1, linewidth=200, suppress=True)

# truth-level:
x = [5, 10, 3]

# response matrix:
R = [[3, 1, 0], [1, 3, 1], [0, 1, 2]]

# pseudo-data:
d = [32, 40, 15]

# convert to numpy arrays
x = np.array(x)
R = np.array(R)
d = np.array(d)

# closure test
d = np.dot(R, x)

n = 4
N = x.shape[0]

print("INFO: N bins:", N)
print("INFO: n-bits encoding:", n)

lmbd = 1.  # regularization strength
L = laplacian(N)

# convert to bits
x_b = discretize_vector(x)
d_b = discretize_vector(d)
R_b = discretize_matrix(R)
L_b = discretize_matrix(L)

print("INFO: Truth-level x:")
print(x, x_b)
print("INFO: pseudo-data d:")
print(d, d_b)
print("INFO: Response matrix:")
print(R)
print(R_b)
print("INFO: Laplacian operator:")
print(L)
print(L_b)

# Create QUBO operator

# linear constraints
h = {}
for j in range(n*N):
    idx = (j)
    h[idx] = 0
    for i in range(N):
        h[idx] += (R_b[i][j]*R_b[i][j] -
                   2*R_b[i][j] * d_b[i] + lmbd*L_b[i][j]*L_b[i][j])
    print("h", idx, ":", h[idx])

# quadratic constraints
J = {}
for j in range(n*N):
    for k in range(j+1, n*N):
        idx = (j, k)
        J[idx] = 0
        for i in range(N):
            J[idx] += 2*(R_b[i][j]*R_b[i][k] + lmbd*L_b[i][k]*L_b[i][k])
        print("J", idx, ":", J[idx])

# QUBO
bqm = dimod.BinaryQuadraticModel(linear=h,
                                 quadratic=J,
                                 offset=0.0,
                                 vartype=dimod.BINARY)
result = dimod.ExactSolver().sample(bqm)
energy_min = 1e10
q = None
for sample, energy in result.data(['sample', 'energy']):
    if energy > energy_min:
        continue
    energy_min = energy
    q = sample.values()
    #print(sample, energy)
print(q, energy_min)
