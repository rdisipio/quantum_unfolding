#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from decimal2binary import *
import likelihood as lh

np.set_printoptions(precision=1, linewidth=200, suppress=True)

# All numbers are 8 bits long.
# Smaller Numpy dtype is uint8, i.e. [0-255]


######################
# Set the inputs here
######################

# binning:
xmin = 0
xmax = 6
n_bins = 3
xedges = np.linspace(xmin, xmax, n_bins+1)  # for uniform binning

# truth-level:
x = [5, 10, 3]

# response matrix:
R = [[5, 1, 0],
     [1, 3, 1],
     [0, 1, 2]]

# pseudo-data:
d = [32, 40, 15]

# convert inputs to appropriate format

print("INFO: bin edges (%i):" % n_bins)
print(xedges)

x = np.array(x, dtype='uint8')
x_b = np.unpackbits(x)
print("INFO: x binary representation:", x_b.shape)
print(x_b)

d = np.array(d, dtype='uint8')
d_b = np.unpackbits(d)
print("INFO: d binary representation:", d_b.shape)
print(d_b)

# Response matrix

R = np.array(R, dtype='uint8')

R_b = d2b(R)
print("INFO: R binary representation:", R_b.shape)
print(R_b)

# Now loop to find the minimum of the likelihood
n_itr = 5000
k_min = -1
z_min = 1000000000
y_hat_best = None
for k in range(n_itr):
    x_hat = np.random.randint(0, int_max, size=(n_bins), dtype='uint8')
    x_hat_b = np.unpackbits(x_hat)

    y_hat_b = binary_matmul(R_b, x_hat_b)

    z = lh.log_gauss(d_b, y_hat_b)
    #print(k, z)

    if z > z_min:
        continue

    z_min = z
    k_min = k
    y_hat_best_b = np.array(y_hat_b, dtype='uint8')

print("INFO: logL minimum found at iteration %i:" % k_min)
print(y_hat_best_b)
y_hat_best = np.packbits(y_hat_best_b)
print(y_hat_best)
print("INFO: truth level:")
print(x_b)
print(x)
