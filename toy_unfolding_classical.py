#!/usr/bin/env python3
import numpy as np
from scipy import optimize
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
n_bits = 8
xedges = np.linspace(xmin, xmax, n_bins + 1)  # for uniform binning

# truth-level:
x = [5, 10, 3]

# response matrix:
R = [[5, 1, 0], [1, 3, 1], [0, 1, 2]]

# pseudo-data:
#d = [32, 40, 15]
d = np.dot(R, x)

# convert inputs to appropriate format

print("INFO: bin edges (%i):" % n_bins)
print(xedges)

x_b = d2b(x)
print("INFO: truth level:")
print(x)
print(x_b)

d_b = d2b(d)
print("INFO: pseudo-data:")
print(d)
print(d_b)

# Response matrix

R_b = d2b(R)
print("INFO: Response matrix:")
print(R)
print("INFO: R binary representation:", R_b.shape)
print(R_b)

params = [d_b, R_b]
# initial guess
x_0 = np.random.randint(0, int_max, size=(n_bins), dtype='uint8')
res = optimize.minimize(lh.log_gauss, x_0, args=params, method='Powell')

print(res)

x_star = np.array([int(n) for n in res.x], dtype='uint8')
x_star_b = np.unpackbits(x_star)
print("INFO: bestfit:")
print(x_star)
print(x_star_b)
print("INFO: truth level:")
print(x_b)
print(x)
