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
xedges = np.linspace(xmin, xmax, n_bins+1)  # for uniform binning

# truth-level:
x = [5, 10, 3]

# response matrix:
R = [[5, 1, 0],
     [1, 3, 1],
     [0, 1, 2]]

# pseudo-data:
#d = [32, 40, 15]
d = np.dot(R, x)

# convert inputs to appropriate format

print("INFO: bin edges (%i):" % n_bins)
print(xedges)

x = np.array(x, dtype='uint8')
x_b = np.unpackbits(x)
print("INFO: truth level:")
print(x)
print(x_b)

d = np.array(d, dtype='uint8')
d_b = np.unpackbits(d)
print("INFO: pseudo-data:")
print(d)
print(d_b)

# Response matrix

R = np.array(R, dtype='uint8')

R_b = d2b(R)
print("INFO: Response matrix:")
print(R)
print("INFO: R binary representation:", R_b.shape)
print(R_b)

params = [d_b, R_b]
# initial guess
x_0 = np.random.randint(0, int_max, size=(n_bins), dtype='uint8')
res = optimize.minimize(lh.log_gauss,
                        x_0,
                        args=params,
                        method='Powell')

print(res)

x_star = np.array([int(n) for n in res.x], dtype='uint8')
x_star_b = np.unpackbits(x_star)
print("INFO: bestfit:")
print(x_star)
print(x_star_b)
print("INFO: truth level:")
print(x_b)
print(x)

exit(0)

# Now loop to find the minimum of the likelihood (brute force approach)
n_itr = 500000
k_min = -1
logL_min = 1000000000000
x_hat_bestfit_b = None
for k in range(n_itr):
    # generate decimal, then convert to binary...
    #x_hat = np.random.randint(0, int_max, size=(n_bins), dtype='uint8')
    #x_hat_b = np.unpackbits(x_hat)

    # or generate binary, then convert to decimal
    x_hat_b = np.random.randint(0, 2, size=(n_bins*n_bits), dtype='uint8')

    # calculate likelihood
    logL = lh.log_gauss(y_hat_b, data=d_b, resp=R_b)
    #print(k, logL_min, logL)

    # update minimum
    if logL > logL_min:
        continue

    print(k, logL_min, logL)

    logL_min = logL
    k_min = k
    x_hat_bestfit_b = np.array(x_hat_b, dtype='uint8')

    if logL == 0:
        break

print("INFO: logL minimum found at iteration %i:" % k_min)
x_hat_bestfit = np.packbits(x_hat_bestfit_b)
print("INFO: bestfit:")
print(x_hat_bestfit_b)
print(x_hat_bestfit)
print("INFO: truth level:")
print(x_b)
print(x)
