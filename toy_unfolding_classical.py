#!/usr/bin/env python3
import numpy as np
from decimal2binary import binary_matmul, d2b, get_int_max
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

# Now loop to find the minimum of the likelihood (brute force approach)
n_iterations = 500000
logL_min = np.finfo('d').max
for k in range(n_iterations):
    # generate decimal, then convert to binary...
    x_hat_b = d2b(np.random.randint(0, get_int_max(n_bits), size=(n_bins), dtype='uint8'))

    # or generate binary, then convert to decimal
    # x_hat_b = np.random.randint(0, 2, size=(n_bins*n_bits), dtype='uint8')

    # apply response to obtain reco-level prediction
    y_hat_b = binary_matmul(R_b, x_hat_b)

    # calculate likelihood
    logL = lh.log_gauss(d_b, y_hat_b)
    # logL, r = lh.test_lhood(d_b, y_hat_b)
    # print(k, logL_min, logL)

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
