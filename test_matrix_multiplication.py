#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from decimal2binary import *

np.set_printoptions(precision=1, linewidth=200, suppress=True)

# All numbers are 4 bits long, i.e. between 0 (0b0000) and 15 (0b1111)

xmin = 0
xmax = 6
Nbins = 3
xedges = np.linspace(xmin, xmax, Nbins+1)

print("INFO: bin edges (%i):" % Nbins)
print(xedges)

# Set here the truth-level distribution
# smaller dtype is uint8, i.e. [0-255]
x = [5, 10, 3]
x = np.array(x, dtype='uint8')  # (3)
print("INFO: x decimal representation:", x.shape)
print(x)

# convert to bit representation
x_b = np.unpackbits(x)  # (4)*8 = (24)
print("INFO: x binary representation:", x_b.shape)
print(x_b)

# Response matrix
R = [[5, 1, 0],
     [1, 3, 1],
     [0, 1, 2]]
R = np.array(R, dtype='uint8')  # (3,3)
print("INFO: Response matrix:", R.shape)
print(R)

R_b = d2b(R)
print("INFO: R binary representation:", R_b.shape)
print(R_b)

y = np.dot(R, x)
#y = np.array(y, dtype='uint8')
print("INFO: y=R*x decimal representation:", y.shape)
print(y)

# convert to bit representation
y_b = np.unpackbits(y)  # (32)
print("INFO: y=R*x binary representation:", y_b.shape)
print(y_b)

z_b = binary_matmul(R_b, x_b)
print("INFO: z_b=R_b*x_b binary representation:", z_b.shape)
print(z_b)

z = np.packbits(z_b)
print("INFO: z=Rx decimal representation:", z.shape)
print(z)
