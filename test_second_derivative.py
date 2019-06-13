#!/usr/bin/env python3

from decimal2binary import *

x = [5, 8, 12, 18, 15, 9, 3]

x = np.uint8(x)
x_b = np.unpackbits(x)
print("x:", x, x_b)

n = x.shape[0]
n_bit = 8

lap = laplacian(n)
lap_b = d2b(lap)

print(lap)
print(lap_b)

y = lap.dot(x)
y = np.uint8(y)
print(y)
#y_b = np.packbits(y)
#print("y:", y, y_b)
