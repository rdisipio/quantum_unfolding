#!/usr/bin/env python3

from decimal2binary import *
np.set_printoptions(precision=1, linewidth=200, suppress=True)

A = [[2, 1, 0], [1, 2, 1], [0, 1, 3]]
A = 2 * np.eye(3)

x = [12, 5, 9]

n = 8
A = np.array(A)
x = np.array(x)
b = A.dot(x)
q = discretize_vector(x, n)

print(A)
print(x)
print("x:", x)
print("x:", np.unpackbits(np.uint8(x)))
print("q:", q)

print("b:", b)
print("b:", np.unpackbits(np.uint8(b)))

A2 = discretize_matrix(A, n)

print(A2)
print(q)
b2 = A2.dot(q)
print("b2:", b2)
