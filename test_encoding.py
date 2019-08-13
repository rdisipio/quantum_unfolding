#!/usr/bin/env python3

import numpy as np

from decimal2binary import BinaryEncoder

encoder = BinaryEncoder()

rho = np.array( [4, 4, 4] )
#rho = np.array( [8, 8, 8] )
#rho = np.array( [5,5,5] )

x = np.array( [5, 11, 3 ])
print("INFO: x =", x)

print("INFO: encoding:", rho)
encoder.set_rho( rho )

x_b = encoder.auto_encode( x, auto_range = 0.5 )

print("INFO: alpha:")
print(encoder.alpha)
print("INFO: beta")
print(encoder.beta)

print("INFO: re-encoding...")
y = encoder.decode(x_b)
print("INFO: original:", x)
print("INFO: encoded:", x_b)
print("INFO: decoded:", y)