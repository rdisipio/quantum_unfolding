from math import exp, sqrt
import numpy as np


def log_gauss(d_b, y_b):
    z = np.uint8(0)
    n = d_b.shape[0]
    r = np.zeros(n, dtype='uint8')

    for k in range(n):
        r[k] = ~(d_b[k] ^ y_b[k])
    for k in range(n):
        z ^= r[k]
        #z += abs(d_b[k] - y_b[k])
    return z
