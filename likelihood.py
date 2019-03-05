from math import exp, sqrt
import numpy as np


def log_gauss(d_b, y_b):
    d_d = np.packbits(d_b)
    y_d = np.packbits(y_b)
    n = d_d.shape[0]
    z = 0.
    for k in range(n):
        s = (int(d_d[k]) - int(y_d[k]))
        z += s*s
    return z


def test_lhood(d_b, y_b):
    z = np.uint8(0)
    n = d_b.shape[0]
    r = np.zeros(n, dtype='uint8')

    for k in range(n-1, -1, -1):
        r[k] = ~(d_b[k] ^ y_b[k])
    # for k in range(n-1, -1, -1):
        z ^= (r[k] << k)
        #z += abs(d_b[k] - y_b[k])
    return z
