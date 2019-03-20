import numpy as np
from decimal2binary import *


def log_gauss(x, *params):

    d_b = params[0][0]
    R_b = params[0][1]
    n_bits = d_b.shape[0]
    n_bins = n_bits // 8

    x = np.uint8([int(n) for n in x])
    x_b = np.unpackbits(x)

    # apply response to obtain reco-level prediction
    y_b = binary_matmul(R_b, x_b)

    z = 0.
    for k in range(n):
        # s = (int(d_d[k]) - int(y_d[k]))
        # z += s*s
        s = d_d[k] - y_d[k]
        z += abs(s)

    # regularization parameter (Tikhonov 2nd derivative)
    rho = 0.
    for k in range(n - 1, 1, n - 2):
        rho += abs(y_d[k - 1] - y_d[k + 1])

    # strength of regularization
    lmbd = 0
    z += lmbd * rho

    return z


def test_lhood(d_b, y_b):
    z = np.uint8(0)
    n = d_b.shape[0]
    r = np.zeros(n, dtype='uint8')
    # n_bits = 8 * n
    for k in range(n - 1, -1, -1):
        r[k] = (d_b[k] ^ y_b[k])
        # z ^= ((r[k] & 1) << k)
        # z ^= (r[k] << k)

    # print(z, r)
    return z, r
