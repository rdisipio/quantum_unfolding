import numpy as np


def laplacian(n):

    lap = np.diag(2*np.ones(n)) + \
        np.diag(-1*np.ones(n-1), 1) + \
        np.diag(-1*np.ones(n-1), -1)

    return lap


def laplacian_nbits(N, n=8):
    A = np.zeros([N, N*n])
    bitvals = [np.power(2., -i) for i in range(n)]
    for i in range(n):
        A[0, i] = -2 * bitvals[i]
        A[0, n + i] = 1 * bitvals[i]
        A[-1, - 2 * n + i] = 1 * bitvals[i]
        A[-1, - n + i] = -2 * bitvals[i]
    for i in range(1, N-1):
        for j in range(n):
            A[i, n * i - 2 * n + j] = 1 * bitvals[j]
            A[i, n * i - n + j] = -2 * bitvals[j]
            A[i, n * i + j] = 1 * bitvals[j]
    return A


def get_int_max(n_bits):
    return 2**n_bits - 1


def d2b(a, n_bits=8):
    '''Convert a list or a list of lists to binary representation
    '''
    A = np.array(a, dtype='uint8')
    if A.ndim == 1:
        return np.unpackbits(A)
    else:
        n_cols = A.shape[0]

        n_vectors = n_cols * n_bits

        R_b = np.zeros([n_vectors, n_vectors], dtype='uint8')

        # the complete space is spanned by (n_cols X n_bits) standard basis vectors v, i.e.:
        # ( 0, 0, ..., 1 )
        # ( 0, 1, ..., 0 )
        # ( 1, 0, ..., 0 )

        # Multiplying Rv "extracts" the column corresponding the non-zero element
        # By iteration, we can convert R from decimal to binary

        for i in range(n_vectors):
            v_bin = np.zeros(n_vectors, dtype='uint8')
            v_bin[i] = 1
            # print(v_bin)

            v_dec = np.packbits(v_bin)
            # print(x_dec)

            u_dec = np.dot(A, v_dec)
            u_bin = np.unpackbits(u_dec)

            R_b[:, i] = u_bin

        return R_b


def half_adder(x, y):
    s = (x ^ y)  # sum   = XOR
    c = (x & y)  # carry = AND
    return s, c


def full_adder(x, y, c0=0):
    s1, c1 = half_adder(x, y)
    s2, c2 = half_adder(s1, c0)
    c = c1 | c2
    return s2, c


def binary_matmul(A, x):

    n = x.shape[0]
    y = np.zeros(n, dtype='uint8')

    for i in range(n - 1, -1, -1):
        c = 0
        for j in range(n - 1, -1, -1):
            p = A[i][j] & x[j]
            c = y[i] & p  # carry bit if y=01+01=10=2
            y[i] ^= p
            y[i - 1] ^= c
    return y


def discretize_vector(x, n=8):
    N = len(x)

    q = np.zeros(N*n)
    for i in range(N-1, -1, -1):
        x_d = int(x[i])
        j = n-1
        while x_d > 0:
            k = i*n + j
            q[k] = x_d % 2
            x_d = x_d // 2
            j -= 1
    return np.uint8(q)


def discretize_matrix(A, n=8):
    # x has N elements (decimal)
    # q has Nx elements (binary)
    # A has N columns
    # D has Nn columns
    # Ax = Dq

    N = A.shape[0]

    D = np.zeros([N, N*n])

    for i in range(N):
        for j in range(n):
            k = (i)*n+j
            D[:, k] = np.power(2, n-j-1) * A[:, i]
    return D
