import numpy as np
import random as rnd


def laplacian(n):

    lap = np.diag(2*np.ones(n)) + \
        np.diag(-1*np.ones(n-1), 1) + \
        np.diag(-1*np.ones(n-1), -1)

    return lap


def laplacian_nbits(N, n=8):
    A = np.zeros([N, N * n])
    bitvals = [np.power(2., -i) for i in range(n)]
    for i in range(n):
        A[0, i] = -2 * bitvals[i]
        A[0, n + i] = 1 * bitvals[i]
        A[-1, -2 * n + i] = 1 * bitvals[i]
        A[-1, -n + i] = -2 * bitvals[i]
    for i in range(1, N - 1):
        for j in range(n):
            A[i, n * i - 2 * n + j] = 1 * bitvals[j]
            A[i, n * i - n + j] = -2 * bitvals[j]
            A[i, n * i + j] = 1 * bitvals[j]
    return A


def discretize_vector(x,n=8):
    N = len(x)
    q = np.zeros(N * n)
    for i in range(N - 1, -1, -1):
        x_d = int(x[i])
        j = n - 1
        while x_d > 0:
            k = i * n + j
            q[k] = x_d % 2
            x_d = x_d // 2
            j -= 1
    return np.uint8(q)


def compact_vector(q, n=8):
    N = q.shape[0] // n
    x = np.zeros(N, dtype='uint8')
    for i in range(N):
        for j in range(n):
            p = np.power(2, n - j - 1)
            x[i] += p * q[(n * i + j)]
    return x


def discretize_matrix(A, n=8):
    # x has N elements (decimal)
    # q has Nx elements (binary)
    # A has N columns
    # D has Nn columns
    # Ax = Dq

    N = A.shape[0]
    M = A.shape[1]
    D = np.zeros([N, M * n])

    for i in range(M):
        for j in range(n):  #-->bits
            k = (i) * n + j
            D[:, k] = np.power(2, n - j - 1) * A[:, i]
    return D


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


def d2b(a, n_bits=8):
    '''Convert a list or a list of lists to binary representation
    '''
    A = np.array(a, dtype='uint8')

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

        #v_dec = np.packbits(v_bin)
        v_dec = compact_vector(v_bin, n_bits)
        # print(x_dec)

        u_dec = np.dot(A, v_dec)
        #u_bin = np.unpackbits(u_dec)
        u_bin = discretize_vector(u_dec, n_bits)

        R_b[:, i] = u_bin

    return R_b


#####################################


class BinaryEncoder(object):
    def __init__(self):
        self.alpha = None
        self.beta = None
        self.rho = None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def set_alpha(self, alpha: np.array):
        self.alpha = np.copy(alpha)

    def set_beta(self, beta: list):
        self.beta = []
        N = len(beta)
        for i in range(N):
            self.beta += [np.copy(beta[i])]

    def set_rho(self, rho: np.array):
        self.rho = np.copy(rho)

    def set_params(self, alpha: np.array, beta: list, rho: np.array):
        self.set_alpha(alpha)
        self.set_beta(beta)
        self.set_rho(rho)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def encode(self, x):
        '''
        :param alpha: offeset
        :param beta: scaling
        :param rho: n-bits encoding
        :param x: vector in base-10
        :return: Returns binary-encoded vector
        '''
        from decimal import Decimal

        N = len(x)

        n_bits_total = int(sum(self.rho))
        x_b = np.zeros(n_bits_total, dtype='uint')

        for i in range(N - 1, -1, -1):
            n = int(self.rho[i])
            x_d = x[i] - self.alpha[i]

            for j in range(0, n, 1):
                a = int(np.sum(self.rho[:i]) + j)

                more_than = Decimal(x_d) // Decimal(self.beta[i][a])
                equal_to = np.isclose(x_d, self.beta[i][a])

                x_b[a] = min([1, more_than or equal_to])

                x_d = x_d - x_b[a] * self.beta[i][a]

        return x_b

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def random_encode(self, x, floats=False):
        '''
        encode x chosing alpha and beta parameters at random from
        suitable set
        '''
        N = len(self.rho)
        self.alpha = np.zeros(N)
        self.beta = [None] * N

        for i in range(0, N, 1):
            n = self.rho[i]
            self.beta[i] = np.zeros(n)
            if floats:
                self.alpha[i] = rnd.uniform(0, x[i])
            else:
                self.alpha[i] = rnd.randrange(x[i])
            if x[i] == 0:
                self.alpha[i] = -1
            scale_factor = (x[i] - self.alpha[i]) / rnd.randrange(
                1, np.power(2, n))

            for j in range(n):
                self.beta[i][j] = scale_factor * np.power(2, n - j - 1)

        x_b = self.encode(x)

        return x_b

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def auto_encode(self, x, auto_range=0.5):
        ''' 
        if range is [ x*(1-h), x*(1+h)], e.g. x +- 50%
        alpha = x*(1-h) = lowest possible value
        width = x*(1+h) - x*(1-h) = x + hx -x + hx = 2hx
        then divide the range in n steps
        '''

        N = len(self.rho)
        n_bits_total = sum(self.rho)
        self.alpha = np.zeros(N)
        self.beta = np.zeros([N, n_bits_total])

        for i in range(N - 1, -1, -1):
            n = self.rho[i]
            self.alpha[i] = (1. - auto_range) * x[i]
            #w = 2 * auto_range*x[i] / float(n)
            w = 2 * auto_range * x[i] / np.power(2, n)

            for j in range(n):
                a = np.sum(self.rho[:i]) + j
                self.beta[i][a] = w * np.power(2, n - j - 1)

        x_b = self.encode(x)

        return x_b

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def decode(self, x_b):
        '''
        :param alpha: offeset (vector)
        :param beta: scaling (array)
        :param rho: n-bits encoding (vector)
        :param x: binary vector
        :return: Returns decoded vector
        '''

        N = len(self.alpha)
        x = np.zeros(N)
        for i in range(N - 1, -1, -1):
            x[i] = self.alpha[i]
            n = int(self.rho[i])
            for j in range(0, n, 1):
                a = int(np.sum(self.rho[:i]) + j)
                x[i] += self.beta[i][a] * x_b[a]

        return x


#~~~~~~~~~~~~~~~~~~~~~~~~~~~
