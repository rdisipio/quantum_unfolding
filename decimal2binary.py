import numpy as np
import random as rnd

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

def discretize_vector(x, encoding = [] ):
    N = len(x)
    n = 0

    if len(encoding) == 0:
        n = 4
        encoding = np.array( [n]*N )

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


def compact_vector(q, encoding = []):
    n = 0
    N = 0
    if len(encoding) == 0:
        n = 4
        N = q.shape[0] // n
        encoding = np.array( [n]*N )

    x = np.zeros(N, dtype='uint8')
    for i in range(N):
        for j in range(n):
            p = np.power(2, n-j-1)
            x[i] += p*q[(n*i+j)]
    return x


def discretize_matrix(A, encoding = [] ):
    # x has N elements (decimal)
    # q has Nx elements (binary)
    # A has N columns
    # D has Nn columns
    # Ax = Dq

    N = A.shape[0]
    M = A.shape[1]

    n = 0
    if len(encoding) == 0:
        n = 4

    D = np.zeros([N, M*n])

    for i in range(M):
        for j in range(n): #-->bits
            k = (i)*n+j
            D[:, k] = np.power(2, n-j-1) * A[:, i]
    return D


#####################################


class BinaryEncoder(object):

    def __init__(self):
        self.alpha = None
        self.beta  = None
        self.rho   = None
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def set_alpha( self, alpha : np.array ):
        self.alpha = np.copy( alpha )
    
    def set_beta( self, beta : list ):
        self.beta  = []
        N = len(beta)
        for i in range(N):
            self.beta += [ np.copy( beta[i]) ]
    
    def set_rho( self, rho : np.array ):
        self.rho = np.copy( rho )
    
    def set_params(self, alpha : np.array, beta : list, rho : np.array ):
        self.set_alpha( alpha )
        self.set_beta( beta )
        self.set_rho( rho )

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def encode( self, x ):
        '''
        :param alpha: offeset
        :param beta: scaling
        :param rho: n-bits encoding
        :param x: vector in base-10
        :return: Returns binary-encoded vector
        '''
        from decimal import Decimal

        N = len(x)
        
        n_bits_total = sum( self.rho )
        x_b = np.zeros( n_bits_total, dtype='uint' )

        for i in range(N-1, -1, -1):
            n = self.rho[i]
            x_d = x[i] - self.alpha[i]

            for j in range(0,n,1):
                a = np.sum(self.rho[:i]) + j

                more_than = Decimal(x_d) // Decimal(self.beta[i][a] )
                equal_to = int(np.isclose(x_d, self.beta[i][a] ) )

                x_b[a] = min([1, more_than or equal_to ])
                
                x_d = x_d - x_b[a]*self.beta[i][a]

        return x_b
        
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def random_encode(self, x, floats=False ):
        '''
        encode x chosing alpha and beta parameters at random from
        suitable set
        '''
        N=len(self.rho)
        self.alpha=np.zeros(N)
        self.beta = [ None ] * N

        for i in range(0, N, 1):
            n = self.rho[i]
            self.beta[i] = np.zeros(n)
            if floats:
                self.alpha[i] = rnd.uniform(0,x[i])
            else:
                self.alpha[i] = rnd.randrange(x[i])
            if x[i] == 0:
                self.alpha[i] = -1
            scale_factor = (x[i] - self.alpha[i])/rnd.randrange(1,np.power(2,n)) 

            for j in range(n):
                self.beta[i][j] = scale_factor * np.power(2,n-j-1) 
        
        x_b = self.encode(x)
    
        return x_b

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def auto_encode( self, x, auto_range = 0.5 ):
        ''' 
        if range is [ x*(1-h), x*(1+h)], e.g. x +- 50%
        alpha = x*(1-h) = lowest possible value
        width = x*(1+h) - x*(1-h) = x + hx -x + hx = 2hx
        then divide the range in n steps
        '''

        N = len(self.rho)
        n_bits_total = sum(self.rho)
        self.alpha = np.zeros(N)
        self.beta  = np.zeros([N,n_bits_total])

        for i in range(N-1, -1, -1):
            n = self.rho[i]
            self.alpha[i] = ( 1. - auto_range ) * x[i]
            w = 2 * auto_range*x[i] / float(n)
            
            for j in range(n):
                a = np.sum(self.rho[:i])+j
                self.beta[i][a] = w * np.power(2, n-j-1)

        x_b = self.encode(x)

        return x_b

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def decode( self, x_b ):
        '''
        :param alpha: offeset (vector)
        :param beta: scaling (array)
        :param rho: n-bits encoding (vector)
        :param x: binary vector
        :return: Returns decoded vector
        '''

        N = len(self.alpha)
        x = np.zeros( N )
        for i in range(N-1, -1, -1):
            x[i] = self.alpha[i]
            n = self.rho[i]
            for j in range(0, n, 1):
                a = np.sum(self.rho[:i])+j
                x[i] += self.beta[i][j] * x_b[a]

        return x
        
#~~~~~~~~~~~~~~~~~~~~~~~~~~~
