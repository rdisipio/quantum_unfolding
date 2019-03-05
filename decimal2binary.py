import numpy as np


def d2b(R, n_bits=8):
    n_cols = R.shape[0]

    n_vectors = n_cols*n_bits

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

        u_dec = np.dot(R, v_dec)
        u_bin = np.unpackbits(u_dec)

        R_b[:, i] = u_bin

    return R_b


def half_adder( x, y ):
    s = (x ^ y) # sum   = XOR
    c = (x & y) # carry = AND
    return s, c

def full_adder( x, y, c0=0 ):
    s1, c1 = half_adder(x,y)
    s2, c2 = half_adder(s1,c0)
    c = c1 | c2
    return s2, c

def binary_matmul( A, x ):

    n = x.shape[0]
    y = np.zeros( n, dtype='uint8' )

    for i in range(n-1,-1,-1):
       c = 0
       for j in range(n-1,-1,-1):
          p = A[i][j] & x[j]
          c = y[i] & p # carry bit if y=01+01=10=2
          y[i] ^= p
          y[i-1] ^= c
    return y
