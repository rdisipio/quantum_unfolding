#!/usr/bin/env python3

import numpy as np
n_bits = 8

a = [9, 5]
b = [4, 3]

a = np.uint8(a)
b = np.uint8(b)

a_b = np.unpackbits(a)
b_b = np.unpackbits(b)
print("a:", a, a_b)
print("b:", b, b_b)

c = a - b
c_b = np.unpackbits(c)
print("c:", c, c_b)

n = a.shape[0]
s = np.zeros(n, dtype='uint8')
s_b = np.zeros(8 * n, dtype='uint8')

for i in np.arange(n - 1, -1, -1):
    borr = 0
    # each number is represented by 8 bits (1byte)
    for k in np.arange(7, -1, -1):
        j = n_bits * i + k
        h = np.power(2, 8 - k - 1)

        diff = a_b[j] ^ b_b[j]
        borr = (~a_b[j]) & b_b[j]

        s_b[j] ^= diff
        s_b[j - 1] ^= borr
        #print(i, k, a_b[j], b_b[j], diff, borr)

        s[i] += h * s_b[j]

#s = np.packbits(s_b)
print("result (bin):", s_b, s)
print("result (dec):", c_b, c)
