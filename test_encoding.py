#!/usr/bin/env python3

import numpy as np
import random as rnd

from decimal2binary import BinaryEncoder

def random_sign():
    rnd.randrange(-1,1,2)

encoder = BinaryEncoder()
for i in range(0,500):
    #rho = np.array( [4, 4, 4] )
    
    #rho = np.array( [5,5,5] )
    scope = 10000
    x = np.array( [rnd.randrange(-scope,scope) for i in range(rnd.randrange(1,10)) ])
    rho = np.array( [rnd.randrange(1,12) for i in range(len(x))] )
    print("INFO: x =", x)
    
    print("INFO: encoding:", rho)
    encoder.set_rho( rho )
    
    #x_b = encoder.auto_encode( x, auto_range = 0.5 )
    x_b = encoder.random_encode(x,floats=True)
    
    print("INFO: alpha:")
    print(encoder.alpha)
    print("INFO: beta")
    print(encoder.beta)
    
    print("INFO: re-encoding...")
    y = encoder.decode(x_b)
    print("INFO: original:", x)
    print("INFO: encoded:", x_b)
    print("INFO: decoded:", y)
    np.testing.assert_allclose(x,y,err_msg="encode/decode did not pass, x=%s, y=%s" % (x,y))

print("ran %s succesful encoding/decoding tests" % (i+1))
