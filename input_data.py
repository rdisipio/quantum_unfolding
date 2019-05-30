import numpy as np

# truth-level:
x = [5, 8, 12, 6, 2]  # signal
z = [6, 9, 13, 5, 3]  # pseudo-data

# nominal response matrix:
R = [[1, 1, 0, 0, 0],
     [1, 2, 1, 0, 0],
     [0, 1, 3, 1, 0],
     [0, 0, 1, 3, 1],
     [0, 0, 0, 1, 2]
     ]

x = np.array(x, dtype='uint8')
R = np.array(R, dtype='uint8')
z = np.array(z, dtype='uint8')
y = np.dot(R, x)
d = np.dot(R, z)
