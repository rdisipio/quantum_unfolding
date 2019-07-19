import numpy as np

# truth-level:
x = [5, 8, 12, 6, 2]  # signal
z = [6, 9, 13, 5, 3]  # pseudo-data

#x = [1, 3, 2, 2, 1]
#z = [1, 2, 3, 2, 1]

#x = [2, 4, 3]
#z = [2, 3, 2]

# nominal response matrix:
R0 = [[1, 1, 0, 0, 0],
      [1, 2, 1, 0, 0],
      [0, 1, 3, 1, 0],
      [0, 0, 1, 3, 1],
      [0, 0, 0, 1, 2],
      ]

#R0 = np.diag([2., 2., 2.])

x = np.array(x)  # , dtype='uint8')
R0 = np.array(R0)  # , dtype='uint8')
z = np.array(z)  # , dtype='uint8')
y = np.dot(R0, x)
d = np.dot(R0, z)

# the following matrices encode
# the effects of different systematics, i.e.
# y1 = R1*x
# y2 = R2*x

Nbins = x.shape[0]
Nsyst = 2
Nparams = Nbins + Nsyst

# R1 = [[2, 1, 0, 0, 0],
#      [1, 2, 1, 0, 0],
#      [0, 1, 2, 1, 0],
#      [0, 0, 1, 3, 1],
#      [0, 0, 0, 1, 3]
#      ]
#y1 = np.dot(R1, x)

# R2 = [[1, 0, 0, 0, 0],
#      [0, 1, 1, 1, 0],
#      [0, 1, 3, 1, 0],
#      [0, 1, 1, 2, 1],
#      [0, 0, 0, 1, 2]
#      ]
#y2 = np.dot(R2, x)
#dy1 = y1 - y
#dy2 = y2 - y

# syst1 = overall shift
# syst2 = shape change
dy1 = [1, 1, 1, 1, 1]
dy2 = [1, 2, 3, 2, 1]
#dy1 = [1, 1, 1]
#dy2 = [1, 2, 1]

S = np.block([
      [np.zeros([Nbins, Nbins]), np.zeros([Nbins,Nsyst])], 
      [np.zeros([Nsyst, Nbins]), np.eye(Nsyst)]
])

# rectangular matrix
T = np.vstack((dy1, dy2)).T
R = np.block([[R0, T]])
