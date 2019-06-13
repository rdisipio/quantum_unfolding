import numpy as np

# truth-level:
x = [5, 8, 12, 6, 2]  # signal
z = [6, 9, 13, 5, 3]  # pseudo-data

# nominal response matrix:
R0 = [[1, 1, 0, 0, 0],
      [1, 2, 1, 0, 0],
      [0, 1, 3, 1, 0],
      [0, 0, 1, 3, 1],
      [0, 0, 0, 1, 2]
      ]

x = np.array(x, dtype='uint8')
R0 = np.array(R0, dtype='uint8')
z = np.array(z, dtype='uint8')
y = np.dot(R0, x)
d = np.dot(R0, z)

# the following matrices encode
# the effects of different systematics, i.e.
# y1 = R1*x
# y2 = R2*x

Nbins = x.shape[0]
Nsyst = 2
Nparams = Nbins + Nsyst

R1 = [[2, 1, 0, 0, 0],
      [1, 2, 1, 0, 0],
      [0, 1, 2, 1, 0],
      [0, 0, 1, 3, 1],
      [0, 0, 0, 1, 3]
      ]
y1 = np.dot(R1, x)

R2 = [[1, 0, 0, 0, 0],
      [0, 1, 1, 1, 0],
      [0, 1, 3, 1, 0],
      [0, 1, 1, 2, 1],
      [0, 0, 0, 1, 2]
      ]
y2 = np.dot(R2, x)

dy1 = y1 - y
dy2 = y2 - y

S = np.vstack((dy1, dy2)).T
I = np.eye(Nsyst)
O = np.zeros([Nsyst, Nbins])

# R = np.block([[R0, S],
#              [O, I]])
R = np.block([[R0, S]])

s = [1, 2]

# the numbers below have been obtained
# by running the scripts on the QPU

# $ ./unfolding_qubo.py -b qpu -n 5000 -l 0
qpu_reads_unreg = np.array([
    [5, 10, 13,  5,  3],
    [1, 13, 12,  6,  1],
    [10,  5, 14,  5,  3],
    [10,  7, 15,  4,  3],
    [0, 14, 12,  5,  3],
    [5,  9, 13, 6,  3],
    [6, 10, 13,  5,  4],
    [11,  5, 14,  6,  2],
    [0, 14, 11,  6,  3],
    [6,  9, 13,  5,  3],
])

# $ ./unfolding_qubo.py -b qpu -n 5000 -l 1
qpu_reads_reg = np.array([
    [6, 10, 12,  7,  0],
    [4, 12, 11,  7,  1],
    [7, 12, 12,  6,  3],
    [5, 10, 12,  6,  2],
    [5, 10, 12,  5,  2],
    [5, 12, 11,  7,  1],
    [4, 10, 12,  6,  2],
    [7, 13, 12,  7,  1],
    [7, 12, 13,  7,  2],
    [7, 10, 12,  7,  2],
])

unf_data = [
    [6,  9, 13,  5,  3],  # true
    [6.67, 9.95, 12.29, 5.36, 2.13],  # IB4
    [6,  9, 13,  5,  3],  # neal
    [0, 0, 0, 0, 0],  # palceholder1
    [0, 0, 0, 0, 0]  # palceholder2
]

unf_data_unc = [
    [0., 0., 0., 0., 0],
    [3.07, 2.61, 2.40, 1.75, 1.35],  # IB4
    [0., 0., 0., 0., 0],  # neal
    [0, 0, 0, 0, 0],  # palceholder1
    [0, 0, 0, 0, 0],  # palceholder2
]

unf_data[3] = np.mean(qpu_reads_unreg, axis=0)
unf_data_unc[3] = np.std(qpu_reads_unreg, axis=0)
unf_data[4] = np.mean(qpu_reads_reg, axis=0)
unf_data_unc[4] = np.std(qpu_reads_reg, axis=0)

unf_data_labels = [
    'True value',
    'D\'Agostini ItrBayes ($N_{itr}$=4)',
    'QUBO (CPU, neal)',
    'QUBO (QPU, $\lambda$=0)',
    'QUBO (QPU, $\lambda$=1)'
]

# with systeamtics

unf_data_syst = [
    [6,  9, 13,  5,  3, 1, 2],  # true
    [0.5,   1.81,  9.26, 14.94,  7.47,  0.,    0.],  # IB4
    [6,  9, 13,  5,  3,  1,  2],  # neal
    [0, 0, 0, 0, 0, 0, 0],  # QPU unreg
    [0, 0, 0, 0, 0, 0, 0],  # QPU reg
]
unf_data_syst_unc = [
    [0., 0., 0., 0., 0., 0., 0.],
    [0.36, 0.7,  1.72, 3.92, 2.95, 0.,   0.],
    [0., 0., 0., 0., 0., 0., 0.],
    [0, 0, 0, 0, 0, 0, 0],  # QPU unreg
    [0, 0, 0, 0, 0, 0, 0],  # QPU reg
]

# l=0
qpu_reads_syst_unreg = np.array([
    [2, 13, 11,  5,  4,  1,  2],
    [7,  4, 15,  5,  3,  1,  1],
    [3,  7, 13,  7,  2,  1,  1],
    [15,  7, 14,  3,  3,  1,  3],
    [11,  6, 14,  3,  5,  1,  2],
    [6,  6, 10,  5,  5,  0,  1],
    [3,  6, 13,  7,  2,  1,  1],
    [4,  3, 10,  8,  2,  0,  0],
    [0, 11,  7,  8,  2,  0,  1],
    [11,  3, 13,  3,  7,  0,  1],
])

# l=1.0
qpu_reads_syst_reg = np.array([
    [2, 4, 8, 9, 3, 0, 0],
    [2, 4, 8, 7, 4, 0, 0],
    [4, 8, 8, 5, 3, 0, 1],
    [3, 7, 9, 6, 3, 0, 0],
    [5, 9, 7, 7, 7, 0, 1],
    [1, 7, 9, 7, 3, 0, 0],
    [2, 7, 8, 8, 2, 0, 0],
    [5, 9, 9, 6, 1, 0, 1],
    [1, 6, 9, 7, 5, 0, 0],
    [4, 7, 9, 9, 5, 0, 1],
])

# l=0.5
qpu_reads_syst_reg = np.array([
    [5, 11,  9,  6,  4,  0,  2],
    [6, 12, 12,  6,  1,  1,  3],
    [5, 10,  8,  6,  3,  0,  1],
])

unf_data_syst[3] = np.mean(qpu_reads_syst_unreg, axis=0)
unf_data_syst_unc[3] = np.std(qpu_reads_syst_unreg, axis=0)
unf_data_syst[4] = np.mean(qpu_reads_syst_reg, axis=0)
unf_data_syst_unc[4] = np.std(qpu_reads_syst_reg, axis=0)
