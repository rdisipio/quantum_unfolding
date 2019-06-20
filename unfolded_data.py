import numpy as np

# the numbers below have been obtained
# by running the scripts on the QPU

# $ ./unfolding_qubo.py -b qpu -n 5000 -l 0

# low-noise new device

# default annealing schedule
qpu_reads_unreg = np.array([
    [1, 12, 12, 5, 2],  # E = -5473.0 chi2/dof = 1.39
    [4, 12, 11, 7, 0],  # E = -5462.0 chi2/dof = 1.44
    [1, 14, 12, 6, 2],  # E = -5458.0 chi2/dof = 1.89
    [3, 11, 13, 5, 3],  # E = -5479.0 chi2/dof = 0.49
    [2, 11, 12, 5, 4],  # E = -5475.0 chi2/dof = 0.88
    [1, 12, 12, 6, 2],  # E = -5478.0 chi2/dof = 1.44
    [8, 7, 13, 8, 0],  # E = -5435.0 chi2/dof = 1.48
    [5, 9, 14, 4, 4],  # E = -5478.0 chi2/dof = 0.19
    [3, 11, 12, 6, 2],  # E = -5482.0 chi2/dof = 0.64
    [4, 10, 12, 6, 1],  # E = -5473.0 chi2/dof = 0.60
    [0, 14, 11, 6, 2],  # E = -5479.0 chi2/dof = 2.40
    [6, 8, 14, 4, 5],  # E = -5473.0 chi2/dof = 0.43
    [5, 11, 13, 5, 3],  # E = -5471.0 chi2/dof = 0.15
    [3, 12, 10, 7, 2],  # E = -5465.0 chi2/dof = 1.08
    [8, 8, 14, 4, 4],  # E = -5480.0 chi2/dof = 0.35
    [2, 12, 12, 5, 3],  # E = -5482.0 chi2/dof = 0.94
    [9, 6, 14, 5, 3],  # E = -5480.0 chi2/dof = 0.64
    [3, 12, 12, 6, 2],  # E = -5478.0 chi2/dof = 0.78
    [7, 8, 14, 4, 3],  # E = -5479.0 chi2/dof = 0.14
    [7, 8, 14, 5, 3],  # E = -5480.0 chi2/dof = 0.09
])


# default annealing schedule
qpu_reads_reg = np.array([
    [3, 12, 12, 6, 1],  # E = -5301.0 chi2/dof = 1.03
    [5, 11, 12, 5, 3],  # E = -5362.0 chi2/dof = 0.17
    [7, 12, 11, 7, 1],  # E = -5357.0 chi2/dof = 0.90
    [6, 13, 11, 6, 2],  # E = -5335.0 chi2/dof = 0.65
    [8, 10, 12, 6, 1],  # E = -5340.0 chi2/dof = 0.60
    [6, 10, 11, 6, 1],  # E = -5393.0 chi2/dof = 0.49
    [6, 12, 11, 7, 1],  # E = -5364.0 chi2/dof = 0.86
    [5, 10, 12, 6, 1],  # E = -5385.0 chi2/dof = 0.47
    [6, 10, 12, 7, 1],  # E = -5387.0 chi2/dof = 0.58
    [6, 12, 12, 6, 3],  # E = -5364.0 chi2/dof = 0.32
    [3, 13, 13, 7, 0],  # E = -5163.0 chi2/dof = 1.77
    [5, 10, 12, 7, 1],  # E = -5388.0 chi2/dof = 0.62
    [6, 11, 12, 5, 1],  # E = -5347.0 chi2/dof = 0.46
    [6, 11, 11, 6, 2],  # E = -5411.0 chi2/dof = 0.32
    [7, 13, 11, 7, 1],  # E = -5305.0 chi2/dof = 1.10
    [6, 12, 10, 6, 3],  # E = -5372.0 chi2/dof = 0.47
    [3, 10, 13, 6, 2],  # E = -5326.0 chi2/dof = 0.54
    [7, 10, 10, 7, 0],  # E = -5330.0 chi2/dof = 1.19
    [5, 11, 12, 7, 2],  # E = -5392.0 chi2/dof = 0.46
    [4, 10, 12, 8, 2],  # E = -5356.0 chi2/dof = 0.75
])

unf_data = [
    [6,  9, 13,  5,  3],  # true
    #[6.67, 9.95, 12.29, 5.36, 2.13],  # IB4
    [6.7, 10.,  12.3,  5.4,  2.1],  # IB4
    [6,  9, 13,  5,  3],  # neal
    [0, 0, 0, 0, 0],  # palceholder1
    [0, 0, 0, 0, 0]  # palceholder2
]

unf_data_unc = [
    [0., 0., 0., 0., 0],
    #[3.07, 2.61, 2.40, 1.75, 1.35],  # IB4
    [3.1, 2.6, 2.4, 1.7, 1.4],  # IB4
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

#################
# with systeamtics

unf_data_syst = [
    [6,  9, 13,  5,  3, 1, 2],  # true
    #[0.5,   1.81,  9.26, 14.94,  7.47,  0.,    0.],  # IB4
    [8.09, 11.41, 13.45,  6.57,  3.05,  0.,    0.],  # IB4
    [6,  9, 13,  5,  3,  1,  2],  # neal
    [0, 0, 0, 0, 0, 0, 0],  # QPU unreg
    [0, 0, 0, 0, 0, 0, 0],  # QPU reg
]

unf_data_syst_unc = [
    [0., 0., 0., 0., 0., 0., 0.],
    #[0.36, 0.7,  1.72, 3.92, 2.95, 0.,   0.],
    [3.45, 2.79, 2.49, 1.94, 1.7,  0.,   0.],
    [0., 0., 0., 0., 0., 0., 0.],
    [0, 0, 0, 0, 0, 0, 0],  # QPU unreg
    [0, 0, 0, 0, 0, 0, 0],  # QPU reg
]

# low-noise new device
qpu_reads_syst_unreg = np.array([
    [13, 3, 15, 4, 3, 3, 2],  # E = -7167.0 chi2/dof = 4.17
    [5, 9, 13, 4, 4, 0, 3],  # E = -7176.0 chi2/dof = 0.55
    [7, 9, 14, 5, 2, 0, 2],  # E = -7165.0 chi2/dof = 0.39
    [7, 7, 13, 5, 1, 1, 3],  # E = -7169.0 chi2/dof = 0.61
    [4, 6, 10, 3, 2, 2, 7],  # E = -7170.0 chi2/dof = 4.25
    [7, 10, 14, 5, 4, 3, 0],  # E = -7168.0 chi2/dof = 1.67
    [2, 10, 11, 4, 2, 4, 3],  # E = -7176.0 chi2/dof = 3.28
    [7, 7, 14, 5, 2, 3, 1],  # E = -7175.0 chi2/dof = 1.38
    [3, 6, 11, 3, 2, 4, 5],  # E = -7171.0 chi2/dof = 4.36
    [5, 6, 10, 3, 2, 0, 7],  # E = -7178.0 chi2/dof = 4.12
    [2, 13, 11, 4, 1, 2, 2],  # E = -7114.0 chi2/dof = 1.82
    [7, 9, 15, 5, 4, 2, 0],  # E = -7178.0 chi2/dof = 0.95
    [6, 5, 14, 1, 0, 8, 3],  # E = -7106.0 chi2/dof = 14.39
    [3, 7, 7, 5, 0, 3, 7],  # E = -7156.0 chi2/dof = 6.05
    [1, 7, 9, 1, 2, 2, 9],  # E = -7151.0 chi2/dof = 8.72
    [2, 7, 10, 4, 2, 3, 5],  # E = -7164.0 chi2/dof = 3.21
    [9, 7, 11, 7, 1, 7, 0],  # E = -7098.0 chi2/dof = 10.60
    [3, 8, 11, 6, 0, 5, 2],  # E = -7162.0 chi2/dof = 5.28
    [0, 11, 13, 4, 3, 5, 1],  # E = -7170.0 chi2/dof = 5.79
    [8, 11, 15, 5, 4, 0, 0],  # E = -7164.0 chi2/dof = 1.19
])


qpu_reads_syst_reg = np.array([
    [5, 8, 6, 3, 1, 2, 6],  # E = -6980.0 chi2/dof = 3.80
    [2, 6, 6, 1, 0, 3, 11],  # E = -7087.0 chi2/dof = 14.53
    [3, 6, 6, 5, 0, 3, 9],  # E = -7074.0 chi2/dof = 9.44
    [3, 8, 7, 3, 0, 5, 7],  # E = -7081.0 chi2/dof = 9.17
    [5, 7, 10, 3, 0, 4, 6],  # E = -7014.0 chi2/dof = 5.53
    [2, 6, 8, 3, 1, 6, 5],  # E = -7052.0 chi2/dof = 9.31
    [7, 10, 11, 7, 2, 1, 3],  # E = -7081.0 chi2/dof = 0.55
    [4, 9, 7, 3, 1, 3, 7],  # E = -7066.0 chi2/dof = 5.52
    [2, 7, 7, 2, 0, 6, 6],  # E = -7037.0 chi2/dof = 10.92
    [4, 8, 9, 6, 1, 4, 4],  # E = -7101.0 chi2/dof = 3.64
    [11, 12, 12, 7, 3, 0, 1],  # E = -6959.0 chi2/dof = 1.89
    [2, 7, 7, 5, 1, 3, 7],  # E = -7090.0 chi2/dof = 5.93
    [7, 12, 12, 6, 3, 1, 1],  # E = -7077.0 chi2/dof = 0.49
    [6, 11, 11, 7, 1, 1, 1],  # E = -7074.0 chi2/dof = 0.85
    [5, 8, 8, 4, 1, 5, 5],  # E = -7097.0 chi2/dof = 6.06
    [2, 5, 6, 3, 0, 1, 10],  # E = -7132.0 chi2/dof = 11.00
    [4, 6, 7, 5, 1, 7, 5],  # E = -7056.0 chi2/dof = 11.57
    [7, 11, 12, 6, 3, 0, 1],  # E = -7084.0 chi2/dof = 0.60
    [3, 6, 7, 5, 1, 3, 7],  # E = -7103.0 chi2/dof = 5.78
    [5, 8, 7, 4, 2, 1, 7],  # E = -7116.0 chi2/dof = 4.02
])

unf_data_syst[3] = np.mean(qpu_reads_syst_unreg, axis=0)
unf_data_syst_unc[3] = np.std(qpu_reads_syst_unreg, axis=0)
unf_data_syst[4] = np.mean(qpu_reads_syst_reg, axis=0)
unf_data_syst_unc[4] = np.std(qpu_reads_syst_reg, axis=0)
