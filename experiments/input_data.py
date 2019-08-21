import numpy as np

input_data = {
    # Peak-like distribution, e.g. invariant mass
    'peak': {
        'truth': np.array([5, 8, 12, 6, 2]),
        'pdata': np.array([6, 9, 13, 5, 3]),
        'IB4': {
            'mean': np.array([6.67, 9.95, 12.29, 5.36, 2.13]),
            'rms': np.array([3.07, 2.61, 2.4, 1.75, 1.35]),
        },
    },

    # Steeply-falling distribution (e.g. pT)
    # $ ./unfolding_baseline.py -o falling
    # NB: does works only with Python 2.7, blame ROOT
    'falling': {
        'truth': np.array([1124, 266, 88, 30, 7]),
        'pdata': np.array([1110, 270, 82, 32, 5]),
        'IB4': {
            'mean': np.array([1124.82, 262.42, 84.84, 28.58, 6.31]),
            'rms': np.array([40.77, 8.15, 8.6, 5.73, 2.48]),
        }
    }
}

# nominal response matrix:
R0 = np.array([
    [1, 1, 0, 0, 0],
    [1, 2, 1, 0, 0],
    [0, 1, 3, 1, 0],
    [0, 0, 1, 3, 1],
    [0, 0, 0, 1, 2],
])

#R0 = np.diag([2., 2., 2.])

# strength of systematics in pseudo-data
sigma_syst = np.array([1.0, -1.0])
