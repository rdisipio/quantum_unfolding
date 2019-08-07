import numpy as np


input_data = {
      # Peak-like distribution, e.g. invariant mass
      'peak' : {
            'truth' : np.array( [5, 8, 12, 6, 2] ),
            'pdata' : np.array( [6, 9, 13, 5, 3] ),
            'IB4' : {
                  'mean' : np.array( [6.7, 10.,  12.3,  5.4,  2.1] ),
                  'rms' : np.array( [3.1, 2.6, 2.4, 1.7, 1.4] ),
            },
      },

      # Steeply-falling distribution (e.g. pT)
      'falling' : {
            'truth' : np.array( [14, 9, 5, 4, 2] ),
            'pdata' : np.array( [15, 8, 5, 3, 2] ),
            'IB4' : {
                  'mean' : np.array( [14.1,  9.3,  4.7,  2.9,  1.7] ),
                  'rms' : np.array( [4.8, 1.8, 1.6, 1.3, 1.3]),
            }
      }
}

# nominal response matrix:
R0 = np.array( [
      [1, 1, 0, 0, 0],
      [1, 2, 1, 0, 0],
      [0, 1, 3, 1, 0],
      [0, 0, 1, 3, 1],
      [0, 0, 0, 1, 2],
      ] )

#R0 = np.diag([2., 2., 2.])