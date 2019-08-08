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
      # $ ./unfolding_baseline.py -o falling 
      # NB: does works only with Python 2.7, blame ROOT
      'falling' : {
            'truth' : np.array( [14, 7, 5, 4, 2] ),
            'pdata' : np.array( [15, 6, 4, 2, 1] ),
            'IB4' : {
                  'mean' : np.array( [15.4,  9.3,  4.3, 2.9,  1.5] ),
                  'rms' : np.array( [5.1, 1.8 , 1.5, 1.3, 1.2]),
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