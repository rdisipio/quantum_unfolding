#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns

# sns.set(color_codes=True)

data = {
    'True value': [6,  9, 13,  5,  3],
    'D\'Agostini ItrBayes (nitr=4)': [6,  9, 12,  5,  2],
    #    'SVD (k=3)': [9, 11, 10,  4,  2],
    'QUBO (neal)': [6,  9, 13,  5,  3],
    #    'QPU (custom schedule, reads=1k)': [3, 10, 13,  5,  3],
    'QPU (custom schedule, reads=5k)': [9,  7, 14,  5,  2],
    'QPU (default schedule, reads=10k)': [1, 13, 12,  5,  3],
}

n_methods = len(data)
ibin = np.arange(n_methods)
colors = ['black', 'red', 'blue', 'seagreen',
          'gold', 'cyan', 'violet', 'navyblue']
bar_width = 0.1

fig, ax = plt.subplots(tight_layout=True, figsize=(10, 4))

for i in range(len(ibin)):
    plt.bar(ibin+(3-i)*bar_width, list(data.values())[i],
            color=colors[i],
            width=bar_width,
            label=list(data.keys())[i],
            align='center')
plt.legend()
plt.ylabel("Unfolded")
plt.xlabel("Bin")
plt.show()
