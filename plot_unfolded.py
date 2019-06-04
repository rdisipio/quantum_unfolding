#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns

# sns.set(color_codes=True)

labels = [
    'True value',
    'D\'Agostini ItrBayes (nitr=4)',
    'QUBO (neal)',
    'QPU (custom schedule, reads=5k)',
    'QPU (default schedule, reads=10k)'
]

data = [
    [6,  9, 13,  5,  3],  # true
    [6,  9, 12,  5,  2],
    [6,  9, 13,  5,  3],
    [9,  7, 14,  5,  2],
    [1, 13, 12,  5,  3],
]
unc = [
    [0., 0., 0., 0., 0],
    [0., 0., 0., 0., 0],
    [0., 0., 0., 0., 0],
    [0.5, 0.5, 0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5, 0.5, 0.5],
]

n_methods = len(data)
ibin = np.arange(n_methods)
colors = ['black', 'red', 'blue', 'seagreen', 'gold']
#          'gold', 'cyan', 'violet', 'navyblue']
#colors = ['black', 'salmon', 'royalblue', 'lightgreen', 'gold']
markers = ['o', 'v', '^', 'x', 'D']
bar_width = 0.1

fig, ax = plt.subplots(tight_layout=True, figsize=(10, 4))

for i in range(5):
    plt.errorbar(x=ibin+0.05*i, y=data[i],
                 yerr=unc[i],
                 color=colors[i],
                 fmt=markers[i],
                 label=labels[i])
plt.legend()
plt.ylabel("Unfolded")
plt.xlabel("Bin")
plt.show()
