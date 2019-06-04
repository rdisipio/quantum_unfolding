#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns

# sns.set(color_codes=True)

labels = [
    'True value',
    'D\'Agostini ItrBayes (nitr=4)',
    'QUBO (CPU, neal)',
    'QUBO (QPU, $\lambda$=0)',
    'QUBO (QPU, $\lambda$=1)'
    #'QPU (default schedule, reads=10k)'
]

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

data = [
    [6,  9, 13,  5,  3],  # true
    [6,  9, 12,  5,  2],
    [6,  9, 13,  5,  3],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    #[9,  7, 14,  5,  2],
    #[1, 13, 12,  5,  3],
]
unc = [
    [0., 0., 0., 0., 0],
    [0., 0., 0., 0., 0],
    [0., 0., 0., 0., 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
]
data[3] = np.mean(qpu_reads_unreg, axis=0)
unc[3] = np.std(qpu_reads_unreg, axis=0)
data[4] = np.mean(qpu_reads_reg, axis=0)
unc[4] = np.std(qpu_reads_reg, axis=0)

n_methods = len(data)
# ibin = np.arange(n_methods)
ibin = np.array([1, 2, 3, 4, 5])
colors = ['black', 'red', 'gold', 'seagreen', 'blue']
#          'gold', 'cyan', 'violet', 'navyblue']
# colors = ['black', 'salmon', 'royalblue', 'lightgreen', 'gold']
markers = ['o', 'v', '^', 'D', 'o']
bar_width = 0.1

fig, ax = plt.subplots(tight_layout=True, figsize=(10, 6))

plt.step(ibin, data[0], where='mid',
         label=labels[0], color='black', linestyle='dashed')
for i in range(1, 5):
    plt.errorbar(x=ibin+0.05*i-0.1, y=data[i],
                 yerr=unc[i],
                 color=colors[i],
                 fmt=markers[i],
                 ms=10,
                 label=labels[i])
plt.xlim(0.5, 5.5)
plt.legend()
plt.ylabel("Unfolded")
plt.xlabel("Bin")
ax.xaxis.label.set_fontsize(12)
ax.yaxis.label.set_fontsize(12)
# ax.get_xticklabels().set_fontsize(12)
# ax.get_yticklabels().set_fontsize(12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
fig.savefig("unfolded.png")
