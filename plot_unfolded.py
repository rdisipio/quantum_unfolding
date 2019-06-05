#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns

from input_data import *

# sns.set(color_codes=True)

n_methods = len(unf_data)
# ibin = np.arange(n_methods)
ibin = np.array([1, 2, 3, 4, 5])
colors = ['black', 'red', 'gold', 'seagreen', 'blue']
#          'gold', 'cyan', 'violet', 'navyblue']
# colors = ['black', 'salmon', 'royalblue', 'lightgreen', 'gold']
markers = ['o', 'v', '^', 'D', 'o']
bar_width = 0.1

fig, ax = plt.subplots(tight_layout=True, figsize=(10, 6))

plt.step(ibin, unf_data[0], where='mid',
         label=unf_data_labels[0], color='black', linestyle='dashed')
for i in range(1, 5):
    plt.errorbar(x=ibin+0.1*i-0.2, y=unf_data[i],
                 yerr=unf_data_unc[i],
                 color=colors[i],
                 fmt=markers[i],
                 ms=10,
                 label=unf_data_labels[i])
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
