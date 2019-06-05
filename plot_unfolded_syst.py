#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns

from input_data import *

# sns.set(color_codes=True)

n_methods = len(unf_data_syst)
print("INFO: plotting %i histograms:" % n_methods)
print(unf_data_labels[:n_methods])

Nbins = 5
Nsyst = 2
ibin = np.arange(1, Nbins+1)

colors = ['black', 'red', 'gold', 'seagreen', 'blue']
#          'gold', 'cyan', 'violet', 'navyblue']
# colors = ['black', 'salmon', 'royalblue', 'lightgreen', 'gold']
markers = ['o', 'v', '^', 'D', 'o']
bar_width = 0.1

# fig, ax = plt.subplots(tight_layout=True, figsize=(10, 6))
fig = plt.figure(figsize=(12, 6))
bottom = 0.1
left = 0.05
right = 0.15
width = 0.65
height = 0.85
padding = 0.07
ax_main = plt.axes([left, bottom, width, height])
ax_syst = plt.axes([left+width+padding, bottom,
                    1.0-width-right, height])

ax_main.step(ibin, unf_data_syst[0][:Nbins], where='mid',
             label=unf_data_labels[0], color='black', linestyle='dashed')

for i in range(1, n_methods):
    y = unf_data_syst[i][:Nbins]
    dy = unf_data_syst_unc[i][:Nbins]
    print(y)
    print(dy)
    ax_main.errorbar(x=ibin+0.1*i-0.2,
                     y=y,
                     yerr=dy,
                     color=colors[i],
                     fmt=markers[i],
                     ms=10,
                     label=unf_data_labels[i])
ax_main.set_xlim(0.5, 5.5)
ax_main.legend()
ax_main.set_ylabel("Unfolded")
ax_main.set_xlabel("Bin")
ax_main.xaxis.label.set_fontsize(14)
ax_main.yaxis.label.set_fontsize(14)
ax_main.tick_params(labelsize=14)

for isyst in range(Nsyst):
    l_length = 0.2
    x = unf_data_syst[0][Nbins+isyst]
    ymin = (1/float(Nsyst))*(0.5+isyst)-l_length
    ymax = (1/float(Nsyst))*(0.5+isyst)+l_length
    print(x, ymin, ymax)
    ax_syst.axvline(x=x,
                    ymin=ymin,
                    ymax=ymax,
                    color='black',
                    linestyle='dashed')

    for imethod in range(1, n_methods):
        x = unf_data_syst[imethod][Nbins+isyst]
        y = isyst - 0.05*imethod + 0.1
        dx = unf_data_syst_unc[imethod][Nbins+isyst]

        ax_syst.errorbar(x,
                         y,
                         xerr=dx,
                         color=colors[imethod],
                         fmt=markers[imethod],
                         ms=10,
                         label=unf_data_labels[imethod])


# ax_syst.get_yaxis().set_visible(False)
ax_syst.set_xlim(-0.5, 2.5)
ax_syst.set_ylim(-0.5, Nsyst-0.5)
ax_syst.xaxis.label.set_fontsize(14)
ax_syst.set_xlabel("$\lambda$")
ax_syst.tick_params(labelsize=14)
ax_syst.set_yticks(np.arange(Nsyst))
ax_syst.set_yticklabels(["syst1", "syst2"])


plt.show()
fig.savefig("unfolded_syst.png")
