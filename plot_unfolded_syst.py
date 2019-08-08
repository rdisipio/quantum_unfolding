#!/usr/bin/env python3

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
rc('legend',**{'fontsize': 13})

parser = argparse.ArgumentParser("Quantum unfolding plotter")
parser.add_argument('-o', '--observable', default='peak')
parser.add_argument('-e', '--encoding', default=4)
args = parser.parse_args()

nbits = int(args.encoding)
obs = args.observable

known_methods = [
    #'IB4',
    'sim',
    'qpu_lonoise_reg0_4bits',
    'qpu_lonoise_reg0_8bits',
]
n_methods = len(known_methods)

labels = {
    'pdata'             : "True value",
    #'IB4'               : "D\'Agostini ItrBayes ($N_{itr}$=4)",
    'sim'               : "QUBO (CPU, Neal)",
    'qpu_lonoise_reg0_4bits'  : "QUBO (QPU, lower noise, $\lambda$=0, $\gamma$=1, 4-bits)",
    'qpu_lonoise_reg0_8bits'  : "QUBO (QPU, lower noise, $\lambda$=0, $\gamma$=1, 8-bits)",
    #'hyb_reg0'          : "QUBO (Hybrid, $\lambda$=0)",
    #'hyb_reg1'          : "QUBO (Hybrid, $\lambda$=1)",
}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def FromFile( csv_file ):
    data = np.genfromtxt( csv_file, delimiter=',' )

    return {
        'mean' : np.mean( data, axis=0 ),
        'rms'  : np.std( data, axis=0)
    }

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from input_data import *

#z = np.hstack( [input_data[obs]['pdata'], sigma_syst] )
z = input_data[obs]['pdata']
nbins = input_data[obs]['pdata'].shape[0]

unfolded_data = {
        'pdata' : {
            'mean' : z,
            'rms'  : np.zeros(nbins),
        },
        #'IB4' : input_data[obs]['IB4'],
        'sim'              : FromFile(f"results_syst.obs_{obs}.sim.reg_0.gamma_1.4bits.csv"),
        'qpu_lonoise_reg0_4bits' : FromFile(f"results_syst.obs_{obs}.qpu_lonoise.reg_0.gamma_1.4bits.csv"),
        'qpu_lonoise_reg0_8bits' : FromFile(f"results_syst.obs_{obs}.qpu_lonoise.reg_0.gamma_1.8bits.csv"),
        
        #'hyb_reg0'         : FromFile(f"results.obs_{obs}.hyb.reg_0.csv"),
        #'hyb_reg1'         : FromFile(f"results.obs_{obs}.hyb.reg_1.csv"),
}

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

ax_main.step([0] + list(ibin), 
        [unfolded_data['pdata']['mean'][0]]+list(unfolded_data['pdata']['mean']),
        label=labels['pdata'], color='black', linestyle='dashed')


for i in range(1, n_methods+1):
    method = known_methods[i-1]

    print(method)
    print(unfolded_data[method]['mean'])
    print(unfolded_data[method]['rms'])

    plt.errorbar(x=ibin+0.1*i-0.8, 
                 y=unfolded_data[method]['mean'][:Nbins],
                 yerr=unfolded_data[method]['rms'][:Nbins],
                 color=colors[i],
                 fmt=markers[i],
                 ms=10,
                 label=labels[method])

plt.xlim(-0.2, 5.2)
ax_main.legend()
ax_main.set_ylabel("Unfolded")
ax_main.set_xlabel("Bin")
ax_main.xaxis.label.set_fontsize(14)
ax_main.yaxis.label.set_fontsize(14)
#ax_main.xticks(np.arange(5)+0.5, [1, 2, 3, 4, 5], fontsize=14)
ax_main.tick_params(labelsize=14)

for isyst in range(Nsyst):
    l_length = 0.2
    x = sigma_syst[isyst]
    ymin = (1/float(Nsyst))*(0.5+isyst)-l_length
    ymax = (1/float(Nsyst))*(0.5+isyst)+l_length
    print(x, ymin, ymax)
    ax_syst.axvline(x=x,
                    ymin=ymin,
                    ymax=ymax,
                    color='black',
                    linestyle='dashed')

    for imethod in range(1, n_methods):
        method = known_methods[i-1]

        x = unfolded_data[method]['mean'][Nbins+isyst]
        y = isyst - 0.05*imethod + 0.1
        dx = unfolded_data[method]['rms'][Nbins+isyst]

        ax_syst.errorbar(x,
                         y,
                         xerr=dx,
                         color=colors[imethod],
                         fmt=markers[imethod],
                         ms=10,
                         label=labels[method])


# ax_syst.get_yaxis().set_visible(False)
ax_syst.set_xlim(-0.5, 5.5)
ax_syst.set_ylim(-0.5, Nsyst-0.5)
ax_syst.xaxis.label.set_fontsize(14)
ax_syst.set_xlabel("$\lambda$")
ax_syst.tick_params(labelsize=14)
ax_syst.set_yticks(np.arange(Nsyst))
ax_syst.set_yticklabels(["syst1", "syst2"])


plt.show()
fig.savefig("unfolded_syst.png")
