#!/usr/bin/env python3

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
rc('legend',**{'fontsize': 13})

known_methods = [
    'IB4',
    'sim',
    'qpu_4bits_reg0',
    'qpu_4bits_reg1',
    'qpu_8bits_reg0',
    'qpu_8bits_reg1',
]
n_methods = len(known_methods)

labels = {
    'pdata'             : "True value",
    'IB4'               : "D\'Agostini ItrBayes ($N_{itr}$=4)",
    'sim'               : "QUBO (CPU, Neal)",
    'qpu_4bits_reg0'  : "QUBO (QPU, 4 bits enc, $\lambda$=0)",
    'qpu_4bits_reg1'  : "QUBO (QPU, 4 bits enc, $\lambda$=1)",
    'qpu_8bits_reg0'  : "QUBO (QPU, 8 bits enc, $\lambda$=0)",
    'qpu_8bits_reg1'  : "QUBO (QPU, 8 bits enc, $\lambda$=1)",
  
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

parser = argparse.ArgumentParser("Quantum unfolding plotter")
parser.add_argument('-o', '--observable', default='peak')
args = parser.parse_args()

obs = args.observable

z = input_data[obs]['pdata']
nbins = z.shape[0]

unfolded_data = {
        'pdata' : {
            'mean' : z,
            'rms'  : np.zeros(nbins),
        },
        'IB4' : input_data[obs]['IB4'],
        'sim'              : FromFile(f"results.obs_{obs}.sim.reg_0.csv"),
        'qpu_4bits_reg0' : FromFile(f"results.obs_{obs}.qpu_lonoise.reg_0.4bits.csv"),
        'qpu_4bits_reg1' : FromFile(f"results.obs_{obs}.qpu_lonoise.reg_1.4bits.csv"),
        'qpu_8bits_reg0' : FromFile(f"results.obs_{obs}.qpu_lonoise.reg_0.8bits.csv"),
        'qpu_8bits_reg1' : FromFile(f"results.obs_{obs}.qpu_lonoise.reg_1.8bits.csv"),
}


colors = ['black', 'red', 'gold', 'seagreen', 'blue','violet','cyan']
#          'gold', 'cyan', 'violet', 'navyblue']
# colors = ['black', 'salmon', 'royalblue', 'lightgreen', 'gold']
markers = ['o', 'v', '^', 'D', 'o', 'D', 'o']
bar_width = 0.1

fig, ax = plt.subplots(tight_layout=True, figsize=(10, 6))

ibin = np.arange(1,nbins+1) # position along the x-axis

print("Truth")
print(unfolded_data['pdata']['mean'])

# First, plot truth-level distribution
plt.step([0] + list(ibin), 
        [unfolded_data['pdata']['mean'][0]]+list(unfolded_data['pdata']['mean']),
        label=labels['pdata'], color='black', linestyle='dashed')


#plt.step(ibin, unfolded_data['pdata']['mean'], where='mid',
#         label=labels['pdata'], color='black', linestyle='dashed')

for i in range(1, n_methods+1):
    method = known_methods[i-1]

    print(method)
    print(unfolded_data[method]['mean'])
    print(unfolded_data[method]['rms'])

    plt.errorbar(x=ibin+0.1*i-0.8, 
                 y=unfolded_data[method]['mean'],
                 yerr=unfolded_data[method]['rms'],
                 color=colors[i],
                 fmt=markers[i],
                 ms=10,
                 label=labels[method])

plt.xlim(-0.2, 5.2)
plt.legend()
plt.ylabel("Unfolded")
plt.xlabel("Bin")
ax.xaxis.label.set_fontsize(14)
ax.yaxis.label.set_fontsize(14)
plt.xticks(np.arange(5)+0.5, [1, 2, 3, 4, 5], fontsize=14)
plt.yticks(fontsize=14)
plt.show()
fig.savefig( f"unfolded_{obs}_nbits.pdf")
