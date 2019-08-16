#!/usr/bin/env python3

import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set()
sns.set_style("white")

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
rc('legend',**{'fontsize': 13})

known_methods = [
    #'IB4',
    #'sim',
    'lower noise 4bits',
    'lower noise 8bits',
    'regular noise 4bits',
    'regular noise 8bits',
]
n_methods = len(known_methods)

labels = {
    'pdata'             : "True value",
    'IB4'               : "D\'Agostini ItrBayes ($N_{itr}$=4)",
    'sim'               : "QUBO (CPU, Neal)",
    'lower noise 4bits'  : "QUBO (QPU, lower noise, 4 bits enc)",
    'lower noise 8bits'  : "QUBO (QPU, lower noise, 8 bits enc)",
    'regular noise 4 bits'  : "QUBO (QPU, regular noise, 4 bits enc)",
    'regular noise 8 bits'  : "QUBO (QPU, regular noise, 8 bits enc)",
}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def FromFile( csv_file ):
    data = np.genfromtxt( csv_file, delimiter=',' )

    data = np.swapaxes(data,0,1)

    return data

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from input_data import *

parser = argparse.ArgumentParser("Quantum unfolding plotter")
parser.add_argument('-o', '--observable', default='falling')
args = parser.parse_args()

obs = args.observable

z = input_data[obs]['pdata']
nbins = z.shape[0]

pdata = {
            'mean' : z,
            'rms'  : np.zeros(nbins),
        }
    
unfolded_data = {
        'lower noise 4bits' : FromFile(f"csv/results.obs_{obs}.qpu_lonoise.reg_0.4bits.csv"),
        'lower noise 8bits' : FromFile(f"csv/results.obs_{obs}.qpu_lonoise.reg_0.8bits.csv"),
        'regular noise 4bits' : FromFile(f"csv/results.obs_{obs}.qpu_hinoise.reg_0.4bits.csv"),
        'regular noise 8bits' : FromFile(f"csv/results.obs_{obs}.qpu_hinoise.reg_0.8bits.csv"),
}

nreads = 20
nbins = 5
raw_data = []
for method in known_methods:
    for iread in range(nreads):
        for ibin in range(nbins):
            raw_data.append( {
                'method' : method,
                'bin'    : (ibin+1),
                'unf'    : unfolded_data[method][ibin][iread],
            } )

df = pd.DataFrame.from_dict(raw_data)
    

colors = {
    'pdata' : 'black',
    'lower noise 4bits' : 'green',
    'lower noise 8bits' : "blue",
    'regular noise 4bits' : 'red',
    'regular noise 8bits' : 'gold'
}
facecolors = {
    'lower noise 4bits' : 'green',
    'lower noise 8bits' : "none",
    'regular noise 4bits' : 'red',
    'regular noise 8bits' : 'none'
}
grayscale = {
    'lower noise 4bits' : 'lightgray',
    'lower noise 8bits' : "lightgray",
    'regular noise 4bits' : 'lightgray',
    'regular noise 8bits' : 'lightgray'
}

fig, ax = plt.subplots(tight_layout=True, figsize=(10, 6))

ibin = np.arange(-0.5,nbins+0.5) # position along the x-axis

# First, plot truth-level distribution
plt.step( list(ibin), 
            [pdata['mean'][0]]+list(pdata['mean']),
            label=labels['pdata'], color='black', linestyle='dashed')

#sns.boxplot( x='bin', y='unf',
#            hue='method',
#             palette=grayscale,
 #            fliersize=0,
 #            data=df,
 #             orient='v' 
 #       )
#stripplot

sns.swarmplot( x='bin', y='unf', 
                hue='method', palette=colors,
                data=df,
                orient='v',
                dodge=True,
                #jitter=True,
                #width=0.4,
                size=6, lw=4, edgecolor='black',
                )

#handles, labels = ax.get_legend_handles_labels()
#new_handles = [ handles[0], handles[5], handles[6], handles[7], handles[8] ]
#new_labels  = [ labels[0], labels[5], labels[6], labels[7], labels[8] ]
#plt.legend(new_handles, new_labels)

plt.xlim(-1, 5)

#plt.legend()
plt.ylabel("Unfolded")
plt.xlabel("Bin")
ax.xaxis.label.set_fontsize(14)
ax.yaxis.label.set_fontsize(14)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)

plt.show()
fig.savefig( f"unfolded_{obs}_nbits_violin.pdf")
