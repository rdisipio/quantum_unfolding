#!/usr/bin/env python3

import argparse

import numpy as np
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)
rc('legend', **{'fontsize': 13})

#sns.set(style="white")

np.set_printoptions(precision=2, linewidth=500, suppress=True)

parser = argparse.ArgumentParser("Quantum unfolding plotter")
parser.add_argument('-o', '--observable', default='peak')
parser.add_argument('-e', '--encoding', default=4)
args = parser.parse_args()

nbits = int(args.encoding)
obs = args.observable

known_methods = [
    'qpu_hinoise_reg0',
    'qpu_lonoise_reg0',
    #    'qpu_lonoise_reg0_gamma0',
    #    'qpu_lonoise_reg0_gamma1',
    #    'qpu_lonoise_reg0_gamma0_48bits',
]
n_methods = len(known_methods)

labels = {
    'qpu_hinoise_reg0':
    "QPU, regular noise",
    'qpu_lonoise_reg0':
    "QPU, lower noise",
    'qpu_lonoise_reg0_gamma0':
    "QPU, lower noise, $\gamma$=0",
    'qpu_lonoise_reg0_gamma1':
    "QPU, lower noise, $\gamma$=1",
    'qpu_lonoise_reg0_gamma0_48bits':
    "QPU, lower noise, $\gamma$=0, encoding=(4,8)"
}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def FromFile(csv_file):
    data = np.genfromtxt(csv_file, delimiter=',')

    return {
        'mean': np.mean(data, axis=0),
        'rms': np.std(data, axis=0),
        'corr': np.corrcoef(data, rowvar=False),
    }


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

unfolded_data = {
    'qpu_hinoise_reg0':
    FromFile(f"csv/results.obs_{obs}.qpu_hinoise.reg_0.4bits.csv"),
    'qpu_lonoise_reg0':
    FromFile(f"csv/results.obs_{obs}.qpu_lonoise.reg_0.4bits.csv"),
    'qpu_lonoise_reg0_gamma0':
    FromFile(
        f"csv/results_syst.obs_{obs}.qpu_lonoise.reg_0.gamma_0.4bits.csv"),
    'qpu_lonoise_reg0_gamma1':
    FromFile(
        f"csv/results_syst.obs_{obs}.qpu_lonoise.reg_0.gamma_1.4bits.csv"),
    'qpu_lonoise_reg0_gamma0_48bits':
    FromFile(
        f"csv/results_syst.obs_{obs}.qpu_lonoise.reg_0.gamma_0.48bits.csv"),
}

Nbins = 5
Nsyst = 2

#f, ax = plt.subplots(figsize=(11, 9))

for method in known_methods:
    print("INFO: correlation matrix for method", method)

    corr = unfolded_data[method]['corr']
    #mask = np.zeros_like(corr, dtype=np.bool)
    #mask[np.triu_indices_from(mask)] = True
    print(unfolded_data[method]['corr'])

    Nparams = Nbins
    names = ["bin1", "bin2", "bin3", "bin4", "bin5"]
    if "gamma" in method:
        Nparams += Nsyst
        names += ["norm", "shape"]

    f = plt.figure()

    ax = f.add_subplot()

    cax = ax.imshow(corr, cmap=plt.cm.get_cmap('bwr'), vmin=-1, vmax=1)
    f.colorbar(cax)
    #plt.title( labels[method])
    ax.set_xlim(-0.5, Nparams - 0.5)
    ax.set_ylim(-0.5, Nparams - 0.5)
    ticks = np.arange(Nparams)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    plt.xticks(rotation=45)

    plt.show()
    f.savefig(f"correlations_{obs}_{nbits}bits_syst_{method}.pdf")
