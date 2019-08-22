#!/usr/bin/env python3

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

np.set_printoptions(precision=3, linewidth=500, suppress=True)

parser = argparse.ArgumentParser("Quantum unfolding plotter")
parser.add_argument('-o', '--observable', default='peak')
parser.add_argument('-e', '--encoding', default=4)
args = parser.parse_args()

nbits = int(args.encoding)
obs = args.observable
n_syst = 2
n_bins = 5

csv_file = f"csv/results_syst.obs_{obs}.qpu_lonoise.reg_0.gamma_0.{nbits}bits.csv"
print("INFO: reading input file", csv_file)

unfolded_data = np.genfromtxt( csv_file, delimiter=',' )
#print("unfolded data:")
#print(unfolded_data)

n_entries = unfolded_data.shape[0]
print("INFO: number of entries:", n_entries)

sigma = unfolded_data[:,n_bins:]
#print("systematics:")
#print(sigma)

#syst_hist_1 = np.histogram( sigma[1], range=[-3,3], bins=12 ) 
#print(syst_hist_1)
#print("syst 0:")
#print(sigma[:,0])

fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=False, tight_layout=True)

srange = [-3,3]
bins = np.arange(-3.0,3.5,0.5)
#bins=12
axs[0].hist( sigma[:,0], range=srange, bins=bins )#, density=True )
axs[1].hist( sigma[:,1], range=srange, bins=bins ) #, density=True ) 
axs[2].hist2d( sigma[:,0], sigma[:,1], 
                range=( srange, srange ), 
                bins=(9,9), 
                cmap=plt.cm.get_cmap('Greys'))


plt.show()
fig.savefig(f"systematics_{obs}_{nbits}bits.pdf")
