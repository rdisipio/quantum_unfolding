#!/usr/bin/env python

import os
import sys
import argparse

from ROOT import *
import numpy as np
import scipy.stats

from input_data import *

np.set_printoptions(precision=2, linewidth=200, suppress=True)

###########################


def array_to_th1(a, hname="h", htitle=";X;Entries"):
    n_bins = a.shape[0]

    h = TH1F(hname, htitle, n_bins, 0.5, n_bins+0.5)

    for i in range(n_bins):
        h.SetBinContent(i+1, a[i])

    return h

#~~~~~~~~~~~~~~~~~~~~~~~~~~


def th1_to_array(h):
    n = h.GetNbinsX()
    a = [h.GetBinContent(i+1) for i in range(n)]
    u = [h.GetBinError(i+1) for i in range(n)]
    a = np.array(a)  # , dtype='uint8')
    u = np.array(u)
    return a, u

#~~~~~~~~~~~~~~~~~~~~~~~~~~


def array_to_th2(a, hname="res", htitle=";reco;truth"):
    n_bins_x = a.shape[0]
    n_bins_y = a.shape[1]

    h = TH2F(hname, htitle, n_bins_x,
             0.5, n_bins_x+0.5, n_bins_y, 0.5, n_bins_y+0.5)

    for i in range(n_bins_x):
        for j in range(n_bins_y):
            h.SetBinContent(i+1, j+1, a[i][j])

    return h

###########################


parser = argparse.ArgumentParser("Quantum unfolding")
parser.add_argument('-l', '--lmbd', default=0.00)
parser.add_argument('-n', '--nreads', default=1000)
args = parser.parse_args()

num_reads = int(args.nreads)

#d = np.array(d, dtype='uint8')

# in case Nsyst>0, extend vectors to include syst shifts
x = np.hstack((x, np.zeros(Nsyst)))
z = np.hstack((z, s))
d = np.dot(R, z)
y = np.dot(R, x)
#y = np.dot(R, z)

print("INFO: Signal truth-level x:")
print(x)
print("INFO: Pseudo-data truth-level z:")
print(z)
print("INFO: signal detector-level y:")
print(y)
print("INFO: pseudo-data d:")
print(d)
print("INFO: Response matrix R (incl syst):")
print(R)

h_z = array_to_th1(z, "data_truth")
h_x = array_to_th1(x, "truth")
h_R = array_to_th2(R, "response")  # .T ?
h_y = array_to_th1(y, "signal")
h_d = array_to_th1(d, "data")
# h_R.Draw("text")
loaded_RooUnfold = gSystem.Load("libRooUnfold.so")
if not loaded_RooUnfold == 0:
    print "INFO: RooUnfold not found."
else:
    print "INFO: RooUnfold found."

# see: http://hepunx.rl.ac.uk/~adye/software/unfold/RooUnfold.html

m_response = RooUnfoldResponse(h_y, h_x, h_R)
m_response.UseOverflow(False)

N = h_x.GetNbinsX()
dof = N-1

unfolder_mi = RooUnfoldInvert("MI", "Matrix Inversion")
unfolder_mi.SetVerbose(2)
unfolder_mi.SetResponse(m_response)
unfolder_mi.SetMeasured(h_d)
h_unf_mi = unfolder_mi.Hreco()
h_unf_mi.SetName("unf_mi")
u_mi, e_mi = th1_to_array(h_unf_mi)
print("INFO: unfolded (MI):")
print(u_mi)
print(e_mi)
chi2_mi, p_mi = scipy.stats.chisquare(u_mi, z)
print("chi2 / dof = %f / %i = %.2f" % (chi2_mi, dof, chi2_mi/float(dof)))
Rinv = np.linalg.pinv(R)
print("INFO: R^-1:")
print(Rinv)
u_mi = np.dot(Rinv, d)
print("INFO: R^-1 d:")
print("u:", u_mi)
print("z:", z)
chi2_mi, p_mi = scipy.stats.chisquare(u_mi, z)
print("chi2 / dof = %f / %i = %.2f" % (chi2_mi, dof, chi2_mi/float(dof)))

unfolder_ib = RooUnfoldBayes("IB", "Iterative Baysian")
unfolder_ib.SetIterations(4)
unfolder_ib.SetVerbose(2)
unfolder_ib.SetSmoothing(0)
unfolder_ib.SetResponse(m_response)
unfolder_ib.SetMeasured(h_d)
h_unf_ib = unfolder_ib.Hreco()
h_unf_ib.SetName("unf_ib")
u_ib, e_ib = th1_to_array(h_unf_ib)
print("INFO: unfolded (IB):")
print(u_ib)
print(e_ib)
chi2_ib, p_ib = scipy.stats.chisquare(u_ib, z)
print("chi2 / dof = %f / %i = %.2f" % (chi2_ib, dof, chi2_ib/float(dof)))


unfolder_svd = RooUnfoldSvd("SVD", "SVD Tikhonov")
unfolder_svd.SetKterm(3)  # usually nbins//2
unfolder_svd.SetVerbose(1)
unfolder_svd.SetResponse(m_response)
unfolder_svd.SetMeasured(h_d)
h_unf_svd = unfolder_svd.Hreco()
h_unf_svd.SetName("unf_svd")
u_svd, e_svd = th1_to_array(h_unf_svd)
print("INFO: unfolded (SVD):")
print(u_svd)
print(e_svd)
chi2_svd, p_svd = scipy.stats.chisquare(u_svd, z)
print("chi2 / dof = %f / %i = %.2f" % (chi2_svd, dof, chi2_svd/float(dof)))

print("INFO: Truth-level z:")
print(z)
