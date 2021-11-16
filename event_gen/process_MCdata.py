import numpy as np
import ROOT
from array import array
import os, os.path
from numba import int32
import numba

ROOT.gSystem.Load("install/lib/libDelphes")

@ROOT.Numba.Declare(['RVec<unsigned int>'], 'RVec<int>')
def find_tagged_jets(tautag):
    pntr=np.array([int32(0)])
    for i in range(0, len(tautag)):
        if tautag[i] == True:
            pntr = np.append(pntr, int32(i))
    return pntr[1:]


def read_tree_files(dir_name):
    dir_loc = str(dir_name+"/")
    chain = ROOT.TChain("Delphes")
    for f in os.listdir(dir_loc):
        chain.Add(str(dir_loc+f))
    return chain

def get_hist(column, type, xbins):
    th1d = ROOT.RDF.TH1DModel
    zeros = np.zeros(xbins+1, dtype=type)
    return th1d(column, column, xbins, zeros)



