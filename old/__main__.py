import numpy as np
import ROOT
from array import array
from ROOT import gROOT
import numba
from numba import jit, jit_module
from event_gen.process_MCdata import *
import subprocess

"""
Load delphes shared library located in 
delphes install library directory
"""
ROOT.gSystem.Load("install/lib/libDelphes")

# ROOT.ROOT.EnableImplicitMT()
ROOT.gStyle.SetOptStat(0)

background_dir = "../Delphes_Background"
signal_dir = "../Delphes_Signal"

back_chain = read_tree_files(background_dir)
sig_chain = read_tree_files(signal_dir)

back_reader = ROOT.ExRootTreeReader(back_chain)
sig_reader = ROOT.ExRootTreeReader(sig_chain)

jets = back_reader.UseBranch("Jet")
towers = back_reader.UseBranch("Tower")
tracks = back_reader.UseBranch("Track")



back_df = ROOT.RDataFrame(back_chain)
sig_df = ROOT.RDataFrame(sig_chain)



back_df = back_df.Define(str("Jet_Pntrs"), "Numba::find_tagged_jets(Jet.TauTag)")
back_df = back_df.Define(str("GenJet_Pntrs"), "Numba::find_tagged_jets(GenJet.TauTag)")


column_names = back_df.GetColumnNames()
columns = {}

for col in column_names:
    try:
        columns[str(col)] = str(back_df.GetColumnType(col))
        #print("{}||{}".format(col, columns[col]))
    except:
        pass

back_hists = {}
sig_hists = {}

for co in column_names:
    col = str(co)
    try:
        if "Float_t" in columns[col] or "Double_t" in columns[col] or "Int_t" in columns[col]:
            if "Jet." in col and col[0] == "J":
                back_hists[col] = back_df.Histo1D(get_hist(str("Background " + col), float, 250), col)
                sig_hists[col] = sig_df.Histo1D(get_hist(str("tau " + col), float, 250), col)
            elif "GenJet." in col and col[0] == "G":
                back_hists[col] = back_df.Histo1D(get_hist(str("Background " + col), float, 250), col)
                sig_hists[col] = sig_df.Histo1D(get_hist(str("tau " + col), float, 250), col)
        else:
            pass
    except:
        pass

#print(column_names)

# input("blob")
