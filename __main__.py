import numpy as np
import ROOT
from array import array
from ROOT import gROOT
import numba
from event_gen.process_MCdata import process_data

ROOT.ROOT.EnableImplicitMT()
ROOT.gStyle.SetOptStat(0)
#ROOT.gSystem.Load("TTreeViewer\/")

background_dir = "Delphes_Background"
signal_dir = "Delphes_Signal"

background_chain = process_data.read_tree_files(background_dir)
signal_chain = process_data.read_tree_files(signal_dir)


background_df = ROOT.RDataFrame(background_chain)
signal_df = ROOT.RDataFrame(signal_chain)

column_names = background_df.GetColumnNames()

jet_cons_B = background_df["Jet"]


input("blob")