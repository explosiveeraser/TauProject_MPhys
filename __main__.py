import numpy as np
import ROOT
from array import array
from ROOT import gROOT
from event_gen.process_MCdata import process_data

ROOT.ROOT.EnableImplicitMT()
ROOT.gStyle.SetOptStat(0)

background_dir = "Delphes_Background"
signal_dir = "Delphes_Signal"

background_chain = process_data.read_tree_files(background_dir)
print(process_data.read_tree_files(signal_dir))
