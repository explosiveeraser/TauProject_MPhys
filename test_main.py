import numpy as np
#import modin.pandas as pd
from DataSet_Reader import Dataset
from Background_DataReader import Background
from Signal_DataReader import Signal
import ROOT
from ROOT import gROOT
import numba
from numba import jit, jit_module
import os, os.path
from tqdm import tqdm, trange
from DataProcessing import DataProcessing

"""
Load delphes shared library located in 
delphes install library directory
"""

ROOT.gSystem.Load("../Delphes-3.5.0/build/libDelphes.so")

try:
  ROOT.gInterpreter.Declare('#include "classes/DelphesClasses.h"')
  ROOT.gInterpreter.Declare('#include "external/ExRootAnalysis/ExRootTreeReader.h"')
except:
  pass


ROOT.ROOT.EnableImplicitMT()
ROOT.gStyle.SetOptStat("ne")

sig_dir = "Delphes_Signal/"
back_dir = "Delphes_Background/"
sig_wPU_dir = "../sdb5/Delphes_Signal_wPU/0_file/"
a_back_wPU_dir = "../sdb5/Delphes_Background_wPU/0_file/"
b_back_wPU_dir = "../sdb5/Delphes_Background_wPU/1_file/"
c_back_wPU_dir = "../sdb5/Delphes_Background_wPU/2_file/"
d_back_wPU_dir = "../sdb5/Delphes_Background_wPU/3_file/"
e_back_wPU_dir = "../sdb5/Delphes_Background_wPU/4_file/"

back_wPU = [a_back_wPU_dir, b_back_wPU_dir, c_back_wPU_dir, d_back_wPU_dir, e_back_wPU_dir]

Data = DataProcessing(sig_dir, back_dir, sig_wPU_dir, back_wPU)

print_hists = False

if print_hists:
  Data.Sig_Hist_Tau()
  Data.Back_Hist_Tau()
  Data.Tau_Sig_Back_Hist()
  Data.Sig_Back_Hist()
  #Data.Print_Test()
  Data.Print_Num_of_Tau()
  Data.Print_Canvases()

input("Enter to quit")

