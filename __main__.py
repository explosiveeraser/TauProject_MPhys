import numpy as np
from ROOT import *
from ROOT import gROOT
import numba
from numba import jit, jit_module
import os, os.path
from event_gen.process_MCdata import *
from DataReader import Dataset

"""
Load delphes shared library located in 
delphes install library directory
"""
ROOT.gSystem.Load("install/lib/libDelphes")

# ROOT.ROOT.EnableImplicitMT()
ROOT.gStyle.SetOptStat(0)

background = Dataset("Delphes_Background/")
signal = Dataset("Delphes_Signal/")


