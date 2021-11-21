import numpy as np
import ROOT
from ROOT import gROOT
import numba
from numba import jit, jit_module
import os, os.path
from tqdm import tqdm, trange
from DelphesAnalysis import *


ROOT.gSystem.Load("install/lib/libDelphes")


class Dataset():
    def __init__(self, directory, get_Histos=False):
        if "/" in directory:
            self.name = directory[:-1]
        else:
            self.name = directory
            directory = directory + "/"
        f_input = os.listdir(directory)
        self.chain = AnalysisEvent.AnalysisEvent(f_input)