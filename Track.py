import array
import gc
import math

import numpy as np
import ROOT
import pandas as pd
from ROOT import gROOT
import numba
from numba import jit, jit_module
import os, os.path
from tqdm import tqdm, trange

ROOT.gSystem.Load("install/lib/libDelphes")

class Track_():

    def __init__(self, entry, idx, event, track_obj, track_particle):
        self.entry = entry
        self.idx = idx
        self.event = event
        self.track_obj = track_obj
        self.PT = track_obj.PT
        self.Eta = track_obj.Eta
        self.Phi = track_obj.Phi
        self.particle = track_particle
        self.TruthTau = self.particle.PID == 15 or self.particle.PID == -15
