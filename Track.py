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

class Track_():

    def __init__(self, entry, idx, event, weight, track_obj, track_particle, hists=True):
        self.entry = entry
        self.idx = idx
        self.event = event
        self.weight = weight
        if hists:
            self.track_obj = track_obj
        else:
            self.track_obj = None
        self.P = track_obj.P
        self.PT = track_obj.PT
        self.Mass = track_obj.Mass
        self.Eta = track_obj.Eta
        self.Phi = track_obj.Phi
        self.L = track_obj.L
        self.D0 = track_obj.D0
        self.DZ = track_obj.DZ
        self.ErrorD0 = track_obj.ErrorD0
        self.ErrorDZ = track_obj.ErrorDZ
        self.particle = track_particle
        #self.TruthTau = self.particle.PID == 15 or self.particle.PID == -15
        self.deltaEta = 0
        self.deltaPhi = 0
        self.deltaR = 0
        self.CoreRegion = False
        self.IsoRegion = False

    def Jet_Association(self, jetEta, jetPhi):
        self.deltaEta = jetEta - self.Eta
        self.deltaPhi = jetPhi - self.Phi
        self.deltaR = math.sqrt((self.deltaEta)**2+(self.deltaPhi)**2)
        if self.deltaR < 0.2:
            self.CoreRegion = True
        elif self.deltaR > 0.2 and self.deltaR <= 0.4:
            self.IsoRegion = True
