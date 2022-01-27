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

class Tower_():

    def __init__(self, entry, evt, weight, tower, hists=True):
        self.entry = entry
        self.event = evt
        self.weight = weight
        if hists:
            self.tower_obj = tower
        else:
            self.tower_obj = None
        self.particles = tower.Particles
        self.E = tower.E
        self.ET = tower.ET
        self.Eta = tower.Eta
        self.Phi = tower.Phi
        self.Edges = tower.Edges
        self.Eem = tower.Eem
        self.Ehad = tower.Ehad
        self.T = tower.T
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
        elif self.deltaR > 0.2 and self.deltaR <= 0.6:
            self.IsoRegion = True
