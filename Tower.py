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

    def __init__(self, entry, evt, weight, tower):
        self.entry = entry
        self.event = evt
        self.weight = weight
        self.tower_obj = tower
        self.ET = tower.ET
        self.Eta = tower.Eta
        self.Phi = tower.Phi
        self.deltaEta = 0
        self.deltaPhi = 0
        self.deltaR = 0

    def Jet_Association(self, jetEta, jetPhi):
        self.deltaEta = jetEta - self.Eta
        self.deltaPhi = jetPhi - self.Phi
        self.deltaR = math.sqrt((self.deltaEta)**2+(self.deltaPhi)**2)
