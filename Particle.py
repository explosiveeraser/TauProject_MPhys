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

class Particle_():

    def __init__(self, entry, evt, particle):
        self.entry = entry
        self.event = evt
        self.particle_obj = particle
        PID = particle.PID
        self.PID = PID
        if PID == 15 or PID == -15:
            self.tau = True
        else:
            self.tau = False
        self.Eta = particle.Eta
        self.Phi = particle.Phi
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