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

    def __init__(self, entry, evt, particle, branch, hists=True):
        self.entry = entry
        self.event = evt
        if hists:
            self.particle_obj = particle
        else:
            self.particle_obj = None
        PID = particle.PID
        self.PID = PID
        self.Eta = particle.Eta
        self.Phi = particle.Phi
        self.deltaEta = 0
        self.deltaPhi = 0
        self.deltaR = 0
        self.CoreRegion = False
        self.IsoRegion = False
        if PID == 15 or PID == -15:
            self.tau = True
            daughters = self.getStableDaughters(branch, particle, [])
            #print(daughters)
            self.tau_prongness = sum([abs(d) for d in daughters])
            #print(self.tau_prongness)
        else:
            self.tau = False

    def Jet_Association(self, jetEta, jetPhi):
        self.deltaEta = jetEta - self.Eta
        self.deltaPhi = jetPhi - self.Phi
        self.deltaR = math.sqrt((self.deltaEta)**2+(self.deltaPhi)**2)
        if self.deltaR < 0.2:
            self.CoreRegion = True
        elif self.deltaR > 0.2 and self.deltaR <= 0.6:
            self.IsoRegion = True

    def getStableDaughters(self, branch, p_object, daughters):
        #print("This particle has: D1: {} and D2: {}".format(p_object.PID, p_object.PID))
        if p_object.Status == 1:
            return [p_object.Charge]
        else:
            if p_object.D1 == p_object.D2:
                d = branch.At(p_object.D1)
                daughters = self.getStableDaughters(branch, d, [])
            elif p_object.D1 != p_object.D2:
                for id in range(p_object.D1, p_object.D2+1):
                    di = branch.At(id)
                    daughters += self.getStableDaughters(branch, di, [])
        return daughters
