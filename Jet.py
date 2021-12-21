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
from Track import Track_
from Tower import Tower_

ROOT.gSystem.Load("install/lib/libDelphes")

class Jet_():

    def __init__(self, entry, idx, event, weight, jet_obj, particles, Event_Tracks, constituents):
        self.entry = entry
        self.idx = idx
        self.event = event
        self.weight = weight
        self.jet_obj = jet_obj
        self.PT = jet_obj.PT
        self.Eta = jet_obj.Eta
        self.Phi = jet_obj.Phi
        self.DR = math.sqrt((jet_obj.DeltaEta) **2 + (jet_obj.DeltaPhi) **2)
        self.MeanSqDR = jet_obj.MeanSqDeltaR
        self.TruthTau = False
        self.TauCan_1Prong = False
        self.TauCan_3Prong = False
        self.particles = particles
        self.TruthTau = self._Contains_Tau(self.particles)
        self.constituents = constituents
        self.Tracks = []
        self.Towers = []
        self._Find_Tracks(Event_Tracks)
        self._Add_Towers()

    def _Find_Tracks(self, evt_tracks):
        num_tracks = len(evt_tracks)
        for idx in range(0, num_tracks):
            check_particle = evt_tracks[idx].particle
            check = self.particles.Contains(check_particle)
            if check:
                track = evt_tracks[idx]
                self.Tracks.append(track)

    def _Add_Towers(self):
        num_const = len(self.constituents)
        for idx in range(0, num_const):
            const = self.constituents[idx]
            if const.ClassName() == "Tower":
                tower = Tower_(self.entry, self.event, self.weight, const)
                self.Towers.append(tower)

    def _Contains_Tau(self, particles):
        num1 = particles.GetEntries()
        found_tau = False
        for i in range(0, num1):
            test = particles.At(i).PID
            if test == 15 or test == -15:
                found_tau = True
        return found_tau
