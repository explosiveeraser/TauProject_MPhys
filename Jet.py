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
        self.TauCan_1Prong = False
        self.TauCan_3Prong = False
        self.particles = particles
        self.TruthTau = self._Contains_Tau(self.particles)
        self.TruthTau = jet_obj.TauTag
        self.constituents = constituents
        self.Tracks = []
        self.Towers = []
        self.Core_Tracks = []
        self.Iso_Tracks = []
        self.Core_Towers = []
        self.Iso_Towers = []
        self._Find_Tracks(Event_Tracks)
        self._Add_Towers()
        self.Regions_Tracks()
        self.Regions_Towers()
        self.Central_Energy_Fraction()
        self.Inverse_MomFrac_LeadTrack()
        self.Maximum_deltaR()
        self.impactP_leadTrack()
        self.F_IsoTracks()

    def _Find_Tracks(self, evt_tracks):
        num_tracks = len(evt_tracks)
        self.num_tracks = 0
        for idx in range(0, num_tracks):
            check_particle = evt_tracks[idx].particle
            check = self.particles.Contains(check_particle)
            if check:
                track = evt_tracks[idx]
                self.Tracks.append(track)
                self.Tracks[self.num_tracks].Jet_Association(self.Eta, self.Phi)
                self.num_tracks += 1

    def _Add_Towers(self):
        num_const = len(self.constituents)
        self.num_towers = 0
        for idx in range(0, num_const):
            const = self.constituents[idx]
            if const.ClassName() == "Tower":
                tower = Tower_(self.entry, self.event, self.weight, const)
                self.Towers.append(tower)
                self.Towers[self.num_towers].Jet_Association(self.Eta, self.Phi)
                self.num_towers += 1

    def _Contains_Tau(self, particles):
        num1 = particles.GetEntries()
        found_tau = False
        for i in range(0, num1):
            test = particles.At(i).PID
            if test == 15 or test == -15:
                print("True")
                found_tau = True
        return found_tau

    def Regions_Tracks(self):
        num = len(self.Tracks)
        for idx in range(0, num):
            track = self.Tracks[idx]
            if track.deltaEta < 0.2:
                self.Core_Tracks.append(track)
            elif track.deltaEta >= 0.2:
                self.Iso_Tracks.append(track)

    def Regions_Towers(self):
        num = len(self.Towers)
        for idx in range(0, num):
            tower = self.Towers[idx]
            if tower.deltaEta < 0.2:
                self.Core_Towers.append(tower)
            elif tower.deltaEta >= 0.2:
                self.Iso_Towers.append(tower)

    #may use Eem instead of ET
    def Central_Energy_Fraction(self, deltaR1=0.1, deltaR2=0.2):
        sE1 = 0
        sE2 = 0
        for tower in self.Core_Towers:
            if tower.deltaR < deltaR1:
                sE1 += tower.ET
            elif tower.deltaR < deltaR2:
                sE2 += tower.ET
        try:
            self.f_cent = sE1/sE2
        except:
            self.f_cent = 9999999999

    def Inverse_MomFrac_LeadTrack(self):
        TE_C = 0
        for tower in self.Core_Towers:
            TE_C += tower.ET
        hPT = 0
        for track in self.Core_Tracks:
            if track.PT >= hPT:
                hPT = track.PT
        try:
            self.iF_leadtrack = TE_C/hPT
        except:
            self.iF_leadtrack = 99999999

    def Maximum_deltaR(self):
        self.max_deltaR = 0
        for track in self.Core_Tracks:
            if track.deltaR >= self.max_deltaR:
                self.max_deltaR = track.deltaR

    def impactP_leadTrack(self):
        try:
            leadtrack = self.Core_Tracks[0]
            for track in self.Core_Tracks:
                if track.PT >= leadtrack.PT:
                    leadtrack = track
            try:
                self.impactD0 = leadtrack.D0/leadtrack.ErrorD0
            except:
                self.impactD0 = 999999999
        except:
            self.impactD0 = 99999999

    #def TransFlightPath

    def F_IsoTracks(self):
        Iso_PT = 0
        All_PT = 0
        for track in self.Iso_Tracks:
            Iso_PT += track.PT
        for track in self.Tracks:
            All_PT += track.PT
        try:
            self.Ftrack_Iso = Iso_PT/All_PT
        except:
            self.Ftrack_Iso = 999999999

    #def mass_TandEMSystem(self):
     #find track masses (use jitted function)?

    #def mass_TrackSys(self):
    #similar to above



