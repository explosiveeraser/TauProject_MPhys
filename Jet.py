import array
import gc
import math
import random

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

    def __init__(self, entry, idx, event, weight, jet_obj, particles, Event_Particles, Event_Tracks, Event_Towers, constituents, hists=True):
        self.entry = entry
        self.idx = idx
        self.event = event
        self.weight = weight
        if hists:
            self.jet_obj = jet_obj
        else:
            self.jet_obj = None
        self.PT = jet_obj.PT
        self.Eta = jet_obj.Eta
        self.Phi = jet_obj.Phi
        self.deltaEta = jet_obj.DeltaEta
        self.deltaPhi = jet_obj.DeltaPhi
        self.charge = jet_obj.Charge
        self.NCharged = jet_obj.NCharged
        self.NNeutral = jet_obj.NNeutrals
        self.DR = math.sqrt((jet_obj.DeltaEta) **2 + (jet_obj.DeltaPhi) **2)
        self.MeanSqDR = jet_obj.MeanSqDeltaR
        self.num_particles = 0
        self.num_tracks = 0
        self.num_towers = 0
        if self.PT > 20.0 and abs(self.Eta) < 2.5:
            self.TauCan_1Prong = True
            self.TauCan_3Prong = True
        else:
            self.TauCan_1Prong = False
            self.TauCan_3Prong = False
    #Remove to avoid dc (double counting)
        #self.particles = particles
        #self.TruthTau = jet_obj.TauTag
    #Removed to avoid double counting
        #self.constituents = constituents
        self.Tracks = []
        self.Towers = []
        self.Particles = []
        # self.Core_Tracks = []
        # self.Iso_Tracks = []
        # self.Core_Towers = []
        # self.Iso_Towers = []
        self.TruthTau = False
        self.numTaus = 0
        # self._Find_Tracks(Event_Tracks)
        # self._Add_Towers()
        self._Find_Particles(Event_Particles)
        self._Find_Tracks(Event_Tracks)
        self._Find_Towers(Event_Towers)
#        self.Regions_Tracks()
 #       self.Regions_Towers()
        self.Central_Energy_Fraction()
        self.Inverse_MomFrac_LeadTrack()
        self.Maximum_deltaR()
#        self.impactP_leadTrack()
        self.F_IsoTracks()


    def _Find_Particles(self, evt_particles):
        num_particles = len(evt_particles)
        for idx in range(0, num_particles):
            check_p = evt_particles[idx]
            if self.check_if_con(check_p.Eta, check_p.Phi):
                check_p.Jet_Association(self.Eta, self.Phi)
                self.Particles.append(check_p)
                self.num_particles += 1
                if check_p.PID == 15 or check_p == -15:
                    self.TruthTau = True
                    self.numTaus += 1

    def check_if_con(self, con_eta, con_phi):
        deltaEta = self.Eta - con_eta
        deltaPhi = self.Phi - con_phi
        deltaR = math.sqrt((deltaEta)**2+(deltaPhi)**2)
        if deltaR <= 0.6:
            return True
        else:
            return False

    def _Find_Tracks(self, evt_tracks):
        num_tracks = len(evt_tracks)
        for idx in range(0, num_tracks):
            check_tr = evt_tracks[idx]
            if self.check_if_con(check_tr.Eta, check_tr.Phi):
                check_tr.Jet_Association(self.Eta, self.Phi)
                self.Tracks.append(check_tr)
                self.num_tracks += 1

    def _Find_Towers(self, evt_towers):
        num_towers = len(evt_towers)
        for idx in range(0, num_towers):
            check_to = evt_towers[idx]
            if self.check_if_con(check_to.Eta, check_to.Phi):
                check_to.Jet_Association(self.Eta, self.Phi)
                self.Towers.append(check_to)
                self.num_towers += 1


    # def _Find_Tracks(self, evt_tracks):
    #     num_tracks = len(evt_tracks)
    #     self.num_tracks = 0
    #     for idx in range(0, num_tracks):
    #         check_particle = evt_tracks[idx].particle
    #         check = self.particles.Contains(check_particle)
    #         if check:
    #             track = evt_tracks[idx]
    #             self.Tracks.append(track)
    #             self.Tracks[self.num_tracks].Jet_Association(self.Eta, self.Phi)
    #             self.num_tracks += 1

    # def _Add_Towers(self):
    #     num_const = len(self.constituents)
    #     self.num_towers = 0
    #     for idx in range(0, num_const):
    #         const = self.constituents[idx]
    #         if const.ClassName() == "Tower":
    #             tower = Tower_(self.entry, self.event, self.weight, const)
    #             self.Towers.append(tower)
    #             self.Towers[self.num_towers].Jet_Association(self.Eta, self.Phi)
    #             self.num_towers += 1

    # def _Contains_Tau(self, particles):
    #     num1 = particles.GetEntries()
    #     found_tau = False
    #     for i in range(0, num1):
    #         test = particles.At(i).PID
    #         if test == 15 or test == -15:
    #             print("True")
    #             found_tau = True
    #     for track in self.Tracks:
    #         particle = track.particle
    #         if particle.PID == 15 or particle.PID == -15:
    #             found_tau = True
    #             print("True")
    #     for tower in self.Towers:
    #         particles = tower.particles
    #         num2 = particles.GetEntries()
    #         for j in range(0, num2):
    #             to_test = particles.At(j).PID
    #             if to_test == 15 or to_test == -15:
    #                 found_tau = True
    #                 print("True")
    #     return found_tau

    # def Regions_Tracks(self):
    #     num = len(self.Tracks)
    #     for idx in range(0, num):
    #         track = self.Tracks[idx]
    #         if track.deltaEta < 0.2:
    #             self.Core_Tracks.append(track)
    #         elif track.deltaEta >= 0.2:
    #             self.Iso_Tracks.append(track)
    #
    # def Regions_Towers(self):
    #     num = len(self.Towers)
    #     for idx in range(0, num):
    #         tower = self.Towers[idx]
    #         if tower.deltaEta < 0.2:
    #             self.Core_Towers.append(tower)
    #         elif tower.deltaEta >= 0.2:
    #             self.Iso_Towers.append(tower)

    #may use Eem instead of ET
    def Central_Energy_Fraction(self, deltaR1=0.1, deltaR2=0.2):
        sE1 = 0
        sE2 = 0
        for tower in self.Towers:
            if tower.CoreRegion:
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
        for tower in self.Towers:
            if tower.CoreRegion:
                TE_C += tower.ET
        hPT = 0
        for track in self.Tracks:
            if track.CoreRegion:
                if track.PT >= hPT:
                    hPT = track.PT
        try:
            self.iF_leadtrack = TE_C/hPT
        except:
            self.iF_leadtrack = 99999999

    def Maximum_deltaR(self):
        self.max_deltaR = 0
        for track in self.Tracks:
            if track.CoreRegion:
                if track.deltaR >= self.max_deltaR:
                    self.max_deltaR = track.deltaR


#    def impactP_leadTrack(self):
#        try:
#            leadtrack = self.Core_Tracks[0]
#            for track in self.Core_Tracks:
#                if track.PT >= leadtrack.PT:
#                    leadtrack = track
#            try:
#                self.impactD0 = leadtrack.D0/leadtrack.ErrorD0
#            except:
#                self.impactD0 = 999999999
#        except:
#            self.impactD0 = 99999999


    #def TransFlightPath

    def F_IsoTracks(self):
        Iso_PT = 0
        All_PT = 0
        for track in self.Tracks:
            if track.IsoRegion:
                Iso_PT += track.PT
        for track in self.Tracks:
            All_PT += track.PT
        try:
            self.Ftrack_Iso = Iso_PT/All_PT
        except:
            self.Ftrack_Iso = 0

    #def mass_TandEMSystem(self):
     #find track masses (use jitted function)?

    #def mass_TrackSys(self):
    #similar to above



