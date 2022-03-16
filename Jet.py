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

    def __init__(self, entry, idx, event, cross_section, mu, rho, jet_obj, particles, Event_Particles, Event_Tracks, Event_Towers, constituents, hists=True):
        self.entry = entry
        self.idx = idx
        self.event = event
        self.cross_section = cross_section
        if hists:
            self.jet_obj = jet_obj
        else:
            self.jet_obj = None
        self.delphes_TauTag = jet_obj.TauTag
        self.PT = jet_obj.PT
        self.Eta = jet_obj.Eta
        self.Phi = jet_obj.Phi
        self.deltaEta = jet_obj.DeltaEta
        self.deltaPhi = jet_obj.DeltaPhi
        self.charge = jet_obj.Charge
        self.mu = mu
        self.rho = rho
        self.NCharged = jet_obj.NCharged
        self.NNeutral = jet_obj.NNeutrals
        self.DR = math.sqrt((jet_obj.DeltaEta) **2 + (jet_obj.DeltaPhi) **2)
        self.MeanSqDR = jet_obj.MeanSqDeltaR
        self.num_particles = 0
        self.num_tracks = 0
        self.num_towers = 0
        if self.PT > 0.0 and abs(self.Eta) < 50:
            self.TauCan_1Prong = True
            self.TauCan_3Prong = True
        else:
            self.TauCan_1Prong = False
            self.TauCan_3Prong = False
        self.Tracks = []
        self.Towers = []
        self.Particles = []
        self.TruthTau = {"1-Prong" : False, "3-Prong" : False, "N>3-Prong" : False}
        self.numTaus = 0
        self._Find_Particles(Event_Particles)
        self._Find_Tracks(Event_Tracks)
        self._Find_Towers(Event_Towers)
        self.Central_Energy_Fraction()
        self.Inverse_MomFrac_LeadTrack()
        self.Maximum_deltaR()
        self.F_IsoTracks()
        self.PT_LC_scale()
        self.ratio_ToEem_P()
        self.frac_trEM_pt()
        self.mass_track_EM_system()
        self.Mass_Track_System()
        self.Trans_Impact_Param_Sign()
        # if self.numTaus > 0:
        #     print("Jet has {} Taus in it.".format(self.numTaus))

    def _Find_Particles(self, evt_particles):
        num_particles = len(evt_particles)
        for idx in range(0, num_particles):
            check_p = evt_particles[idx]
            if self.check_if_con(check_p.Eta, check_p.Phi):
                check_p.Jet_Association(self.Eta, self.Phi)
                self.Particles.append(check_p)
                self.num_particles += 1
                if check_p.PID == 15 or check_p == -15:
                    if check_p.tau_prongness == 1:
                        self.TruthTau["1-Prong"] = True
                    elif check_p.tau_prongness == 3:
                        self.TruthTau["3-Prong"] = True
                    elif check_p.tau_prongness > 3:
                        self.TruthTau["N>3-Prong"] = True
                    self.numTaus += 1

    def check_if_con(self, con_eta, con_phi):
        deltaEta = self.Eta - con_eta
        deltaPhi = self.Phi - con_phi
        deltaR = math.sqrt((deltaEta)**2+(deltaPhi)**2)
        if deltaR <= 0.4:
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

    def PT_LC_scale(self):
        pt_lc_scale = 0.
        eemsum = 0.
        ehadsum = 0.
        ptsum = 0.
        for track in self.Tracks:
            if track.CoreRegion:
                ptsum += track.PT
        for tower in self.Towers:
            eemsum += tower.Eem
            ehadsum += tower.Ehad
        self.pt_lc_scale = (ptsum - ehadsum) / eemsum if eemsum>0. else -999.0


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
        if sE2 != 0:
            self.f_cent = sE1/sE2
        else:
            self.f_cent = -1.
       # if self.f_cent > 1.:
        #    self.f_cent = 1.

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
        if hPT != 0:
            self.iF_leadtrack = TE_C/hPT
        else:
            self.iF_leadtrack = -1.
      #  if self.iF_leadtrack > 4.:
            #self.iF_leadtrack = 4.

    def Maximum_deltaR(self):
        self.max_deltaR = 0
        for track in self.Tracks:
            if track.CoreRegion:
                if track.deltaR >= self.max_deltaR:
                    self.max_deltaR = track.deltaR
        if self.max_deltaR > 0.2:
            self.max_deltaR = 0.2

    def F_IsoTracks(self):
        Iso_PT = 0
        All_PT = 0
        for track in self.Tracks:
            if track.IsoRegion:
                Iso_PT += track.PT
        for track in self.Tracks:
            All_PT += track.PT
        if All_PT != 0:
            self.Ftrack_Iso = Iso_PT/All_PT
        else:
            self.Ftrack_Iso = 10000.

    def ratio_ToEem_P(self):
        sum_Eem = 0.
        core_pt = 0.
        for tower in self.Towers:
            sum_Eem += tower.Eem
        for track in self.Tracks:
            if track.CoreRegion:
                core_pt += track.P
        if abs(core_pt) > 0.:
            self.ratio_Eem_P = sum_Eem/core_pt
        elif core_pt == 0.:
            self.ratio_Eem_P = 10000.

    def frac_trEM_pt(self):
        core_track_pt = 0.
        Eem_tower = 0.
        most_em_towers = [0., 0.]
        for tower in self.Towers:
            if tower.CoreRegion:
                if tower.Eem > most_em_towers[0]:
                    most_em_towers[0] = tower.Eem
                elif tower.Eem > most_em_towers[1]:
                    most_em_towers[1] = tower.Eem
        for towerEem in most_em_towers:
            Eem_tower += towerEem
        for track in self.Tracks:
            if track.CoreRegion:
                core_track_pt += track.PT
        self.frac_trEM_jet_pt = (core_track_pt+Eem_tower)/self.PT

    def mass_track_EM_system(self):
        core_track_mass = 0.
        Eem_tower = 0.
        most_em_towers = [0., 0.]
        for tower in self.Towers:
            if tower.CoreRegion:
                if tower.Eem > most_em_towers[0]:
                    most_em_towers[0] = tower.Eem
                elif tower.Eem > most_em_towers[1]:
                    most_em_towers[1] = tower.Eem
        for track in self.Tracks:
            if track.CoreRegion:
                core_track_mass += track.Mass
        for tower_em in most_em_towers:
            Eem_tower += tower_em
        self.mass_trackplusEM = core_track_mass + Eem_tower

    def Mass_Track_System(self):
        core_track_mass = 0.
        for track in self.Tracks:
            core_track_mass += track.P
        self.mass_of_system = core_track_mass

    def Trans_Impact_Param_Sign(self):
        self.max_trans_impact_param = 0.
        for track in self.Tracks:
            if track.CoreRegion:
                if track.D0 >= self.max_trans_impact_param:
                    self.max_trans_impact_param = track.D0
        if self.max_trans_impact_param > 0.2:
            self.max_trans_impact_param = 0.2




