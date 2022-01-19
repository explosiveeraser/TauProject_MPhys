import array
import gc
import math

import numpy as np
import ROOT
#import pandas as pd
from ROOT import gROOT
import numba
from numba import jit, jit_module
import os, os.path
from tqdm import tqdm, trange

gROOT.ProcessLine(
"""struct HL_vars {\
    Int_t entry;\
    Int_t index;\
    Float_t weight;\
    Float_t PT;\
    Float_t Eta;\
    Float_t Phi;\
    Float_t deltaR;\
    Float_t f_cent;\
    Float_t iF_leadtrack;\
    Float_t max_deltaR;\
    Float_t impactD0;\
    Float_t Ftrack_Iso;\
};""")

gROOT.ProcessLine(
"""struct NewTrack {\
    Int_t entry;\
    Int_t index;\
    Float_t P;\
    Float_t PT;\
    Float_t Eta;\
    Float_t Phi;\
    Float_t L;\
    Float_t D0;\
    Float_t DZ;\
    Float_t ErrorD0;\
    Float_t ErrorDZ;\
    Float_t deltaEta;\
    Float_t deltaPhi;\
    Float_t deltaR;\
};""")

gROOT.ProcessLine(
"""struct NewTower {\
    Int_t entry;\
    Float_t weight;\
    Float_t E;\
    Float_t ET;\
    Float_t Eta;\
    Float_t Phi;\
    Float_t Edges[4];\
    Float_t Eem;\
    Float_t Ehad;\
    Float_t T;\
    Float_t deltaEta;\
    Float_t deltaPhi;\
    Float_t deltaR;\
};"""
)


class Dataset:

    def initialise_parameters(self):
        self.Histograms = {}
        self.Tau_Histograms = {}
        self._branchReader = {}
        self.TauJet_Container = []
        self.JetTestArray = []
        self.TJetTestArray = []
        self.num_of_object = {}
        self.Num_Taus = 0
        self.num_tau_jets = 0
        self.num_nontau_jets = 0
        self.Tau_Tagger = []

    def print_test_arrays(self, array):
        i = 0
        num_taus = 0
        for i in range(0, len(array), 25):
            jet = array[i]
            if True:
                num_taus += 1
            print("Jet Number {}-------------------".format(i))
            print("Entry : {} | IDX : {} | Weight : {} | Jet.PT : {} | Jet.DeltaR : {}".format(jet.entry, jet.idx, jet.weight, jet.PT, jet.DR))
            print("ETA: {} | PHI: {} | Truth_Tau: {} | Flavor: {}".format(jet.Eta, jet.Phi, jet.TruthTau, jet.jet_obj.Flavor))
            for track in jet.Tracks:
                DR = math.sqrt((jet.Eta - track.Eta) ** 2 + (jet.Phi - track.Phi) ** 2)
                print("Track PT : {} | Track DeltaR : {}".format(track.PT, DR))
            for tower in jet.Towers:
                DR = math.sqrt((jet.Eta - tower.Eta) ** 2 + (jet.Phi - tower.Phi) ** 2)
                print("Tower ET : {} | Tower DeltaR : {}".format(tower.ET, DR))
            print("End of Jet--------------")
        print("The total number of taus found are: {}".format(num_taus))

    def Contains_Tau(self, particles):
        num1 = particles.GetEntries()
        found_tau = False
        for i in range(0, num1):
            test = particles.At(i).PID
            if test == 15 or test == -15:
                self.Num_Taus += 1
                found_tau = True
        return found_tau

    def _Read_Hist_Config(self, fname):
        self._HistConfig = {}
        with open(fname) as file:
            for line in file:
                if line.__contains__("#"):
                    pass
                else:
                    [branch, rest] = line.split(".", 1)
                    [leaf, rest] = rest.split(":", 1)
                    [minimum, rest] = rest.split(",", 1)
                    [maximum, rest] = rest.split(";", 1)
                    [dtype, rest] = rest.split(";", 1)
                    [NxBins, Old_New] = rest.split(";", 1)
                    Old_New = Old_New.split("\n")[0]
                    if self._HistConfig.__contains__(branch):
                        self._HistConfig[branch][leaf] = [(float(minimum), float(maximum)), str(dtype), int(NxBins), int(Old_New)]
                    else:
                        self._HistConfig[branch] = {}

    def Book_Histograms(self):
        for branch in self._HistConfig.keys():
            self.Histograms[branch] = {}
            self.Tau_Histograms[branch] = {}
            for leaf in self._HistConfig[branch].keys():
                minimum = self._HistConfig[branch][leaf][0][0]
                maximum = self._HistConfig[branch][leaf][0][1]
                dtype = self._HistConfig[branch][leaf][1]
                NxBins = self._HistConfig[branch][leaf][2]
                Old_New = self._HistConfig[branch][leaf][3]
                self.Add_Histogram(branch, leaf, minimum, maximum, dtype=dtype, NxBins=NxBins, Old_New=Old_New)

    def Add_Histogram(self, branch, leaf, minimum, maximum, dtype="F", NxBins=128, Old_New=1):
        if maximum != minimum:
            if "F" in dtype:
                self.Histograms[branch][leaf] = ROOT.TH1F(branch+"."+leaf, branch+"."+leaf, NxBins, minimum, maximum)
                self.Tau_Histograms[branch][leaf] = ROOT.TH1F(branch + "." + leaf, branch + "." + leaf, NxBins, minimum,
                                                          maximum)
            elif "D" in dtype:
                self.Histograms[branch][leaf] = ROOT.TH1D(branch + "." + leaf, branch + "." + leaf, NxBins, minimum, maximum)
                self.Tau_Histograms[branch][leaf] = ROOT.TH1D(branch + "." + leaf, branch + "." + leaf, NxBins, minimum,
                                                          maximum)
            elif "I" in dtype:
                self.Histograms[branch][leaf] = ROOT.TH1D(branch + "." + leaf, branch + "." + leaf, NxBins, int(minimum), int(maximum))
                self.Tau_Histograms[branch][leaf] = ROOT.TH1D(branch + "." + leaf, branch + "." + leaf, NxBins,
                                                          int(minimum), int(maximum))
            elif "B" in dtype:
                self.Histograms[branch][leaf] = ROOT.TH1I(branch + "." + leaf, branch + "." + leaf, NxBins, int(minimum), int(maximum))
                self.Tau_Histograms[branch][leaf] = ROOT.TH1I(branch + "." + leaf, branch + "." + leaf, NxBins,
                                                          int(minimum), int(maximum))

    def Fill_Histograms(self, branch, object, weight, Def_Obj):
        if branch in self.Histograms.keys():
            for leaf in self.Histograms[branch]:
                if self.Histograms[branch][leaf] != None:
                    dtype = self._HistConfig[branch][leaf][1]
                    Old_New = self._HistConfig[branch][leaf][3]
                    if Old_New == 1:
                        if "I" in dtype or "B" in dtype:
                            self.Histograms[branch][leaf].Fill(numba.types.int32(getattr(object, leaf)), weight)
                        else:
                            self.Histograms[branch][leaf].Fill(getattr(object, leaf), weight)
                    elif Old_New == 2:
                        if "I" in dtype or "B" in dtype:
                            self.Histograms[branch][leaf].Fill(getattr(Def_Obj, leaf), weight)
                        else:
                            self.Histograms[branch][leaf].Fill(getattr(Def_Obj, leaf), weight)
        else:
            pass

    def Fill_Tau_Histograms(self, branch, object, weight, Def_Obj):
        if branch in self.Tau_Histograms.keys():
            for leaf in self.Tau_Histograms[branch]:
                if self.Tau_Histograms[branch][leaf] != None:
                    dtype = self._HistConfig[branch][leaf][1]
                    Old_New = self._HistConfig[branch][leaf][3]
                    if Old_New == 1:
                        if "I" in dtype or "B" in dtype:
                            self.Tau_Histograms[branch][leaf].Fill(numba.types.int32(getattr(object, leaf)), weight)
                        else:
                            self.Tau_Histograms[branch][leaf].Fill(getattr(object, leaf), weight)
                    elif Old_New == 2:
                        if "I" in dtype or "B" in dtype:
                            self.Tau_Histograms[branch][leaf].Fill(getattr(Def_Obj, leaf), weight)
                        else:
                            self.Tau_Histograms[branch][leaf].Fill(getattr(Def_Obj, leaf), weight)
        else:
            pass

    def Normalize_Histograms(self):
        for branch in self.Histograms:
            for leaf in self.Histograms[branch]:
                if self.Histograms[branch][leaf] != None:
                    integral = self.Histograms[branch][leaf].Integral()
                    integral2 = self.Tau_Histograms[branch][leaf].Integral()
                    print(integral)
                    print(integral2)
                    if integral != 0. and integral2 != 0.:
                        self.Histograms[branch][leaf].Scale(1. / integral, "height")
                        self.Tau_Histograms[branch][leaf].Scale(1. / integral2, "height")

    def write_taucan_ttree(self, fname):
        for prong in {'1-Prong', '3-Prong'}:
            file = ROOT.TFile("NewTTrees/"+str(fname)+"_"+prong+".root", "RECREATE")
            tree = ROOT.TTree(fname, str(fname+"_"+prong+" Tree"))
            HL = ROOT.HL_vars()
            track = ROOT.NewTrack()
            tower = ROOT.NewTower()
            tree.Branch( 'HL_Variables' , HL, 'entry:index:weight:PT:Eta:Phi:deltaR:f_cent:iF_leadtrack:max_deltaR:impactD0:Ftrack_Iso')
            BR_track = tree.Branch( 'Track', track, 'entry:index:P:PT:Eta:Phi:L:D0:DZ:ErrorD0:ErrorDZ:deltaEta:deltaPhi:deltaR')
            BR_tower = tree.Branch( 'Tower', tower, 'entry:weight:E:ET:Eta:Phi:Edges:Eem:Ehad:T:deltaEta:deltaPhi:deltaR')
            for jet in tqdm(self.JetArray):
                if jet.PT >= 10.0 and jet.Eta <= 2.5 and len(jet.Tracks) >= 3 and len(jet.Towers) > 1:
                    HL.entry = int(jet.entry)
                    HL.index = int(jet.idx)
                    #HL.event = int(jet.event)
                    HL.weight = jet.PT
                    HL.Eta = jet.Eta
                    HL.Phi = jet.Phi
                    HL.deltaR =jet.DR
                    HL.f_cent = jet.f_cent
                    HL.iF_leadtrack = jet.iF_leadtrack
                    HL.max_deltaR = jet.max_deltaR
                    HL.impactD0 = jet.impactD0
                    HL.Ftrack_Iso = jet.Ftrack_Iso
                    for con_track in jet.Tracks:
                        track.entry = int(con_track.entry)
                        track.index = int(con_track.idx)
                        #track.event = int(con_track.event)
                        track.P = con_track.P
                        track.PT = con_track.PT
                        track.Eta = con_track.Eta
                        track.Phi = con_track.Phi
                        track.L = con_track.L
                        track.D0 = con_track.D0
                        track.DZ = con_track.DZ
                        track.ErrorD0 = con_track.ErrorD0
                        track.ErrorDZ = con_track.ErrorDZ
                        track.deltaPhi = con_track.deltaPhi
                        track.deltaEta = con_track.deltaEta
                        track.deltaR = con_track.deltaR
                        BR_track.Fill()
                    for con_tower in jet.Towers:
                        tower.entry = int(con_tower.entry)
                       # tower.event = int(con_tower.event)
                        tower.weight = con_tower.weight
                        tower.E = con_tower.E
                        tower.ET = con_tower.ET
                        tower.Eta = con_tower.Eta
                        tower.Phi = con_tower.Phi
                        tower.Edges = con_tower.Edges
                        tower.Eem = con_tower.Eem
                        tower.Ehad = con_tower.Ehad
                        tower.T = con_tower.T
                        tower.deltaEta = tower.deltaEta
                        tower.deltaPhi = tower.deltaPhi
                        tower.deltaR = tower.deltaR
                        BR_tower.Fill()
                    tree.Fill()
            tree.Print()
            tree.Write()


    def print_num_of_each_object(self):
        print("--------------{}-----------------".format(self.name))
        for obj in self.num_of_object.keys():
            print("{} has {} entries.".format(obj, self.num_of_object[obj]))
        print("---------------{}----------------".format(self.name))