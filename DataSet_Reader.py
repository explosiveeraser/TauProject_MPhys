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

    def print_num_of_each_object(self):
        print("--------------{}-----------------".format(self.name))
        for obj in self.num_of_object.keys():
            print("{} has {} entries.".format(obj, self.num_of_object[obj]))
        print("---------------{}----------------".format(self.name))