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

ROOT.gSystem.Load("install/lib/libDelphes")


class Dataset:

    def __init__(self, directory, conf_fname="Hist_Config"):
        if "/" in directory:
            self.name = directory[:-1]
        else:
            self.name = directory
            directory = directory + "/"
        self.Histograms = {}
        self.Tau_Histograms = {}
        self.chain = ROOT.TChain("Delphes")
        for f in os.listdir(directory):
            self.chain.Add(directory + f)
        self._Object_Includer = ["Event", "Weight", "Jet", "Particle", "GenMissingET", "MissingET", "ScalarHT", "Track", "Tower"]
        self._reader = ROOT.ExRootTreeReader(self.chain)
        self._branches = list(b for b in map(lambda b: b.GetName(), self.chain.GetListOfBranches()))
        for branch in self._branches:
            if branch not in self._Object_Includer:
                self.chain.SetBranchStatus(branch, status=0)
        self._leaves = dict((a, "") for a in map((lambda a: a.GetFullName()), self.chain.GetListOfLeaves()))
        for leaf in self._leaves.keys():
            temp = self.chain.FindLeaf(leaf.Data())
            self._leaves[leaf] = temp.GetTypeName()
        self._Read_Hist_Config(conf_fname)
        self.Book_Histograms()
        self._nev = self._reader.GetEntries()
        self._branchReader = {}
        self.TauJet_Container = []
        self.JetTestArray = []
        self.TJetTestArray = []
        self.num_of_object = {}
        self.Num_Taus = 0
        self.num_tau_jets = 0
        self.Tau_Tagger = []

        for branch in {"Event", "Weight", "Jet", "Particle", "GenMissingET", "MissingET", "ScalarHT"}:
            self._branchReader[branch] = self._reader.UseBranch(branch)
            self.num_of_object[branch] = 0

        print("Reading in physics objects.")
        for entry in trange(self._nev, desc="Jet Event Loop."):
            self._reader.ReadEntry(entry)
            weight = self._branchReader["Weight"].At(0).Weight
            evt = self._branchReader["Event"].At(0)
            num_Jets = self._branchReader["Jet"].GetEntriesFast()
            self.Tau_Tagger.append([])
            for idx in range(0, num_Jets):
                self.Tau_Tagger[entry].append([])
                jet = self._branchReader["Jet"].At(idx)
                self.num_of_object["Jet"] += 1
                jet_part = jet.Particles
                jet_const = jet.Constituents
                tau_jet = self.Contains_Tau(jet_part, jet_const)
                if tau_jet == True:
                    self.TauJet_Container.append([(entry, idx) ,evt, weight, jet, {"Tracks" : [], "Towers" : []}])
                    self.TJetTestArray.append([(entry, idx), evt, weight, {"Jet" : {
                        "PT" : jet.PT,
                        "DR" : math.sqrt(jet.DeltaEta**2+jet.DeltaPhi**2)
                    }, "Tracks" : {}, "Towers" : {}}])
                    self.num_tau_jets += 1
                    print("Found Tau Jet!")
                    self.Fill_Tau_Histograms("Jet", jet, weight)
                    self.Tau_Tagger[entry][idx] = True
                elif tau_jet == False:
                    self.Fill_Histograms("Jet", jet, weight)
                    self.JetTestArray.append([(entry, idx), evt, weight, {"Jet": {
                        "PT": jet.PT,
                        "DR": math.sqrt(jet.DeltaEta ** 2 + jet.DeltaPhi ** 2)
                    }, "Tracks": {}, "Towers": {}}])
                    self.Tau_Tagger[entry][idx] = False
            for branch in {"GenMissingET", "MissingET", "ScalarET"}:
                if branch in list(self.Histograms.keys()):
                    num = self._branchReader[branch].GetEntriesFast()
                    for idx in range(0, num):
                        obj = self._branchReader[branch].At(idx)
                        self.Fill_Histograms(branch, obj, weight)

        self._branchReader["Track"] = self._reader.UseBranch("Track")
        print("Reading Tracks")
        processed_tjet = 0
        for entry in trange(self._nev, desc="Track Event Loop."):
            self._reader.ReadEntry(entry)
            weight = self._branchReader["Weight"].At(0).Weight
            evt = self._branchReader["Event"].At(0)
            num_Jets = self._branchReader["Jet"].GetEntriesFast()
            for idx in range(0, num_Jets):
                jet = self._branchReader["Jet"].At(idx)
                tau_jet = False
                self.num_of_object["Jet"] += 1
                jet_const = jet.Constituents
                if self.Tau_Tagger[entry][idx] == True:
                    tau_jet = True
                    self.TauJet_Container[processed_tjet][4]["Tracks"] = jet_const
                    processed_tjet += 1
                if not jet_const.IsEmpty():
                    num_tracks = jet_const.GetEntries()
                    for jdx in range(0, num_tracks):
                        track_obj = jet_const.At(jdx)
                        if tau_jet == True:
                            self.Fill_Tau_Histograms("Track", track_obj, weight)
                            #addtestdata
                        elif tau_jet == False:
                            self.Fill_Histograms("Track", track_obj, weight)
                            #addtestdata

        self._branchReader["Tower"] = self._reader.UseBranch("Tower")
        print("Reading Towers")
        processed_tjet = 0
        for entry in trange(self._nev, desc="Tower Event Loop."):
            self._reader.ReadEntry(entry)
            weight = self._branchReader["Weight"].At(0).Weight
            evt = self._branchReader["Event"].At(0)
            num_Jets = self._branchReader["Jet"].GetEntriesFast()
            for idx in range(0, num_Jets):
                jet = self._branchReader["Jet"].At(idx)
                tau_jet = False
                self.num_of_object["Jet"] += 1
                jet_const = jet.Constituents
                if self.Tau_Tagger[entry][idx] == True:
                    tau_jet = True
                    processed_tjet += 1
                if not jet_const.IsEmpty():
                    num_tracks = jet_const.GetEntries()
                    for jdx in range(0, num_tracks):
                        tower_obj = jet_const.At(jdx)
                        cls_name = tower_obj.ClassName()
                        if cls_name == "Tower":
                            if tau_jet == True:
                                self.Fill_Tau_Histograms("Tower", tower_obj, weight)
                                # addtestdata
                            elif tau_jet == False:
                                self.Fill_Histograms("Tower", tower_obj, weight)
                                # addtestdata
        self.Normalize_Histograms()

    def Contains_Tau(self, particles, particles2):
        num1 = particles.GetEntries()
        num2 = particles2.GetEntries()
        found_tau = False
        for i in range(0, num1):
            test = particles.At(i).PID
            if abs(test) == 15:
                self.Num_Taus += 1
                found_tau = True
        if not particles2.IsEmpty():
            for i in range(0, num2):
                test = particles2.At(i).PID
                if abs(test) == 15:
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
                    [dtype, NxBins] = rest.split(";", 1)
                    NxBins = NxBins.split("\n")[0]
                    if self._HistConfig.__contains__(branch):
                        self._HistConfig[branch][leaf] = [(float(minimum), float(maximum)), str(dtype), int(NxBins)]
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
                self.Add_Histogram(branch, leaf, minimum, maximum, dtype=dtype, NxBins=NxBins)

    def Add_Histogram(self, branch, leaf, minimum, maximum, dtype="F", NxBins=128):
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

    def Fill_Histograms(self, branch, object, weight):
        if branch in self.Histograms.keys():
            for leaf in self.Histograms[branch]:
                if self.Histograms[branch][leaf] != None:
                    dtype = self._HistConfig[branch][leaf][1]
                    if "I" in dtype or "B" in dtype:
                        self.Histograms[branch][leaf].Fill(numba.types.int32(getattr(object, leaf)), weight)
                    else:
                        self.Histograms[branch][leaf].Fill(getattr(object, leaf), weight)
        else:
            pass

    def Fill_Tau_Histograms(self, branch, object, weight):
        if branch in self.Tau_Histograms.keys():
            for leaf in self.Tau_Histograms[branch]:
                if self.Tau_Histograms[branch][leaf] != None:
                    dtype = self._HistConfig[branch][leaf][1]
                    if "I" in dtype or "B" in dtype:
                        self.Tau_Histograms[branch][leaf].Fill(numba.types.int32(getattr(object, leaf)), weight)
                    else:
                        self.Tau_Histograms[branch][leaf].Fill(getattr(object, leaf), weight)
        else:
            pass

    def Normalize_Histograms(self):
        for branch in self.Histograms:
            for leaf in self.Histograms[branch]:
                if self.Histograms[branch][leaf] != None:
                    integral = self.Histograms[branch][leaf].Integral()
                    if integral != 0.:
                        self.Histograms[branch][leaf].Scale(1. / integral, "height")
                        self.Tau_Histograms[branch][leaf].Scale(1. / integral, "height")

    def print_num_of_each_object(self):
        print("--------------{}-----------------".format(self.name))
        for obj in self.num_of_object.keys():
            print("{} has {} entries.".format(obj, self.num_of_object[obj]))
        print("---------------{}----------------".format(self.name))