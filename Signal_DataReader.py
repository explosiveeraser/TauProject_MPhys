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
from DataSet_Reader import Dataset

ROOT.gSystem.Load("install/lib/libDelphes")

try:
    ROOT.gInterpreter.Declare('#include "classes/DelphesClasses.h"')
    ROOT.gInterpreter.Declare('#include "external/ExRootAnalysis/ExRootTreeReader.h"')
except:
    pass

class Signal(Dataset):

    def __init__(self, directory, conf_fname="Hist_Config"):
        if "/" in directory:
            self.name = directory[:-1]
        else:
            self.name = directory
            directory = directory + "/"
        self.initialise_parameters()
        self.chain = ROOT.TChain("Delphes")
        for f in os.listdir(directory):
            self.chain.Add(directory + f)
        self._Object_Includer = ["Event", "Weight", "Jet", "Particle", "GenMissingET", "MissingET", "ScalarHT", "Track",
                                 "Tower"]
        self._reader = ROOT.ExRootTreeReader(self.chain)
        self._branches = list(b for b in map(lambda b: b.GetName(), self.chain.GetListOfBranches()))
        for branch in self._branches:
            if branch not in self._Object_Includer:
                self.chain.SetBranchStatus(branch, status=0)
        self.chain.SetBranchStatus("Tower", status=1)
        self._leaves = dict((a, "") for a in map((lambda a: a.GetFullName()), self.chain.GetListOfLeaves()))
        for leaf in self._leaves.keys():
            temp = self.chain.FindLeaf(leaf.Data())
            self._leaves[leaf] = temp.GetTypeName()
        self._Read_Hist_Config(conf_fname)
        self.Book_Histograms()
        self._nev = self._reader.GetEntries()-49000
        for branch in {"Event", "Weight", "Jet", "Particle", "GenMissingET", "MissingET", "ScalarHT", "Track", "Tower"}:
            self._branchReader[branch] = self._reader.UseBranch(branch)
            self.num_of_object[branch] = 0
        self.num_of_object["Tower"] = 0
        print("Reading in physics objects.")
        for entry in trange(self._nev, desc="Background Jet (wTrack) Event Loop."):
            self._reader.ReadEntry(entry)
            weight = self._branchReader["Weight"].At(0).Weight
            evt = self._branchReader["Event"].At(0)
            num_Jets = self._branchReader["Jet"].GetEntries()
            self.Tau_Tagger.append([])
            tracks_particle = []
            track_taus = []
            num_tracks = self._branchReader["Track"].GetEntries()
            num_towers = self._branchReader["Tower"].GetEntries()
            for jdx in range(0, num_tracks):
                track_particle = self._branchReader["Track"].At(jdx).Particle.GetObject()
                tracks_particle.append((jdx, track_particle))
                track_taus.append((jdx, track_particle.PID==15 or track_particle.PID==-15))
                self.Fill_Histograms("Track", self._branchReader["Track"].At(jdx), weight)
            for kdx in range(0, num_towers):
                tower = self._branchReader["Tower"].At(kdx)
                self.Fill_Histograms("Tower", tower, weight)
            for idx in range(0, num_Jets):
                jet = self._branchReader["Jet"].At(idx)
                self.num_of_object["Jet"] += 1
                jet_const = jet.Constituents
                jet_constNum = jet_const.GetEntries()
                jet_particles = jet.Particles
                jet_pNum = jet_particles.GetEntries()
                if True:
                    self.TJetTestArray.append([(entry, idx), evt, weight, {"Jet": {
                        "Jet_Obj": jet,
                        "PT": jet.PT,
                        "DR": math.sqrt(jet.Eta ** 2 + jet.Phi ** 2),
                        "Eta": jet.Eta,
                        "Phi": jet.Phi,
                        "Tau_Check": False
                    }, "Tracks": [], "Towers": []}])
                    self.TJetTestArray[len(self.TJetTestArray)-1][3]["Jet"]["Tau_Check"]=self.Contains_Tau(jet_particles)
                    self.Fill_Tau_Histograms("Jet", jet, weight)
                    for wdx in range(0, num_tracks):
                        check_particle = tracks_particle[wdx][1]
                        part_pos = tracks_particle[wdx][0]
                        check = jet_particles.Contains(check_particle)
                        if check:
                            track = self._branchReader["Track"].At(part_pos)
                            self.Fill_Tau_Histograms("Track", track, weight)
                            self.TJetTestArray[len(self.TJetTestArray) - 1][3]["Tracks"].append(track)
                            if track_taus[wdx][1]:
                                self.num_tau_jets += 1
                                self.TJetTestArray[len(self.TJetTestArray)-1][3]["Jet"]["Tau_Check"] = True
                    for wdx in range(0, jet_constNum):
                        const = jet_const.At(wdx)
                        if const.ClassName() == "Tower":
                            self.Fill_Tau_Histograms("Tower", const, weight)
                            self.TJetTestArray[len(self.TJetTestArray) - 1][3]["Towers"].append(const)
            for branch in {"GenMissingET", "MissingET", "ScalarET"}:
                if branch in list(self.Histograms.keys()):
                    num = self._branchReader[branch].GetEntries()
                    for idx in range(0, num):
                        obj = self._branchReader[branch].At(idx)
                        self.Fill_Histograms(branch, obj, weight)
        self.Normalize_Histograms()