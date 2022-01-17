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
from Jet import Jet_
from Track import Track_
from Tower import Tower_

ROOT.gSystem.Load("../Delphes-3.5.0/libDelphes.so")

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
        self._nev = self._reader.GetEntries()
        for branch in {"Event", "Weight", "Jet", "Particle", "GenMissingET", "MissingET", "ScalarHT", "Track", "Tower"}:
            self._branchReader[branch] = self._reader.UseBranch(branch)
            self.num_of_object[branch] = 0
        self.num_of_object["Tower"] = 0
        self.JetArray = []
        print("Reading in physics objects.")
        for entry in trange(self._nev, desc="Signal Jet (wTrack) Event Loop."):
            self._reader.ReadEntry(entry)
            weight = self._branchReader["Weight"].At(0).Weight
            evt = self._branchReader["Event"].At(0)
            num_Jets = self._branchReader["Jet"].GetEntries()
            self.Tau_Tagger.append([])
            tracks = []
            track_taus = []
            num_tracks = self._branchReader["Track"].GetEntries()
            num_towers = self._branchReader["Tower"].GetEntries()
            for jdx in range(0, num_tracks):
                track = self._branchReader["Track"].At(jdx)
                evt_track = Track_(entry, jdx, evt, track, track.Particle.GetObject())
                tracks.append(evt_track)
                if evt_track.TruthTau:
                    track_taus.append(evt_track)
            for kdx in range(0, num_towers):
                tower = self._branchReader["Tower"].At(kdx)
            for idx in range(0, num_Jets):
                jet = self._branchReader["Jet"].At(idx)
                self.num_of_object["Jet"] += 1
                new_jet = Jet_(entry, idx, evt, weight, jet, jet.Particles, tracks, jet.Constituents)
                self.JetArray.append(new_jet)
                self.Fill_Histograms("Jet", jet, weight, new_jet)
                if new_jet.TruthTau == 1:
                    self.Fill_Tau_Histograms("Jet", jet, weight, new_jet)
                for Track in new_jet.Tracks:
                    self.Fill_Histograms("Track", Track.track_obj, weight, Track)
                    if new_jet.TruthTau == 1:
                        self.Fill_Tau_Histograms("Track", Track.track_obj, weight, Track)
                for Tower in new_jet.Towers:
                    self.Fill_Histograms("Tower", Tower.tower_obj, weight, Tower)
                    if new_jet.TruthTau == 1:
                        self.Fill_Tau_Histograms("Tower", Tower.tower_obj, weight, Tower)
            for branch in {"GenMissingET", "MissingET", "ScalarET"}:
                if branch in list(self.Histograms.keys()):
                    num = self._branchReader[branch].GetEntries()
                    for idx in range(0, num):
                        obj = self._branchReader[branch].At(idx)
                        self.Fill_Histograms(branch, obj, weight, None)
        self.Normalize_Histograms()