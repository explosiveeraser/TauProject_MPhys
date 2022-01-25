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
from DataSet_Reader import Dataset
from Jet import Jet_
from Track import Track_
from Tower import Tower_

ROOT.gSystem.Load("/home/a/Delphes-3.5.0/libDelphes.so")

try:
  ROOT.gInterpreter.Declare('#include "classes/DelphesClasses.h"')
  ROOT.gInterpreter.Declare('#include "external/ExRootAnalysis/ExRootTreeReader.h"')
except:
  pass

class Background(Dataset):

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
        self._Object_Includer = ["Event", "Weight", "Jet", "Particle", "GenMissingET", "MissingET", "ScalarHT", "Track", "Tower"]
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
        for entry in trange(self._nev, desc="Background Jet (wTrack) Event Loop."):
            self._reader.ReadEntry(entry)
            weight = self._branchReader["Weight"].At(0).Weight
            evt = self._branchReader["Event"].At(0)
            num_Jets = self._branchReader["Jet"].GetEntries()
            self.Tau_Tagger.append([])
            tracks = []
            num_tracks = self._branchReader["Track"].GetEntries()
            num_towers = self._branchReader["Tower"].GetEntries()
            for jdx in range(0, num_tracks):
                track = self._branchReader["Track"].At(jdx)
                evt_track = Track_(entry, jdx, evt, track, track.Particle.GetObject())
                tracks.append(evt_track)
            for kdx in range(0, num_towers):
                tower = self._branchReader["Tower"].At(kdx)
            for idx in range(0, num_Jets):
                jet = self._branchReader["Jet"].At(idx)
                self.num_of_object["Jet"] += 1
                new_jet = Jet_(entry, idx, evt, weight, jet, jet.Particles, tracks, jet.Constituents)
                self.JetArray.append(new_jet)
                self.Fill_Histograms("Jet", jet, weight, new_jet)
                if new_jet.TruthTau:
                    self.Fill_Tau_Histograms("Jet", jet, weight, new_jet)
                for Track in new_jet.Tracks:
                    self.Fill_Histograms("Track", Track.track_obj, weight, Track)
                    if new_jet.TruthTau:
                        self.Fill_Tau_Histograms("Track", Track.track_obj, weight, Track)
                for Tower in new_jet.Towers:
                    self.Fill_Histograms("Tower", Tower.tower_obj, weight, Tower)
                    if new_jet.TruthTau:
                        self.Fill_Tau_Histograms("Tower", Tower.tower_obj, weight, Tower)
        for branch in {"GenMissingET", "MissingET", "ScalarET"}:
                if branch in list(self.Histograms.keys()):
                    num = self._branchReader[branch].GetEntriesFast()
                    for idx in range(0, num):
                        obj = self._branchReader[branch].At(idx)
                        self.Fill_Histograms(branch, obj, weight, None)
        self.Normalize_Histograms()

        def write_taucan_ttree(self, fname):
            for prong in {'1-Prong', '3-Prong'}:
                file = ROOT.TFile("NewTTrees/" + str(fname) + "_" + prong + ".root", "RECREATE")
                tree = ROOT.TTree(fname, str(fname + "_" + prong + " Tree"))
                HL = ROOT.HL_vars()
                track = ROOT.NewTrack()
                tower = ROOT.NewTower()
                tree.Branch('HL_Variables', HL,
                            'entry:index:weight:PT:Eta:Phi:deltaEta:deltaPhi:charge:NCharged:NNeutral:deltaR:f_cent:iF_leadtrack:max_deltaR:Ftrack_Iso')
                BR_track = tree.Branch('Track', track,
                                       'entry:index:P:PT:Eta:Phi:L:D0:DZ:ErrorD0:ErrorDZ:deltaEta:deltaPhi:deltaR')
                BR_tower = tree.Branch('Tower', tower,
                                       'entry:weight:E:ET:Eta:Phi:Edges:Eem:Ehad:T:deltaEta:deltaPhi:deltaR')
                for jet in tqdm(self.JetArray):
                    if jet.PT >= 20.0 and jet.Eta <= 2.5 and len(jet.Tracks) >= 1 and len(jet.Towers) > 1:
                        HL.entry = int(jet.entry)
                        HL.index = int(jet.idx)
                        # HL.event = int(jet.event)
                        HL.weight = jet.weight
                        HL.PT = jet.PT
                        HL.Eta = jet.Eta
                        HL.Phi = jet.Phi
                        HL.deltaEta = jet.deltaEta
                        HL.deltaPhi = jet.deltaPhi
                        HL.charge = jet.charge
                        HL.NCharged = jet.NCharged
                        HL.NNeutral = jet.NNeutral
                        HL.deltaR = jet.DR
                        HL.f_cent = jet.f_cent
                        HL.iF_leadtrack = jet.iF_leadtrack
                        HL.max_deltaR = jet.max_deltaR
                        #                    HL.impactD0 = jet.impactD0
                        HL.Ftrack_Iso = jet.Ftrack_Iso
                        n = len(jet.Tracks)
                        trackEntry = ROOT.std.vector("float")(n)
                        trackP = ROOT.std.vector("float")(n)
                        trackIndex = ROOT.std.vector("float")(n)
                        trackPT = ROOT.std.vector("float")(n)
                        trackEta = ROOT.std.vector("float")(n)
                        trackPhi = ROOT.std.vector("float")(n)
                        trackL = ROOT.std.vector("float")(n)
                        trackD0 = ROOT.std.vector("float")(n)
                        trackDZ = ROOT.std.vector("float")(n)
                        trackErrorD0 = ROOT.std.vector("float")(n)
                        trackErrorDZ = ROOT.std.vector("float")(n)
                        trackDeltaPhi = ROOT.std.vector("float")(n)
                        trackDeltaEta = ROOT.std.vector("float")(n)
                        trackdeltaR = ROOT.std.vector("float")(n)
                        m = len(jet.Towers)
                        towerEntry = ROOT.std.vector("float")(m)
                        towerWeight = ROOT.std.vector("float")(m)
                        towerE = ROOT.std.vector("float")(m)
                        towerET = ROOT.std.vector("float")(m)
                        towerEta = ROOT.std.vector("float")(m)
                        towerPhi = ROOT.std.vector("float")(m)
                        towerEdges0 = ROOT.std.vector("float")(m)
                        towerEdges1 = ROOT.std.vector("float")(m)
                        towerEdges2 = ROOT.std.vector("float")(m)
                        towerEdges3 = ROOT.std.vector("float")(m)
                        towerEem = ROOT.std.vector("float")(m)
                        towerEhad = ROOT.std.vector("float")(m)
                        towerT = ROOT.std.vector("float")(m)
                        towerdeltaEta = ROOT.std.vector("float")(m)
                        towerdeltaPhi = ROOT.std.vector("float")(m)
                        towerdeltaR = ROOT.std.vector("float")(m)
                        for idx in range(0, n):
                            con_track = jet.Tracks[idx]
                            trackEntry[idx] = con_track.entry
                            trackIndex[idx] = con_track.idx
                            trackP[idx] = con_track.P
                            trackPT[idx] = con_track.PT
                            trackEta[idx] = con_track.Eta
                            trackPhi[idx] = con_track.Phi
                            trackL[idx] = con_track.L
                            trackD0[idx] = con_track.D0
                            trackDZ[idx] = con_track.DZ
                            trackErrorD0[idx] = con_track.ErrorD0
                            trackErrorDZ[idx] = con_track.ErrorDZ
                            trackDeltaPhi[idx] = con_track.deltaPhi
                            trackDeltaEta[idx] = con_track.deltaEta
                            trackdeltaR[idx] = con_track.deltaR
                        track.entry = int(trackEntry)
                        track.index = int(trackIndex)
                        # track.event = int(con_track.event)
                        track.P = trackP
                        track.PT = trackPT
                        track.Eta = trackEta
                        track.Phi = trackPhi
                        track.L = trackL
                        track.D0 = trackD0
                        track.DZ = trackDZ
                        track.ErrorD0 = trackErrorD0
                        track.ErrorDZ = trackErrorDZ
                        track.deltaPhi = trackDeltaPhi
                        track.deltaEta = trackDeltaEta
                        track.deltaR = trackdeltaR
                        for jdx in range(0, m):
                            con_tower = jet.Towers[jdx]
                            towerEntry[jdx] = con_tower.entry
                            towerWeight[jdx] = con_tower.weight
                            towerE[jdx] = con_tower.E
                            towerET[jdx] = con_tower.ET
                            towerEta[jdx] = con_tower.Eta
                            towerPhi[jdx] = con_tower.Phi
                            towerEdges0[jdx] = con_tower.Edges[0]
                            towerEdges1[jdx] = con_tower.Edges[1]
                            towerEdges2[jdx] = con_tower.Edges[2]
                            towerEdges3[jdx] = con_tower.Edges[3]
                            towerEem[jdx] = con_tower.Eem
                            towerEhad[jdx] = con_tower.Ehad
                            towerT[jdx] = con_tower.T
                            towerdeltaEta[jdx] = con_tower.deltaEta
                            towerdeltaPhi[jdx] = con_tower.deltaPhi
                            towerdeltaR[jdx] = con_tower.deltaR
                        tower.entry = towerEntry
                        # tower.event = int(con_tower.event)
                        tower.weight = towerWeight
                        tower.E = towerE
                        tower.ET = towerET
                        tower.Eta = towerEta
                        tower.Phi = towerPhi
                        tower.Edges0 = towerEdges0
                        tower.Edges1 = towerEdges1
                        tower.Edges2 = towerEdges2
                        tower.Edges3 = towerEdges3
                        tower.Eem = towerEem
                        tower.Ehad = towerEhad
                        tower.T = towerT
                        tower.deltaEta = towerdeltaEta
                        tower.deltaPhi = towerdeltaPhi
                        tower.deltaR = towerdeltaR
                        tree.Fill()
                tree.Print()
                tree.Write()

