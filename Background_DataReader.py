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
        self._nev = self._reader.GetEntries() -49900
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
            towers = []
            particles = []
            num_tracks = self._branchReader["Track"].GetEntries()
            num_towers = self._branchReader["Tower"].GetEntries()
            num_particles = self._branchReader["Particle"].GetEntries()
            for jdx in range(0, num_tracks):
                track = self._branchReader["Track"].At(jdx)
                evt_track = Track_(entry, jdx, evt, track, track.Particle.GetObject())
                tracks.append(evt_track)
            for kdx in range(0, num_towers):
                tower = self._branchReader["Tower"].At(kdx)
                evt_tower = Tower_(entry, evt, weight, tower)
                towers.append(evt_tower)
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
        for branch in {"GenMissingET", "MissingET", "ScalarET", "Particle"}:
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
            hlvars = ROOT.HL_vars()
            track = ROOT.NewTrack()
            tower = ROOT.NewTower()
            tree.Branch("HL_vars", hlvars, 'jet_entry/F:jet_index/F:jet_weight/F:jet_PT/F:jet_Eta/F:jet_Phi/F:jet_deltaEta/F:jet_deltaPhi/F:jet_charge/F:jet_NCharged/F:jet_NNeutral/F:jet_deltaR/F:jet_f_cent/F:jet_iF_leadtrack/F:jet_Ftrack_Iso/F')
            BR_track = tree.Branch('Track', track,
                                   'nTrack/I:entry[nTrack]/F:index[nTrack]/F:P[nTrack]/F:PT[nTrack]/F:Eta[nTrack]/F:Phi[nTrack]/F:L[nTrack]/F:D0[nTrack]/F:DZ[nTrack]/F:ErrorD0[nTrack]/F:ErrorDZ[nTrack]/F:deltaEta[nTrack]/F:deltaPhi[nTrack]/F:deltaR[nTrack]/F')
            BR_tower = tree.Branch('Tower', tower,
                                   'nTower/I:entry[nTower]/F:weight[nTower]/F:E[nTower]/F:ET[nTower]/F:Eta[nTower]/F:Phi[nTower]/F:Edges0[nTower]/F:Edges1[nTower]/F:Edges2[nTower]/F:Edges3[nTower]/F:Eem[nTower]/F:Ehad[nTower]/F:T[nTower]/F:deltaEta[nTower]/F:deltaPhi[nTower]/F:deltaR[nTower]/F')
            for jet in tqdm(self.JetArray):
                if jet.PT >= 20.0 and jet.Eta <= 2.5 and len(jet.Tracks) >= 1 and len(jet.Towers) >= 1:
                    hlvars.jet_entry = jet.entry
                    hlvars.jet_index = jet.idx
                    hlvars.jet_weight = jet.weight
                    hlvars.jet_PT = jet.PT
                    hlvars.jet_Eta = jet.Eta
                    hlvars.jet_Phi = jet.Phi
                    hlvars.jet_deltaEta = jet.deltaEta
                    hlvars.jet_deltaPhi = jet.deltaPhi
                    hlvars.jet_charge = jet.charge
                    hlvars.jet_NCharged = jet.NCharged
                    hlvars.jet_NNeutral = jet.NNeutral
                    hlvars.jet_deltaR = jet.DR
                    hlvars.jet_f_cent = jet.f_cent
                    hlvars.jet_iF_leadtrack = jet.iF_leadtrack
                    hlvars.jet_max_deltaR = jet.max_deltaR
                    hlvars.jet_Ftrack_Iso = jet.Ftrack_Iso
                    n_tr = len(jet.Tracks)
                    n_to = len(jet.Towers)
                    track.nTrack = n_tr
                    tower.nTower = n_to
                    for idx in range(0, 4):
                        con_track = jet.Tracks[idx]
                        track.entry[idx] = con_track.entry
                        track.index[idx] = con_track.idx
                        track.P[idx] = con_track.P
                        track.PT[idx] = con_track.PT
                        track.Eta[idx] = con_track.Eta
                        track.Phi[idx] = con_track.Phi
                        track.L[idx] = con_track.L
                        track.D0[idx] = con_track.D0
                        track.DZ[idx] = con_track.DZ
                        track.ErrorD0[idx] = con_track.ErrorD0
                        track.ErrorDZ[idx] = con_track.ErrorDZ
                        track.deltaEta[idx] = con_track.deltaEta
                        track.deltaPhi[idx] = con_track.deltaPhi
                        track.deltaR[idx] = con_track.deltaR
                    for jdx in range(0, 4):
                        con_tower = jet.Towers[jdx]
                        tower.entry[jdx] = con_tower.entry
                        tower.weight[jdx] = con_tower.weight
                        tower.E[jdx] = con_tower.E
                        tower.ET[jdx] = con_tower.ET
                        tower.Eta[jdx] = con_tower.Eta
                        tower.Phi[jdx] = con_tower.Phi
                        tower.Edges0[jdx] = con_tower.Edges[0]
                        tower.Edges1[jdx] = con_tower.Edges[1]
                        tower.Edges2[jdx] = con_tower.Edges[2]
                        tower.Edges3[jdx] = con_tower.Edges[3]
                        tower.Eem[jdx] = con_tower.Eem
                        tower.Ehad[jdx] = con_tower.Ehad
                        tower.T[jdx] = con_tower.T
                        tower.deltaEta[jdx] = con_tower.deltaEta
                        tower.deltaPhi[jdx] = con_tower.deltaPhi
                        tower.deltaR[jdx] = con_tower.deltaR
                    tree.Fill()
            tree.Print()
            tree.Write()

