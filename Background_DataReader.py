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
from array import array
from tqdm import tqdm, trange
from DataSet_Reader import Dataset
from Jet import Jet_
from Track import Track_
from Tower import Tower_
from Particle import Particle_
import ctypes as c

ROOT.gSystem.Load("/home/a/Delphes-3.5.0/libDelphes.so")

try:
  ROOT.gInterpreter.Declare('#include "classes/DelphesClasses.h"')
  ROOT.gInterpreter.Declare('#include "external/ExRootAnalysis/ExRootTreeReader.h"')
except:
  pass

class Background(Dataset):

    def __init__(self, directory, conf_fname="Hist_Config", print_hist=True):
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
            towers = []
            particles = []
            num_tracks = self._branchReader["Track"].GetEntries()
            num_towers = self._branchReader["Tower"].GetEntries()
            num_particles = self._branchReader["Particle"].GetEntries()
            for ldx in range(0, num_particles):
                particle = self._branchReader["Particle"].At(ldx)
                evt_particle = Particle_(entry, evt, particle, self._branchReader["Particle"], hists=print_hist)
                particles.append(evt_particle)
            for jdx in range(0, num_tracks):
                track = self._branchReader["Track"].At(jdx)
                evt_track = Track_(entry, jdx, evt, track, track.Particle.GetObject(), hists=print_hist)
                tracks.append(evt_track)
            for kdx in range(0, num_towers):
                tower = self._branchReader["Tower"].At(kdx)
                evt_tower = Tower_(entry, evt, weight, tower, hists=print_hist)
                towers.append(evt_tower)
            for idx in range(0, num_Jets):
                jet = self._branchReader["Jet"].At(idx)
                self.num_of_object["Jet"] += 1
                # new_jet = Jet_(entry, idx, evt, weight, jet, jet.Particles, particles, tracks, towers, jet.Constituents, hists=print_hist)
                new_jet = Jet_(entry, idx, evt, weight, jet, None, particles, tracks, towers, None, hists=print_hist)
                self.JetArray.append(new_jet)
                if print_hist:
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
                    for Particle in new_jet.Particles:
                        self.Fill_Tau_Histograms("Particle", Particle.particle_obj, weight, Particle)
                        if Particle.PID == 15 or Particle.PID == -15:
                            self.Fill_Tau_Histograms("Particle", Particle.particle_obj, weight, Particle)
        if print_hist:
            for branch in {"GenMissingET", "MissingET", "ScalarET", "Particle"}:
                    if branch in list(self.Histograms.keys()):
                        num = self._branchReader[branch].GetEntriesFast()
                        for idx in range(0, num):
                            obj = self._branchReader[branch].At(idx)
                            self.Fill_Histograms(branch, obj, weight, None)
        if print_hist:
            self.Normalize_Histograms()

    def write_taucan_ttree(self, fname):
        for prong in {'1-Prong', '3-Prong'}:
            tot_ntr = 0
            tot_nto = 0
            name = fname+prong
            MaxNtrack = 500
            MaxNtower = 500
            jet_entry = array('i', [0])
            jet_index = array('i', [0])
            jet_weight = array('f', [0.])
            jet_PT = array('f', [0.])
            jet_Eta = array('f', [0.])
            jet_Phi = array('f', [0.])
            jet_deltaEta = array('f', [0.])
            jet_deltaPhi = array('f', [0.])
            jet_charge = array('f', [0.])
            jet_NCharged = array('f', [0.])
            jet_NNeutral = array('f', [0.])
            jet_deltaR = array('f', [0.])
            jet_f_cent = array('f', [0.])
            jet_iF_leadtrack = array('f', [0.])
            jet_max_deltaR = array('f', [0.])
            jet_Ftrack_Iso = array('f', [0.])
            jet_TruthTau = array("i", [0])
            nTrack = array('i', [0])
            nTower = array('i', [0])
            track_entry = array('i', MaxNtrack*[0])
            track_PT = array('f', MaxNtrack*[0.])
            track_Eta = array('f', MaxNtrack*[0.])
            track_index = array('i', MaxNtrack*[0])
            track_P = array('f', MaxNtrack*[0.])
            track_Phi = array('f', MaxNtrack*[0.])
            track_L = array('f', MaxNtrack*[0.])
            track_D0 = array('f', MaxNtrack*[0.])
            track_DZ = array('f', MaxNtrack*[0.])
            track_ErrorD0 = array('f', MaxNtrack*[0.])
            track_ErrorDZ = array('f', MaxNtrack*[0.])
            track_deltaEta = array('f', MaxNtrack*[0.])
            track_deltaPhi = array('f', MaxNtrack*[0.])
            track_deltaR = array('f', MaxNtrack*[0.])
            tower_entry = array('i', MaxNtower*[0])
            tower_ET = array('f', MaxNtower*[0.])
            tower_Eta = array('f', MaxNtower*[0.])
            tower_weight = array('f', MaxNtower*[0.])
            tower_E = array('f', MaxNtower*[0.])
            tower_Phi = array('f', MaxNtower*[0.])
            tower_Edges0 = array('f', MaxNtower*[0.])
            tower_Edges1 = array('f', MaxNtower*[0.])
            tower_Edges2 = array('f', MaxNtower*[0.])
            tower_Edges3 = array('f', MaxNtower*[0.])
            tower_Eem = array('f', MaxNtower*[0.])
            tower_Ehad = array('f', MaxNtower*[0.])
            tower_T = array('f', MaxNtower*[0.])
            tower_deltaEta = array('f', MaxNtower*[0.])
            tower_deltaPhi = array('f', MaxNtower*[0.])
            tower_deltaR = array('f', MaxNtower*[0.])
            file = ROOT.TFile("NewTTrees/"+str(fname)+"_"+prong+".root", "RECREATE")
            tree = ROOT.TTree(fname, str(fname+"_"+prong+" Tree"))
            tree.Branch("jet_entry", jet_entry, "jet_entry/I")
            tree.Branch("jet_index", jet_index, "jet_index/I")
            tree.Branch("jet_weight", jet_weight, "jet_weight/F")
            tree.Branch("jet_PT", jet_PT, "jet_PT/F")
            tree.Branch("jet_Eta", jet_Eta, "jet_Eta/F")
            tree.Branch("jet_Phi", jet_Phi, "jet_Phi/F")
            tree.Branch("jet_deltaEta", jet_deltaEta, "jet_deltaEta/F")
            tree.Branch("jet_deltaPhi", jet_deltaPhi, "jet_deltaPhi/F")
            tree.Branch("jet_deltaR", jet_deltaR, "jet_deltaR/F")
            tree.Branch("jet_charge", jet_charge, "jet_charge/F")
            tree.Branch("jet_NCharged", jet_NCharged, "jet_NCharged/F")
            tree.Branch("jet_NNeutral", jet_NNeutral, "jet_NNeutral/F")
            tree.Branch("jet_deltaR", jet_deltaR, "jet_deltaR/F")
            tree.Branch("jet_f_cent", jet_f_cent, "jet_f_cent/F")
            tree.Branch("jet_iF_leadtrack", jet_iF_leadtrack, "jet_iF_leadtrack/F")
            tree.Branch("jet_max_deltaR", jet_max_deltaR, "jet_max_deltaR/F")
            tree.Branch("jet_Ftrack_Iso", jet_Ftrack_Iso, "jet_Ftrack_Iso/F")
            tree.Branch("nTrack", nTrack, "nTrack/I")
            tree.Branch("nTower", nTower, "nTower/I")
            tree.Branch("track_entry", track_entry, "track_entry[nTrack]/I")
            tree.Branch("track_index", track_index, "track_index[nTrack]/I")
            tree.Branch("track_P", track_P, "track_P[nTrack]/F")
            tree.Branch("track_PT", track_PT, "track_PT[nTrack]/F")
            tree.Branch("track_Eta", track_Eta, "track_Eta[nTrack]/F")
            tree.Branch("track_Phi", track_Phi, "track_Phi[nTrack]/F")
            tree.Branch("track_L", track_L, "track_L[nTrack]/F")
            tree.Branch("track_D0", track_D0, "track_D0[nTrack]/F")
            tree.Branch("track_DZ", track_DZ, "track_DZ[nTrack]/F")
            tree.Branch("track_ErrorD0", track_ErrorD0, "track_ErrorD0[nTrack]/F")
            tree.Branch("track_ErrorDZ", track_ErrorDZ, "track_ErrorDZ[nTrack]/F")
            tree.Branch("track_deltaEta", track_deltaEta, "track_deltaEta[nTrack]/F")
            tree.Branch("track_deltaPhi", track_deltaPhi, "track_deltaPhi[nTrack]/F")
            tree.Branch("track_deltaR", track_deltaR, "track_deltaR[nTrack]/F")
            tree.Branch("tower_entry", tower_entry, "tower_entry[nTower]/I")
            tree.Branch("tower_weight", tower_weight, "tower_weight[nTower]/F")
            tree.Branch("tower_E", tower_E, "tower_E[nTower]/F")
            tree.Branch("tower_ET", tower_ET, "tower_ET[nTower]/F")
            tree.Branch("tower_Eta", tower_Eta, "tower_Eta[nTower]/F")
            tree.Branch("tower_Phi", tower_Phi, "tower_Phi[nTower]/F")
            tree.Branch("tower_Edges0", tower_Edges0, "tower_Edges0[nTower]/F")
            tree.Branch("tower_Edges1", tower_Edges1, "tower_Edges1[nTower]/F")
            tree.Branch("tower_Edges2", tower_Edges2, "tower_Edges2[nTower]/F")
            tree.Branch("tower_Edges3", tower_Edges3, "tower_Edges3[nTower]/F")
            tree.Branch("tower_Eem", tower_Eem, "tower_Eem[nTower]/F")
            tree.Branch("tower_Ehad", tower_Ehad, "tower_Ehad[nTower]/F")
            tree.Branch("tower_T", tower_T, "tower_T[nTower]/F")
            tree.Branch("tower_deltaEta", tower_deltaEta, "tower_deltaEta[nTower]/F")
            tree.Branch("tower_deltaPhi", tower_deltaPhi, "tower_deltaPhi[nTower]/F")
            tree.Branch("tower_deltaR", tower_deltaR, "tower_deltaR[nTower]/F")
            tree.Branch("jet_TruthTau", jet_TruthTau, "jet_TruthTau/I")
            for jet in tqdm(self.JetArray):
                if jet.PT >= 10.0 and abs(jet.Eta) <= 2.5 and len(jet.Tracks) >= 1 and len(jet.Towers) >= 1:
                    jet_entry[0] = int(jet.entry)
                    jet_index[0] = int(jet.idx)
                    jet_weight[0] = jet.weight
                    jet_PT[0] = jet.PT
                    jet_Eta[0] = jet.Eta
                    jet_Phi[0] = jet.Phi
                    jet_deltaEta[0] = jet.deltaEta
                    jet_deltaPhi[0] = jet.deltaPhi
                    jet_charge[0] = jet.charge
                    jet_NCharged[0] = jet.NCharged
                    jet_NNeutral[0] = jet.NNeutral
                    jet_deltaR[0] = jet.DR
                    jet_f_cent[0] = jet.f_cent
                    jet_iF_leadtrack[0] = jet.iF_leadtrack
                    jet_max_deltaR[0] = jet.max_deltaR
                    jet_iF_leadtrack[0] = jet.iF_leadtrack
                    jet_TruthTau[0] = jet.TruthTau[prong].__int__()
                    n_tr = len(jet.Tracks)
                    n_to = len(jet.Towers)
                    nTrack[0] = n_tr
                    nTower[0] = n_to
                    tot_ntr += n_tr
                    tot_nto += n_to
                    for idx in range(0, n_tr):
                        con_track = jet.Tracks[idx]
                        track_entry[idx] = 3#con_track.entry
                        track_index[idx] = con_track.idx
                        track_P[idx] = con_track.P
                        track_PT[idx] = con_track.PT
                        track_Eta[idx] = con_track.Eta
                        track_Phi[idx] = con_track.Phi
                        track_L[idx] = con_track.L
                        track_D0[idx] = con_track.D0
                        track_DZ[idx] = con_track.DZ
                        track_ErrorD0[idx] = con_track.ErrorD0
                        track_ErrorDZ[idx] = con_track.ErrorDZ
                        track_deltaEta[idx] = con_track.deltaEta
                        track_deltaPhi[idx] = con_track.deltaPhi
                        track_deltaR[idx] = con_track.deltaR
                    for jdx in range(0, n_to):
                        con_tower = jet.Towers[jdx]
                        tower_entry[jdx] = 5#con_tower.entry
                        tower_weight[jdx] = con_tower.weight
                        tower_E[jdx] = con_tower.E
                        tower_ET[jdx] = con_tower.ET
                        tower_Eta[jdx] = con_tower.Eta
                        tower_Phi[jdx] = con_tower.Phi
                        tower_Edges0[jdx] = con_tower.Edges[0]
                        tower_Edges1[jdx] = con_tower.Edges[1]
                        tower_Edges2[jdx] = con_tower.Edges[2]
                        tower_Edges3[jdx] = con_tower.Edges[3]
                        tower_Eem[jdx] = con_tower.Eem
                        tower_Ehad[jdx] = con_tower.Ehad
                        tower_T[jdx] = con_tower.T
                        tower_deltaEta[jdx] = con_tower.deltaEta
                        tower_deltaPhi[jdx] = con_tower.deltaPhi
                        tower_deltaR[jdx] = con_tower.deltaR
                    tree.Fill()
            print("Total number of tracks in this tree are: "+str(tot_ntr))
            print("Total number of towers in this tree are: " + str(tot_nto))
            tree.Print()
            tree.Write()
            file.Write()
            file.Close()