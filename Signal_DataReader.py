import array
import gc
import math

import numpy as np
import ROOT
import pandas as pd
from ROOT import gROOT
import numba
from array import array
from numba import jit, jit_module
import os, os.path
from tqdm import tqdm, trange
from DataSet_Reader import Dataset
from Jet import Jet_
from Track import Track_
from Tower import Tower_
from Particle import Particle_
from ROOT import addressof
import ctypes


ROOT.gSystem.Load("../Delphes-3.5.0/build/libDelphes.so")

try:
  ROOT.gInterpreter.Declare('#include "classes/DelphesClasses.h"')
  ROOT.gInterpreter.Declare('#include "external/ExRootAnalysis/ExRootTreeReader.h"')
except:
  pass

class Signal(Dataset):

    def __init__(self, directory, conf_fname="Hist_Config", print_hist=True, pile_up=False):
        if "/" in directory:
            self.name = directory[:-1]
        else:
            self.name = directory
            directory = directory + "/"
        self.initialise_parameters()
        self.chain = ROOT.TChain("Delphes")
        for f in os.listdir(directory):
            self.chain.Add(directory + f)
        self.pile_up = pile_up
        if not pile_up:
            self._Object_Includer = ["Event", "Weight", "Jet", "Particle", "GenMissingET", "MissingET", "ScalarHT", "Track",
                                     "Tower"]
        elif pile_up:
            self._Object_Includer = ["Event", "Weight", "Jet", "Particle", "GenMissingET", "MissingET", "ScalarHT",
                                     "Track",
                                     "Tower", "PileUpMix"]
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
        if not pile_up:
            for branch in {"Event", "Weight", "Jet", "Particle", "GenMissingET", "MissingET", "ScalarHT", "Track", "Tower"}:
                self._branchReader[branch] = self._reader.UseBranch(branch)
                self.num_of_object[branch] = 0
        elif pile_up:
            for branch in {"Event", "Weight", "Jet", "Particle", "GenMissingET", "MissingET", "ScalarHT", "Track", "Tower", "PileUpMix"}:
                self._branchReader[branch] = self._reader.UseBranch(branch)
                self.num_of_object[branch] = 0
        self.num_of_object["Tower"] = 0
        self.JetArray = []
        print("Reading in physics objects.")
        for entry in trange(self._nev, desc="Signal Jet (wTrack) Event Loop."):
            self._reader.ReadEntry(entry)
            #weight = self._branchReader["Weight"].At(0).Weight
            evt = self._branchReader["Event"].At(0)
            #weight = event cross section
            weight = evt.CrossSection

            #Scale cross sections to mean around 1
            if weight > 0 and not pile_up:
                weight /= 0.00321
            elif weight > 0 and pile_up:
                weight /= 0.00319

            num_Jets = self._branchReader["Jet"].GetEntries()
            self.Tau_Tagger.append([])
            tracks = []
            towers = []
            particles = []
            if pile_up:
                pileup_particles = []
            num_tracks = self._branchReader["Track"].GetEntries()
            num_towers = self._branchReader["Tower"].GetEntries()
            num_particles = self._branchReader["Particle"].GetEntries()
            if pile_up:
                num_pileup = self._branchReader["PileUpMix"].GetEntries()
            for ldx in range(0, num_particles):
                particle = self._branchReader["Particle"].At(ldx)
                evt_particle = Particle_(entry, evt, particle, self._branchReader["Particle"], particle.PID,hists=print_hist)
                particles.append(evt_particle)
            if pile_up:
                for ldx in range(0, num_particles):
                    pileup_part = self._branchReader["PileUpMix"].At(ldx)
                    evt_pileup_part = Particle_(entry, evt, pileup_part, self._branchReader["PileUpMix"], pileup_part.PID, hists=print_hist)
                    particles.append(evt_pileup_part)
            for jdx in range(0, num_tracks):
                track = self._branchReader["Track"].At(jdx)
                evt_track = Track_(entry, jdx, evt, weight, track, track.Particle.GetObject(), hists=print_hist)
                tracks.append(evt_track)
            for kdx in range(0, num_towers):
                tower = self._branchReader["Tower"].At(kdx)
                evt_tower = Tower_(entry, evt, weight, tower, hists=print_hist)
                towers.append(evt_tower)
            for idx in range(0, num_Jets):
                jet = self._branchReader["Jet"].At(idx)
                self.num_of_object["Jet"] += 1
                # new_jet = Jet_(entry, idx, evt, weight, jet, jet.Particles, particles, tracks, towers, jet.Constituents)
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
                for branch in {"GenMissingET", "MissingET", "ScalarET"}:
                    if branch in list(self.Histograms.keys()):
                        num = self._branchReader[branch].GetEntries()
                        for idx in range(0, num):
                            obj = self._branchReader[branch].At(idx)
                            self.Fill_Histograms(branch, obj, weight, None)
        if print_hist:
            self.Normalize_Histograms()

    def write_taucan_ttree(self, fname):
        for prong in {'1-Prong', '3-Prong'}:
            tot_ntr = 0
            tot_nto = 0
            name = fname + prong
            MaxNtrack = 500
            MaxNtower = 500
            jet_entry = array('i', [0])
            jet_index = array('i', [0])
            jet_cross_section = array('f', [0.])
            jet_PT = array('f', [0.])
            jet_PT_LC_scale = array('f', [0.])
            jet_f_cent = array('f', [0.])
            jet_iF_leadtrack = array('f', [0.])
            jet_max_deltaR = array('f', [0.])
            jet_Ftrack_Iso = array('f', [0.])
            jet_ratio_ToEem_P = array('f', [0.])
            jet_frac_trEM_pt = array('f', [0.])
            jet_mass_track_EM_system = array('f', [0.])
            jet_mass_track_system = array('f', [0.])
            jet_trans_impact_param_sig = array('f', [0.])
            jet_TruthTau = array('i', [0])
            jet_delphesTauTag = array('i', [0])
            nTrack = array('i', [0])
            nTower = array('i', [0])
            track_entry = array('i', MaxNtrack * [0])
            track_cross_section = array('f', MaxNtrack * [0.])
            track_PT = array('f', MaxNtrack * [0.])
            track_index = array('i', MaxNtrack * [0])
            track_D0 = array('f', MaxNtrack * [0.])
            track_DZ = array('f', MaxNtrack * [0.])
            track_deltaEta = array('f', MaxNtrack * [0.])
            track_deltaPhi = array('f', MaxNtrack * [0.])
            tower_entry = array('i', MaxNtower * [0])
            tower_cross_section = array('f', MaxNtower * [0.])
            tower_ET = array('f', MaxNtower * [0.])
            tower_Edges0 = array('f', MaxNtower * [0.])
            tower_Edges1 = array('f', MaxNtower * [0.])
            tower_Edges2 = array('f', MaxNtower * [0.])
            tower_Edges3 = array('f', MaxNtower * [0.])
            tower_deltaEta = array('f', MaxNtower * [0.])
            tower_deltaPhi = array('f', MaxNtower * [0.])
            file = ROOT.TFile("NewTTrees/" + str(fname) + "_" + prong + ".root", "RECREATE")
            tree = ROOT.TTree(fname, str(fname + "_" + prong + " Tree"))
            tree.Branch("jet_entry", jet_entry, "jet_entry/I")
            tree.Branch("jet_index", jet_index, "jet_index/I")
            tree.Branch("jet_cross_section", jet_cross_section, "jet_cross_section/F")
            tree.Branch("jet_PT", jet_PT, "jet_PT/F")
            tree.Branch("jet_PT_LC_scale", jet_PT_LC_scale, "jet_PT_LC_scale/F")
            tree.Branch("jet_f_cent", jet_f_cent, "jet_f_cent/F")
            tree.Branch("jet_iF_leadtrack", jet_iF_leadtrack, "jet_iF_leadtrack/F")
            tree.Branch("jet_max_deltaR", jet_max_deltaR, "jet_max_deltaR/F")
            tree.Branch("jet_Ftrack_Iso", jet_Ftrack_Iso, "jet_Ftrack_Iso/F")
            tree.Branch("jet_ratio_ToEem_P", jet_ratio_ToEem_P, "jet_ratio_ToEem_P/F")
            tree.Branch("jet_frac_trEM_pt", jet_frac_trEM_pt, "jet_frac_trEM_pt/F")
            tree.Branch("jet_mass_track_EM_system", jet_mass_track_EM_system, "jet_mass_track_EM_system/F")
            tree.Branch("jet_mass_track_system", jet_mass_track_system, "jet_mass_track_system/F")
            tree.Branch("jet_trans_impact_param_sig", jet_trans_impact_param_sig, 'jet_trans_impact_param_sig/F')
            tree.Branch("nTrack", nTrack, "nTrack/I")
            tree.Branch("nTower", nTower, "nTower/I")
            tree.Branch("track_entry", track_entry, "track_entry[nTrack]/I")
            tree.Branch("track_cross_section", track_cross_section, "track_cross_section[nTrack]/I")
            tree.Branch("track_index", track_index, "track_index[nTrack]/I")
            tree.Branch("track_PT", track_PT, "track_PT[nTrack]/F")
            tree.Branch("track_D0", track_D0, "track_D0[nTrack]/F")
            tree.Branch("track_DZ", track_DZ, "track_DZ[nTrack]/F")
            tree.Branch("track_deltaEta", track_deltaEta, "track_deltaEta[nTrack]/F")
            tree.Branch("track_deltaPhi", track_deltaPhi, "track_deltaPhi[nTrack]/F")
            tree.Branch("tower_entry", tower_entry, "tower_entry[nTower]/I")
            tree.Branch("tower_cross_section", tower_cross_section, "tower_cross_section[nTower]/I")
            tree.Branch("tower_ET", tower_ET, "tower_ET[nTower]/F")
            tree.Branch("tower_Edges0", tower_Edges0, "tower_Edges0[nTower]/F")
            tree.Branch("tower_Edges1", tower_Edges1, "tower_Edges1[nTower]/F")
            tree.Branch("tower_Edges2", tower_Edges2, "tower_Edges2[nTower]/F")
            tree.Branch("tower_Edges3", tower_Edges3, "tower_Edges3[nTower]/F")
            tree.Branch("tower_deltaEta", tower_deltaEta, "tower_deltaEta[nTower]/F")
            tree.Branch("tower_deltaPhi", tower_deltaPhi, "tower_deltaPhi[nTower]/F")
            tree.Branch("jet_TruthTau", jet_TruthTau, "jet_TruthTau/I")
            tree.Branch("jet_delphesTauTag", jet_delphesTauTag, "jet_delphesTauTag/I")
            num_jet_wCC = 0
            num_jet_woCC = 0
            for jet in tqdm(self.JetArray):
                if jet.PT >= 20.0 and abs(jet.Eta) <= 2.5 and (
                        abs(jet.Eta) < 1.37 or abs(jet.Eta) > 1.52) and len(jet.Tracks) >= 1 and len(
                        jet.Towers) >= 1 and jet.TruthTau[prong]:
                    num_jet_woCC += 1
                if jet.PT >= 20.0 and abs(jet.Eta) <= 2.5 and abs(jet.charge) == 1 and (abs(jet.Eta) < 1.37 or abs(jet.Eta) > 1.52) and len(jet.Tracks) >= 1 and len(
                        jet.Towers) >= 1 and jet.TruthTau[prong]:
                    num_jet_wCC += 1
                    jet_entry[0] = int(jet.entry)
                    jet_index[0] = int(jet.idx)
                    jet_cross_section[0] = jet.cross_section
                    jet_PT[0] = jet.PT
                    jet_PT_LC_scale[0] = jet.pt_lc_scale
                    jet_f_cent[0] = jet.f_cent
                    jet_iF_leadtrack[0] = jet.iF_leadtrack
                    jet_max_deltaR[0] = jet.max_deltaR
                    jet_Ftrack_Iso[0] = jet.Ftrack_Iso
                    jet_ratio_ToEem_P[0] = jet.ratio_Eem_P
                    jet_frac_trEM_pt[0] = jet.frac_trEM_jet_pt
                    jet_mass_track_EM_system[0] = jet.mass_trackplusEM
                    jet_mass_track_system[0] = jet.mass_of_system
                    jet_trans_impact_param_sig[0] = jet.max_trans_impact_param
                    jet_TruthTau[0] = jet.TruthTau[prong].__int__()
                    jet_delphesTauTag[0] = jet.delphes_TauTag.__int__()
                    n_tr = len(jet.Tracks)
                    n_to = len(jet.Towers)
                    nTrack[0] = n_tr
                    nTower[0] = n_to
                    tot_ntr += n_tr
                    tot_nto += n_to
                    for idx in range(0, n_tr):
                        con_track = jet.Tracks[idx]
                        track_entry[idx] = 3  # con_track.entry
                        track_cross_section[idx] = con_track.weight
                        track_index[idx] = con_track.idx
                        track_PT[idx] = con_track.PT
                        track_D0[idx] = con_track.D0
                        track_DZ[idx] = con_track.DZ
                        track_deltaEta[idx] = con_track.deltaEta
                        track_deltaPhi[idx] = con_track.deltaPhi
                    for jdx in range(0, n_to):
                        con_tower = jet.Towers[jdx]
                        tower_entry[jdx] = 5  # con_tower.entry
                        tower_cross_section[jdx] = con_tower.weight
                        tower_ET[jdx] = con_tower.ET
                        tower_Edges0[jdx] = con_tower.Edges[0]
                        tower_Edges1[jdx] = con_tower.Edges[1]
                        tower_Edges2[jdx] = con_tower.Edges[2]
                        tower_Edges3[jdx] = con_tower.Edges[3]
                        tower_deltaEta[jdx] = con_tower.deltaEta
                        tower_deltaPhi[jdx] = con_tower.deltaPhi
                    tree.Fill()
            print("Total number of tracks in this tree are: " + str(tot_ntr))
            print("Total number of towers in this tree are: " + str(tot_nto))
            print("Total tau jets with Charge condition: {}".format(num_jet_wCC))
            print("Total tau jets without Charge cond: {}".format(num_jet_woCC))
            tree.Print()
            tree.Write()
            file.Write()
            file.Close()

