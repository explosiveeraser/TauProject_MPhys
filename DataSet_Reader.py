import array

import numpy as np
import ROOT
from ROOT import gROOT
import numba
from numba import jit, jit_module
import os, os.path
from tqdm import tqdm, trange

ROOT.gSystem.Load("install/lib/libDelphes")


class Events:
    def __init__(self, name, title):
        self.name = name
        self.title = title
        self.Events = np.array()

    def add_event(self, event):
        self.Events = np.append(self.Events, event)

    def get_jets(self):
        return self.Events[:].Jets


class Event:
    def __init__(self, name, title, event_obj, weight, entry):
        self.name = name
        self.title = title
        self.Event = event_obj
        self.weight = weight
        self.entry = entry
        self.Tracks = np.array()
        self.Towers = np.array()
        self.Jets = np.array()

    def add_track(self, track):
        self.Tracks = np.append(self.Tracks, track)

    def add_tower(self, tower):
        self.Towers = np.append(self.Towers, tower)

    def build_jet(self, jet_obj, towers, tracks):
        if towers.IsEmpty() and tracks.IsEmpty():
            raise AssertionError("Assertion error")
        elif not towers.IsEmpty() or not tracks.IsEmpty():
            self.Jets = np.append(self.Jets, Jet(self.name, self.name, jet_obj, tracks, towers))
            return True

    def __array__(self):
        return [self.Event, self.weight, self.Jets, self.Tracks, self.Tracks]


ROOT.gInterpreter.Declare('''
ROOT::Math::PxPyPzE4D<float> set_vec(float px, float py, float pz, float E) {
    auto vec = ROOT::Math::PxPyPzE4D<float>(px, py, pz, E);
    return vec;
}
''')


class Jet:
    def __init__(self, name, title, jet_obj, track_ref, tower_ref):
        self.name = name
        self.title = title
        self.jet = jet_obj
        self.Constituent_Tracks = track_ref
        self.Constituent_Towers = tower_ref

    def is_tau_tagged(self):
        return self.jet.TauTag

class Track:
    def __init__(self, name, title, track_obj):
        self.name = name
        self.title = title
        self.track = track_obj


class Tower:
    def __init__(self, name, title, tower_obj):
        self.name = name
        self.title = title
        self.tower = tower_obj


"""class Vertex(ROOT.TNamed):
    def __init__(self, name, title, v_obj):
        self.name = name
        self.title = title
        self.T = v_obj.T
        self.X = v_obj.X
        self.Y = v_obj.Y
        self.Z = v_obj.Z
        self.ErrorT = v_obj.ErrorT
        self.ErrorX = v_obj.ErrorX
        self.ErrorY = v_obj.ErrorY
        self.ErrorZ = v_obj.ErrorZ
        self.Index = v_obj.Index
        self.NDF = v_obj.NDF
        self.Sigma = v_obj.Sigma
        self.SumPT2 = v_obj.SumPT2
        self.GenDeltaZ = v_obj.GenDeltaZ
        self.BTVSumPT2 = v_obj.BTVSumPT2

"""


class Dataset:
    def __init__(self, directory, get_Histos=False):
        if "/" in directory:
            self.name = directory[:-1]
        else:
            self.name = directory
            directory = directory + "/"
        self.DataSet = []
        self.Events = []
        self.Histograms = {}
        self.chain = ROOT.TChain("Delphes")
        for f in os.listdir(directory):
            self.chain.Add(directory + f)
        self._branches = list(b for b in map(lambda b: b.GetName(), self.chain.GetListOfBranches()))
        self._leaves = dict((a, "") for a in map((lambda  a: a.GetFullName()), self.chain.GetListOfLeaves()))
        for leaf in self._leaves.keys():
            temp = self.chain.FindLeaf(leaf.Data())
            self._leaves[leaf] = temp.GetTypeName()
        hist_incl = []
        for i in {"Track." , "Tower.", "EFlowTrack.", "GenJet.", "GenMissingET.", "Jet.", "MissingET.", "ScalarHT."}:
            hist_incl.append(ROOT.TString(i))
            self.Histograms[i[:-1]] = {}
        for leaf in tqdm(self._leaves.keys()):
            for incl in hist_incl:
                if leaf.Contains(incl) and (not leaf.Contains(ROOT.TString("fUniqueID")) and not leaf.Contains(ROOT.TString("fBits"))):
                    if ("Float" in self._leaves[leaf] or "Int" in self._leaves[leaf]) and "_size" not in self._leaves[leaf]:
                        leaf_obj = self.chain.FindLeaf(leaf.Data())
                        self.Add_Histogram(leaf_obj)
        self._reader = ROOT.ExRootTreeReader(self.chain)
        self._nev = self._reader.GetEntries() - 49000
        self._branchReader = {}
        self.Physics_ObjectArrays = {}
        for branch in self._branches:
            self._branchReader[branch] = self._reader.UseBranch(branch)
            self.Physics_ObjectArrays[branch] = []
        print("Reading in physics objects.")
#        excluder = ["LHCOEvent", "LHEFEvent", "LHEFWeight", "HepMCEvent", "Photon", "Electron", "Muon"]
        includer = ["Event", "Weight", "Particle", "Track", "Tower", "EFlowTrack", "GenJet", "GenMissingET", "Jet", "MissingET", "ScalarHT"]
        for entry in trange(self._nev, desc="Event Loop."):
            self._reader.ReadEntry(entry)
            w = self._branchReader["Weight"].At(0).Weight
            evt = self._branchReader["Event"].At(0)
            self.Events.append([entry, evt, w])
            for branch_name in includer:
                branch = self._branchReader[branch_name]
                length = branch.GetEntriesFast()
                for idx in range(0, length):
                    object = branch.At(idx)
                    self.Physics_ObjectArrays[branch_name].append(object)
                    if branch_name in self.Histograms.keys():
                        self.Fill_Histograms(object)
                    del object
                del branch
                del length
            del w
            del evt


    def Add_Histogram(self, object):
        [branch, leaf] = object.GetFullName().Data().split(".")
        minimum = object.GetMinimum()
        maximum = object.GetMaximum()
        self.Histograms[branch][leaf] = ROOT.TH1F(branch+"."+leaf, branch+"."+leaf, 128, minimum, maximum)

    def Fill_Histograms(self, object):
        branch = object.ClassName()
        if branch in self.Histograms.keys():
            for leaf in self.Histograms[branch]:
                try:
                    self.Histograms[branch][leaf].Fill(getattr(object, leaf))
                except:
                    self.Histograms[branch][leaf] = None
        else:
            pass



