import numpy as np
import ROOT
from ROOT import gROOT
import numba
from numba import jit, jit_module
import os, os.path
from tqdm import tqdm, trange

ROOT.gSystem.Load("install/lib/libDelphes")


class Events(ROOT.TNamed):
    def __init__(self, name, title):
        self.name = name
        self.title = title
        self.Events = np.array()

    def add_event(self, event):
        self.Events = np.append(self.Events, event)


class Event(ROOT.TNamed):
    def __init__(self, name, title, event_obj, weight, entry):
        self.name = name
        self.title = title
        self.Number = entry
        self.ReadTime = event_obj.ReadTime
        self.ProcTime = event_obj.ProcTime
        self.EvtWeight = weight.Weight
        self.Tracks = np.array()
        self.Towers = np.array()
        self.Jets = np.array()

    def add_track(self, track):
        self.Tracks = np.append(self.Tracks, track)

    def add_tower(self, tower):
        self.Towers = np.append(self.Towers, tower)

    def build_jet(self, jet_obj, towers, tracks):
        if towers.IsEmpty() and tracks.IsEmpty():
            return AssertionError
        elif not towers.IsEmpty() or not tracks.IsEmpty():
            self.Jets = np.append(self.Jets, Jet(self.name, self.name, jet_obj, tracks, towers))
            return True


ROOT.gInterpreter.Declare('''
ROOT::Math::PxPyPzE4D<float> set_vec(float px, float py, float pz, float E) {
    auto vec = ROOT::Math::PxPyPzE4D<float>(px, py, pz, E);
    return vec;
}
''')


class Jet(ROOT.TNamed):
    def __init__(self, name, title, jet_obj, track_ref, tower_ref):
        self.name = name
        self.title = title
        self.jet = jet_obj
        self.Constituent_Tracks = track_ref
        self.Constituent_Towers = tower_ref




class Track(ROOT.TNamed):
    def __init__(self, name, title, track_obj):
        self.name = name
        self.title = title
        self.track = track_obj


class Tower(ROOT.TNamed):
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


class Dataset():
    def __init__(self, directory, get_Histos=False):
        if "/" in directory:
            self.name = directory[:-1]
        else:
            self.name = directory
            directory = directory + "/"
        self.Events = Events(self.name, self.name)
        self.chain = ROOT.TChain("Delphes")
        for f in os.listdir(directory):
            self.chain.Add(directory + f)
        self.reader = ROOT.ExRootTreeReader(self.chain)
        branches = self.chain.GetListOfBranches()
        nev = self.reader.GetEntries() - 49000
        event = self.reader.UseBranch("Event")
        weight = self.reader.UseBranch("Weight")
        jet = self.reader.UseBranch("Jet")
        print(self.reader.GetInfo("Jet"))
        tower = self.reader.UseBranch("Tower")
        track = self.reader.UseBranch("Track")
        # vertex = self.reader.UseBranch("Vertex")
        print("Reading in physics objects.")
        for entry in trange(nev, desc="Event Loop."):
            self.reader.ReadEntry(entry)
            w = weight.At(0)
            w_e = w.Weight
            evt = event.At(0)
            new_event = Event("Event_" + str(entry), "Event_" + str(entry), evt, w, entry)
            for j_i in range(0, jet.GetEntries()):
                jet_obj = jet.At(j_i)
                jet_constituents = jet_obj.Constituents
                towers = np.array()
                tracks = np.array()
                for c_i in range(0, jet_constituents.GetEntries()):
                    const = jet_constituents.At(c_i)
                    if const.ClassName() == "Tower":
                        new_tower = Tower("Constituent_" + str(c_i), "Constituent_" + str(c_i), const)
                        new_event.add_tower(new_tower)
                        towers = np.append(towers,new_tower)
                    elif const.ClassName() == "Track":
                        new_track = Track("Constituent_" + str(c_i), "Constituent_" + str(c_i), const)
                        new_event.add_track(new_track)
                        tracks = np.append(tracks, new_track)
                new_event.build_jet(jet_obj, towers, tracks)
            self.Events.add_event(new_event)
        print(
            "Constructed {} dataset with {} trees and {} total events.".format(self.name, self.chain.GetNtrees(), nev))
        if get_Histos:
            self.Histos = {}
            for cls in ["Jet", "Tower", "Track"]:
                self.build_histogram(cls)


