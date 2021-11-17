import numpy as np
import ROOT
from ROOT import gROOT
import numba
from numba import jit, jit_module
import os, os.path
from tqdm import tqdm

ROOT.gSystem.Load("install/lib/libDelphes")

class Events(ROOT.TNamed):
    def __init__(self, name, title):
        self.name = name
        self.title = title
        self.Events = ROOT.TObjArray()

    def add_event(self, event):
        self.Events.Add(event)


class Event(ROOT.TNamed):
    def __init__(self, name, title, event_obj, weight):
        self.name = name
        self.title = title
        self.Number = event_obj.Number
        self.ReadTime = event_obj.ReadTime
        self.ProcTime = event_obj.ProcTime
        self.EvtWeight = weight.Weight
        self.Tracks = ROOT.TObjArray()
        self.Towers = ROOT.TObjArray()
        self.Jets = ROOT.TObjArray()

    def add_track(self, track):
        self.Tracks.Add(track)

    def add_tower(self, tower):
        self.Towers.Add(tower)

    def build_jet(self, jet_obj, towers, tracks):
        if towers.IsEmpty() and tracks.IsEmpty():
            return AssertionError
        elif not towers.IsEmpty() or not tracks.IsEmpty():
            self.Jets.Add(Jet(self.name, self.name, jet_obj, tracks, towers))
            return True


class Jet(ROOT.TNamed):
    def __init__(self, name, title, jet_obj, track_ref, tower_ref):
        self.name = name
        self.title = title
        self.fUniqueID = jet_obj.fUniqueID
        self.PT = jet_obj.PT
        self.Eta = jet_obj.Eta
        self.Phi = jet_obj.Phi
        self.T = jet_obj.T
        self.Mass = jet_obj.Mass
        self.DeltaEta = jet_obj.DeltaEta
        self.DeltaPhi = jet_obj.DeltaPhi
        self.Flavor = jet_obj.Flavor
        self.FlavorAlgo = jet_obj.Algo
        self.FlavorPhys = jet_obj.FlavorPhys
        self.BTag = jet_obj.BTag
        self.BTagAlgo = jet_obj.BTagAlgo
        self.BTagPhys = jet_obj.BTagPhys
        self.TauTag = jet_obj.TauTag
        self.TauWeight = jet_obj.TauWeight
        self.Charge = jet_obj.Charge
        self.EhadOverEem = jet_obj.EhadOverEem
        self.NCharged = jet_obj.NCharged
        self.NNeutrals = jet_obj.NNeutrals
        self.NeutralEnergyFraction = jet_obj.NeutralEnergyFraction
        self.ChargedEnergyFraction = jet_obj.ChargedEnergyFraction
        self.Beta = jet_obj.Beta
        self.BetaStar = jet_obj.BetaStar
        self.MeanSqDeltaR = jet_obj.MeanSqDeltaR
        self.PTD = jet_obj.PTD
        self.FracPT_total, self.FracPT = self.Set_4Vectors(jet_obj.FracPt)
        self.TSub_total, self.TSub = self.Set_4Vectors(jet_obj.Tau)
        self.TrimmedP4 = jet_obj.TrimmedP4
        self.PrunedP4 = jet_obj.PrunedP4
        self.SoftDroppedP4 = jet_obj.SoftDroppedP4
        self.NSubJetsTrimmed = jet_obj.NSubJetsTrimmed
        self.NSubJetsPruned = jet_obj.NSubJetsPruned
        self.NSubJetsSoftDropped = jet_obj.NSubJetsSoftDropped
        self.Constituent_Tracks = track_ref
        self.Constituent_Towers = tower_ref

    def Set_4Vectors(self, array):
        total = array[0]
        vector = ROOT.Math.PxPyPzE4D(array[1],array[2],array[3],array[4])
        return total, vector


class Track(ROOT.TNamed):
    def __init__(self, name, title, track_obj):
        self.name = name
        self.title = title
        self.fUniqueID = track_obj.fUniqueID
        self.PID = track_obj.PID
        self.Charge = track_obj.Charge
        self.P = track_obj.P
        self.PT = track_obj.PT
        self.Eta = track_obj.Eta
        self.Phi = track_obj.Phi
        self.CtgTheta = track_obj.CtgTheta
        self.Curvature_Inverse = track_obj.C
        self.Mass = track_obj.Mass
        self.EtaOuter = track_obj.EtaOuter
        self.PhiOuter = track_obj.PhiOuter
        self.T = track_obj.T
        self.X = track_obj.X
        self.Y = track_obj.Y
        self.Z = track_obj.Z
        self.TOuter = track_obj.TOuter
        self.XOuter = track_obj.XOuter
        self.YOuter = track_obj.YOuter
        self.ZOuter = track_obj.ZOuter
        self.Xd = track_obj.Xd
        self.Yd = track_obj.Yd
        self.Zd = track_obj.Zd
        self.L = track_obj.L
        self.D0 = track_obj.D0
        self.DZ = track_obj.DZ
        self.Nclusters = track_obj.Nclusters
        self.dNdx = track_obj.dNdx
        self.ErrorP = track_obj.ErrorP
        self.ErrorPT = track_obj.ErrorPT
        self.ErrorPhi = track_obj.ErrorPhi
        self.ErrorCtgTheta = track_obj.ErrorCtgTheta
        self.ErrorT = track_obj.ErrorT
        self.ErrorD0 = track_obj.ErrorD0
        self.ErrorDZ = track_obj.ErrorDZ
        self.ErrorC = track_obj.ErrorC
        self.ErrorD0Phi = track_obj.ErrorD0Phi
        self.ErrorD0C = track_obj.ErrorD0C
        self.ErrorD0DZ = track_obj.ErrorD0DZ
        self.ErrorD0CtgTheta = track_obj.ErrorD0CtgTheta
        self.ErrorPhiC = track_obj.ErrorPhiC
        self.ErrorPhiDZ = track_obj.ErrorPhiDZ
        self.ErrorPhiCtgTheta = track_obj.ErrorPhiCtgTheta
        self.ErrorCDZ = track_obj.ErrorCDZ
        self.ErrorCCtgTheta = track_obj.ErrorCCtgTheta
        self.ErrorDZCtgTheta = track_obj.ErrorDZCtgTheta
        self.Particle = track_obj.Particle
        self.VertexIndex = track_obj.VertexIndex


class Tower(ROOT.TNamed):
    def __init__(self, name, title, tower_obj):
        self.name = name
        self.title = title
        self.ET = tower_obj.ET
        self.Eta = tower_obj.Eta
        self.Phi = tower_obj.Phi
        self.E = tower_obj.E
        self.T = tower_obj.T
        self.NTimeHits = tower_obj.NTimeHits
        self.Eem = tower_obj.Eem
        self.Etrk = tower_obj.Etrk
        self.Edges = tower_obj.Edges


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
    def __init__(self, directory):
        if "/" in directory:
            self.name = directory[:-1]
        else:
            self.name = directory
            directory = directory + "/"
        self.Events = Events(self.name, self.name)
        self.chain = ROOT.TChain("Delphes")
        for f in os.listdir(directory):
            self.chain.Add(directory+f)
        self.reader = ROOT.ExRootTreeReader(self.chain)
        ###
        nev = self.reader.GetEntries()
        event = self.reader.UseBranch("Event")
        weight = self.reader.UseBranch("Weight")
        jet = self.reader.UseBranch("Jet")
        tower = self.reader.UseBranch("Tower")
        track = self.reader.UseBranch("Track")
        #vertex = self.reader.UseBranch("Vertex")
        print("Reading in physics objects.")
        for entry in tqdm(range(0, nev)):
            self.reader.ReadEntry(entry)
            w = weight.At(entry)
            evt = event.At(entry)
            new_event = Event(str("Event_"+entry), str("Event_"+entry), evt, w)
            for j_i in tqdm(range(0, jet.GetEntries())):
                jet_obj = jet.At(j_i)
                jet_constituents = jet_obj.Constituents
                towers = ROOT.TObjArray()
                tracks = ROOT.TObjArray()
                for c_i in tqdm(range(0, jet_constituents.GetEntries())):
                    const = jet_constituents.At(c_i)
                    if const.ClassName() == "Tower":
                        new_tower = Tower(const)
                        new_event.add_tower(new_tower)
                        towers.Add(new_tower)
                        print(str("added tower (con no:"+c_i+") to jet "+j_i))
                    elif const.ClassName() == "Track":
                        new_track = Track(const)
                        new_event.add_track(new_track)
                        tracks.Add(new_track)
                        print(str("added track (con no:" + c_i + ") to jet " + j_i))
                new_event.build_jet(jet_obj, towers, tracks)
                print(str("Completed jet: "+j_i))
            self.Events.add_event(new_event)
        print("Constructed {} dataset with {} trees and {} total events.".format(self.name, self.chain.GetNtrees(), nev))

