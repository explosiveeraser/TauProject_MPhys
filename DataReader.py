import numpy as np
import ROOT
from ROOT import gROOT
import numba
from numba import jit, jit_module
import os, os.path

ROOT.gSystem.Load("install/lib/libDelphes")

class Event(ROOT.TNamed):
    def __init__(self, name, title):
        self.name = name
        self.title = title



class Jet(ROOT.TNamed):
    def __init__(self, jet_obj, track_ref, tower_ref):
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





class Dataset():
    def __init__(self, directory):
        if "/" in directory:
            self.name = directory[:-1]
        else:
            self.name = directory
            directory = directory + "/"
        self.chain = ROOT.TChain("Delphes")
        for f in os.listdir(directory):
            self.chain.Add(directory+f)
        self.reader = ROOT.ExRootTreeReader(self.chain)
        ###
        nev = self.reader.GetEntries()
        event = self.reader.UseBranch("Event")
        jet = self.reader.UseBranch("Jet")
        tower = self.reader.UseBranch("Tower")
        track = self.reader.UseBranch("Track")
        vertex = self.reader.UseBranch("Vertex")

