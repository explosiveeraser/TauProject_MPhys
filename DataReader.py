import numpy as np
import ROOT
from ROOT import gROOT
import numba
from numba import jit, jit_module
import os, os.path

ROOT.gSystem.Load("install/lib/libDelphes")

class Jet():
    def __init__(self, jet_BranchEntry, jet_reader):
        self.branchEntry = jet_BranchEntry
        self.reader = jet_reader
        self.prop_df = ROOT.RDataFrame()


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
        self.branches = self.chain.GetListOfBranches()
        self.leaves = self.chain.GetListOfLeaves()
        self.reader = ROOT.ExRootTreeReader(self.chain)
        ###
        nev = self.reader.GetEntries()
        jet = self.reader.UseBranch("Jet")
        tower = self.reader.UseBranch("Tower")
        track = self.reader.UseBranch("Track")
        for entry in range(0, nev):
            self.reader.ReadEntry(entry)
            if jet.GetEntries() > 0:
                jobj = jet.At(0)
        for const in jobj.Constituents:
            print(const.ClassName())
        ###
        self.df = {}
        for branch in self.branches:
            name = branch.GetName()
            self.df[name] = ROOT.RDataFrame(self.chain, {"Event", name})
        self.df_const = ROOT.RDataFrame(self.chain, {"Event", "Jet.Constituents"})