import array
import gc

import numpy as np
import ROOT
import pandas as pd
from ROOT import gROOT
import numba
from numba import jit, jit_module
import os, os.path
from tqdm import tqdm, trange

ROOT.gSystem.Load("install/lib/libDelphes")

"""
Histogram Varaibles to Include including PU variables
{ "T","X","Y","Z","ErrorT", "ErrorX","ErrorY","ErrorZ", 
  "Sigma", "SumPT2", "GenDeltaZ","BTVSumPT2","ET","Eta","Phi","E","T"
    ,"NTimeHits", "Eem","Ehad","Charge","P","PT","Eta","Phi","CtgTheta"
  , "C", "Mass", "EtaOuter","PhiOuter", "T","X","Y","Z","TOuter","XOuter",
  "YOuter","ZOuter","Xd","Yd","Zd","L","D0","DZ","Nclusters","dNdx","ErrorP",
  "ErrorPT","ErrorPhi","ErrorCtgTheta","ErrorT","ErrorD0","ErrorDZ","ErrorC",
  "ErrorD0Phi","ErrorD0C","ErrorD0DZ","ErrorD0CtgTheta","ErrorPhiC","ErrorPhiDZ",
  "ErrorPhiCtgTheta","ErrorCDZ","ErrorCCtgTheta","ErrorDZCt","PT",
  "Eta","Phi","T","Mass","DeltaEta","DeltaPhi","Flavor","FlavorAlgo",
  "FlavorPhys","BTag","BTagAlgo","BTagPhys","TauTag","TauWeight","Charge",
  "EhadOverEem","NCharged","NNeutrals","NeutralEnergyFraction","ChargedEnergyFraction",
  "Beta","BetaStar","MeanSqDeltaR","PTD"
}

excluding PU and empties and errors
{
"T","X","Y","Z", "Sigma", "SumPT2", "GenDeltaZ","BTVSumPT2","ET","Eta","Phi","E","T"
    ,"NTimeHits", "Eem","Ehad","Charge","P","PT","Eta","Phi","CtgTheta"
  , "C", "Mass", "EtaOuter","PhiOuter", "T","X","Y","Z","TOuter","XOuter",
  "YOuter","ZOuter","Xd","Yd","Zd","L","D0","DZ","Nclusters","dNdx","PT",
  "Eta","Phi","T","Mass","DeltaEta","DeltaPhi","TauTag","TauWeight","Charge",
  "EhadOverEem","NCharged","NNeutrals","NeutralEnergyFraction","ChargedEnergyFraction",
  "MeanSqDeltaR","PTD"
}
"""

class Dataset:

    fAllHists = {
"T","X","Y","Z", "Sigma", "SumPT2", "GenDeltaZ","BTVSumPT2","ET","Eta","Phi","E","T"
    ,"NTimeHits", "Eem","Ehad","Charge","P","PT","Eta","Phi","CtgTheta"
  , "C", "Mass", "EtaOuter","PhiOuter", "T","X","Y","Z","TOuter","XOuter",
  "YOuter","ZOuter","Xd","Yd","Zd","L","D0","DZ","Nclusters","dNdx","PT",
  "Eta","Phi","T","Mass","DeltaEta","DeltaPhi","TauTag","TauWeight","Charge",
  "EhadOverEem","NCharged","NNeutrals","NeutralEnergyFraction","ChargedEnergyFraction",
  "MeanSqDeltaR","PTD"
}
    def __init__(self, directory, Histo_VarIncl = fAllHists):
        if "/" in directory:
            self.name = directory[:-1]
        else:
            self.name = directory
            directory = directory + "/"
        self.DataSet = []
        self.Events = []
        self.Histograms = {}
        self.Properties_Array = pd.DataFrame()
        self._filler_dict = {}
        self._num_of_prop = {}
        self.chain = ROOT.TChain("Delphes")
        for f in os.listdir(directory):
            self.chain.Add(directory + f)
        self._branches = list(b for b in map(lambda b: b.GetName(), self.chain.GetListOfBranches()))
        self._leaves = dict((a, "") for a in map((lambda a: a.GetFullName()), self.chain.GetListOfLeaves()))
        for leaf in self._leaves.keys():
            temp = self.chain.FindLeaf(leaf.Data())
            self._leaves[leaf] = temp.GetTypeName()
        hist_incl = []
        for i in {"Track." , "Tower.", "EFlowTrack.", "GenJet.", "GenMissingET.", "Jet.", "MissingET.", "ScalarHT."}:
            for leaf in Histo_VarIncl:
                hist_incl.append(ROOT.TString(i+leaf))
            self.Histograms[i[:-1]] = {}
            self._filler_dict[i[:-1]] = {}
            self._num_of_prop[i[:-1]] = {}
        self._reader = ROOT.ExRootTreeReader(self.chain)
        for leaf in tqdm(self._leaves.keys()):
            if leaf in hist_incl:
                if ("Float" in self._leaves[leaf] or "Int" in self._leaves[leaf]) or "Bool" in self._leaves[leaf]:
                    name = leaf.Data().split(".")
                    leaf_obj = self.chain.FindLeaf(leaf.Data())
                    self.Add_Histogram(leaf_obj)
        self._nev = self._reader.GetEntries()
        self._branchReader = {}
        self.Physics_ObjectArrays = {}
        self.num_of_object = {}
        for branch in self._branches:
            self._branchReader[branch] = self._reader.UseBranch(branch)
            self.Physics_ObjectArrays[branch] = []
        print("Reading in physics objects.")
#        excluder = ["LHCOEvent", "LHEFEvent", "LHEFWeight", "HepMCEvent", "Photon", "Electron", "Muon"]
        includer = ["Event", "Weight", "Particle", "Track", "Tower", "EFlowTrack", "GenJet", "GenMissingET", "Jet", "MissingET", "ScalarHT"]
        self.update_interval = 841
        for incl in includer:
            self.num_of_object[incl] = 0
        df_update = False
        for entry in trange(self._nev, desc="Event Loop."):
            self._reader.ReadEntry(entry)
            w = self._branchReader["Weight"].At(0).Weight
            evt = self._branchReader["Event"].At(0)
            self.Events.append([entry, evt, w])
            for branch_name in includer:
                branch = self._branchReader[branch_name]
                length = branch.GetEntries()
                self.num_of_object[branch_name] += length
                for idx in range(0, length):
                    object = branch.At(idx)
                    self.Physics_ObjectArrays[branch_name].append(object)
                    if branch_name in list(self.Histograms.keys()):
                        self.Fill_Histograms(branch_name, object, df_update, entry)
                    del object
                del branch
                del length
            del w
            del evt
        self.Normalize_Histograms()


    def Add_Histogram(self, object, maximum=0, minimum=0):
        [branch, leaf] = object.GetFullName().Data().split(".")
        self.Histograms[branch][leaf] = ROOT.TH1F(branch+"."+leaf, branch+"."+leaf, 128, minimum, maximum)
        self.Histograms[branch][leaf].SetBit(ROOT.TH1.kXaxis)
        self._filler_dict[branch][leaf] = []
        self._num_of_prop[branch][leaf] = 0

    def Fill_Histograms(self, branch, object, update_df, entry):
        if branch in self.Histograms.keys():
            for leaf in self.Histograms[branch]:
                if self.Histograms[branch][leaf] != None:
                    if self.Histograms[branch][leaf].GetEntries() == 0:
                        maximum = getattr(object, leaf)
                        minimum = getattr(object, leaf) - maximum*0.00005
                        self.Histograms[branch][leaf].SetMaximum(maximum)
                        self.Histograms[branch][leaf].SetMinimum(minimum)
                    try:
                        self.Histograms[branch][leaf].Fill(getattr(object, leaf))
                        self._num_of_prop[branch][leaf] += 1
                    except:
                        self.Histograms[branch][leaf] = None
                        print("{}.{}".format(branch, leaf))
        else:
            pass

    def Normalize_Histograms(self):
        for branch in self.Histograms:
            for leaf in self.Histograms[branch]:
                if self.Histograms[branch][leaf] != None:
                    integral = self.Histograms[branch][leaf].Integral()
                    if integral != 0.:
                        self.Histograms[branch][leaf].Scale(1. / integral, "height")

    def print_num_of_each_object(self):
        print("--------------{}-----------------".format(self.name))
        for obj in self.num_of_object.keys():
            print("{} has {} entries.".format(obj, self.num_of_object[obj]))
        print("---------------{}----------------".format(self.name))