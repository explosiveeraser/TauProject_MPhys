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


class Dataset:

    def __init__(self, directory, conf_fname="Hist_Config"):
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
        self._Object_Includer = ["Event", "Weight", "Particle", "GenMissingET", "MissingET", "ScalarHT", "GenJet",
                                 "Jet", "EFlowTrack","EFlowNeutralHadron", "EFlowPhoton", "Tower", "Track"]
        self._reader = ROOT.ExRootTreeReader(self.chain)
        self._branches = list(b for b in map(lambda b: b.GetName(), self.chain.GetListOfBranches()))
        for branch in self._branches:
            if branch not in self._Object_Includer:
                self.chain.SetBranchStatus(branch, status=0)
        self._leaves = dict((a, "") for a in map((lambda a: a.GetFullName()), self.chain.GetListOfLeaves()))
        for leaf in self._leaves.keys():
            temp = self.chain.FindLeaf(leaf.Data())
            self._leaves[leaf] = temp.GetTypeName()
        self._Read_Hist_Config(conf_fname)
        for branch in self._HistConfig.keys():
            self.Histograms[branch] = {}
            for leaf in self._HistConfig[branch].keys():
                minimum = self._HistConfig[branch][leaf][0][0]
                maximum = self._HistConfig[branch][leaf][0][1]
                dtype = self._HistConfig[branch][leaf][1]
                NxBins = self._HistConfig[branch][leaf][2]
                self.Add_Histogram(branch, leaf, minimum, maximum, dtype=dtype, NxBins=NxBins)
        self._nev = self._reader.GetEntries()
        self._branchReader = {}
        self.Physics_ObjectArrays = {}
        self.num_of_object = {}
        for branch in self._branches:
            if branch in self._Object_Includer:
                self._branchReader[branch] = self._reader.UseBranch(branch)
                self.Physics_ObjectArrays[branch] = []
                self.num_of_object[branch] = 0
        print("Reading in physics objects.")
        Jet_File_Name = self.name + "_Jet_Const.txt"
        Jet_File_Name2 = self.name + "_Jet_Particles.txt"
        try:
            os.remove(Jet_File_Name)
        except:
            pass
        try:
            os.remove(Jet_File_Name2)
        except:
            pass
        Jet_file = open(Jet_File_Name, "a")
        Jet_file2 = open(Jet_File_Name2, "a")
        for entry in trange(self._nev, desc="Event Loop."):
            self._reader.ReadEntry(entry)
            w = self._branchReader["Weight"].At(0).Weight
            evt = self._branchReader["Event"].At(0)
            self.Events.append([entry, evt, w])
            for branch_name in self._Object_Includer:
                if "Jet" in branch_name:
                    branch = self._branchReader[branch_name]
                    length = branch.GetEntries()
                    self.num_of_object[branch_name] += length
                    for idx in range(0, length):
                        object = branch.At(idx)
                        particles = object.Particles
                        const = object.Constituents
                        contains_tau = False
                        Jet_file2.write("{} {}.{} Contains: ".format(branch_name,entry, idx))
                        Jet_file.write("{} {}.{} Contains: ".format(branch_name,entry, idx))
                        first = True
                        for p in range(0, particles.GetEntries()):
                            p_obj = particles.At(p)
                            try:
                                Jet_file2.write(str(p_obj.PID) + ", ")
                                for c in range(0, const.GetEntries()):
                                    c_obj = const.At(c)
                                    if first == True:
                                        Jet_file.write(c_obj.GetName() + ", ")
                                    try:
                                        c_part = c_obj.Particles()
                                    except:
                                        c_part = 0
                                    if c_part != 0:
                                        for c_p in range(0, c_part.GetEntries()):
                                            cp_obj = c_part.At(c_p)
                                            if p_obj.PID == 15 or p_obj == -15 or cp_obj == 15 or cp_obj == -15:
                                                contains_tau = True
                                    else:
                                        if p_obj.PID == 15 or p_obj == -15:
                                            contains_tau = True
                                if first == True:
                                    first = False
                            except:
                                pass
                        Jet_file.write("\n")
                        Jet_file2.write("\n")
                        if contains_tau == True:
                            #Jet_file2.write("Jet {}.{} Contains: ".format(entry, idx))
                            #Jet_file.write("Jet {}.{} Contains: ".format(entry, idx))
                            #for ldx in range(0, particles.GetEntries()):
                             #   particle = particles.At(ldx)
                              #  Jet_file2.write(str(particle.PID)+ ", ")
                            #Jet_file2.write("\n")
                            self.Fill_Histograms(branch_name, object, w)
                            #self.Physics_ObjectArrays[branch_name].append(object)
                            const = object.Constituents
                            for jdx in range(0, const.GetEntries()):
                                const_obj = const.At(jdx)
                                cls_name = const_obj.ClassName()
                             #   Jet_file.write(cls_name + ", ")
                                if cls_name in list(self.Histograms.keys()):
                                    self.Fill_Histograms(cls_name, const_obj, w)
                            #Jet_file.write("\n")
        Jet_file.close()
        Jet_file2.close()
        self.Normalize_Histograms()

    def _Read_Hist_Config(self, fname):
        self._HistConfig = {}
        with open(fname) as file:
            for line in file:
                if line.__contains__("#"):
                    pass
                else:
                    [branch, rest] = line.split(".", 1)
                    [leaf, rest] = rest.split(":", 1)
                    [minimum, rest] = rest.split(",", 1)
                    [maximum, rest] = rest.split(";", 1)
                    [dtype, NxBins] = rest.split(";", 1)
                    NxBins = NxBins.split("\n")[0]
                    if self._HistConfig.__contains__(branch):
                        self._HistConfig[branch][leaf] = [(float(minimum), float(maximum)), str(dtype), int(NxBins)]
                    else:
                        self._HistConfig[branch] = {}

    def Add_Histogram(self, branch, leaf, minimum, maximum, dtype="F", NxBins=128):
        if maximum != minimum:
            if "F" in dtype:
                self.Histograms[branch][leaf] = ROOT.TH1F(branch+"."+leaf, branch+"."+leaf, NxBins, minimum, maximum)
            elif "D" in dtype:
                self.Histograms[branch][leaf] = ROOT.TH1D(branch + "." + leaf, branch + "." + leaf, NxBins, minimum, maximum)
            elif "I" in dtype:
                self.Histograms[branch][leaf] = ROOT.TH1D(branch + "." + leaf, branch + "." + leaf, NxBins, int(minimum), int(maximum))
            elif "B" in dtype:
                self.Histograms[branch][leaf] = ROOT.TH1I(branch + "." + leaf, branch + "." + leaf, NxBins, int(minimum), int(maximum))

    def Fill_Histograms(self, branch, object, weight):
        if branch in self.Histograms.keys():
            for leaf in self.Histograms[branch]:
                if self.Histograms[branch][leaf] != None:
                    dtype = self._HistConfig[branch][leaf][1]
                    if "I" in dtype or "B" in dtype:
                        self.Histograms[branch][leaf].Fill(numba.types.int32(getattr(object, leaf)), weight)
                    else:
                        self.Histograms[branch][leaf].Fill(getattr(object, leaf), weight)
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