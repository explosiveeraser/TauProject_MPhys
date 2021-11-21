import inspect
from inspect import getargspec
import numpy as np
#import DataAnalysis
import ROOT
from collections import Iterable
import numba
import os, os.path
from tqdm import tqdm, trange


ROOT.gSystem.Load("install/lib/libDelphes")



class DelphesDataFrame(ROOT.RDataFrame):

    def __init__(self, directory):
        self.chain = ROOT.TChain("Delphes", "Delphes")
        input_files = []
        if "/" in directory:
            self.name = directory[:-1]
        else:
            self.name = directory
            directory = directory + "/"
        for f in os.listdir(directory):
            self.chain.Add(directory+f)
        ROOT.RDataFrame.__init__(self, self.chain)
        self.reader = ROOT.ExRootTreeReader(self.chain)
        self.nev = self.reader.GetEntries() - 49000
        jets = self.reader.UseBranch("Jet")
        tower = self.reader.UseBranch("Tower")
        track = self.reader.UseBranch("Track")
        events = []
        for entry in trange(self.nev, desc="Event Loop:"):
            self.reader.ReadEntry(entry)
            for idx in range(0, jets.GetEntriesFast()):
                jet_tracksArray = []
                jet_towersArray = []
                jetsArray = []
                jet_object = jets.At(idx)
                jet_consts = jet_object.Constituents
                for cdx in range(0, jet_consts.GetEntriesFast()):
                    const = jet_consts.At(cdx)
                    if const.ClassName() == "Tower":
                        jet_towersArray.append(const)
                    elif const.ClassName() == "Track":
                        jet_tracksArray.append(const)
                jetsArray.append([jet_object, jet_tracksArray, jet_towersArray])
                events.append([entry, jet_object, jet_tracksArray, jet_towersArray])
        self.Events = np.array(events)
        events_transpose = self.Events.T
        print(events_transpose)
        self.JetDF = ROOT.RDF.MakeNumpyDataFrame({"entry": events_transpose[0], "Jet": events_transpose[1], "Jet_Tracks": events_transpose[2], "Jet_Towers": events_transpose[3]})

