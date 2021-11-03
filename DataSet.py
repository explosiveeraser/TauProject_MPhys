import numpy as np
from array import array
import ROOT
from ROOT import gROOT

class DataSet():

    def __init__(self, ttree):
        initialDFrame = ROOT.RDataFrame(ttree)
        initialDFrame = initialDFrame.Define('Lepton_event', 'lep_n>0')
        initialDFrame = initialDFrame.Define('Jet_event', 'jet_n>0')
        initialDFrame = initialDFrame.Define('Photon_event', 'photon_n>0')
        initialDFrame = initialDFrame.Define('LargeR_event', 'largeRjet_n>0')
        initialDFrame = initialDFrame.Define('Tau_event', 'tau_n>0')
        self.DFrame = initialDFrame
        self.filters = {}


    def particle_Filter(self, particle):
        filter = self.DFrame.Filter(str("good"+particle))
        self.filters = {}

