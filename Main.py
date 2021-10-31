import numpy as np
import pandas as pd
from array import array
import ROOT
from ROOT import gROOT

ROOT.gStyle.SetOptStat(0)
c1 = ROOT.TCanvas("c1", "Title", 900, 700)
c1.SetGrid()



myFile = ROOT.TFile.Open("1lep1tau/Data/data_A.1lep1tau.root", "READ")
myTree = myFile.mini

dFrame = ROOT.RDataFrame(myTree)

histo = dFrame.Histo1D("tau_pt")
histo.Draw()
c1.Update()

input("Press <ret> to end -> ")