import numpy as np
import pandas as pd
from array import array
import ROOT
from ROOT import gROOT
from DataSet import DataSet

ROOT.ROOT.EnableImplicitMT()
ROOT.gStyle.SetOptStat(0)

c1 = ROOT.TCanvas("c1", "Title", 900, 700)
c1.SetGrid()



myFile = ROOT.TFile.Open("1lep1tau/MC/mc_341123.ggH125_tautaulh.1lep1tau.root", "READ")
myTree = myFile.mini

dataset = DataSet(myTree)

#ROOT.gInterpreter.Declare("""
#using VecF_t = const ROOT::RVec<float>&;
#using VecB_t = const ROOT::RVec<bool>&;
#VecF_t FindGoodTau(VecB_t x, VecF_t tau_pt) {
#    return tau_pt[x];
#
#}
#""")

#tau = dataset.DFrame.Filter("Tau_event").Histo1D('tau_pt')
tau1 = dataset.DFrame.Define("trig_tau", "tau_pt[tau_truthMatched]").Histo1D("trig_tau")
jet1 = dataset.DFrame.Filter("Tau_event").Histo1D("jet_pt")


#tau.Draw()
tau1.Draw()
jet1.Draw("SAME")
c1.Update()

input("Press <ret> to end -> ")