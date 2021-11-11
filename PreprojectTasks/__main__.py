import numpy as np
from array import array
import ROOT
from ROOT import gROOT

ROOT.ROOT.EnableImplicitMT()
ROOT.gStyle.SetOptStat(0)

data_path = "1lep1tau/Data/"
MC_path = "1lep1tau/MC/"
TFile_name = "data_A.1lep1tau.root"

mc_file = ROOT.TFile.Open(str(data_path+TFile_name), "READ")
mc_Tree = mc_file.mini

mc_DFrame = ROOT.RDataFrame(mc_Tree)

#Add weights
mc_DFrame = mc_DFrame.Define("weight", "scaleFactor_PILEUP * scaleFactor_TAU * mcWeight")

#Filter out everything not a jet or tau
mc_DFrame = mc_DFrame.Filter("jet_n > 0 || tau_n > 0")

#Define Events with "Good" jets and "Good" taus
goodJetCrit = "(jet_n > 0) && jet_pt > 10 && abs(jet_eta) < 2.5"
goodTauCrit = "(tau_n > 0) && ((abs(tau_eta) < 1.37) || (abs(tau_eta) > 1.52)) && (tau_pt > 20)"

mc_DFrame = mc_DFrame.Define("goodJet", goodJetCrit)
mc_DFrame = mc_DFrame.Define("goodTau", goodTauCrit)

"""
Make histograms
"""

histos = {}

#jet histos
for h in ["jet_pt", "jet_E"]:
    histos[h] = mc_DFrame.Histo1D(ROOT.RDF.TH1DModel(h, h, 128, np.zeros(129, float)), h)

#jet and tau abs(eta) histos
for h in ["jet_eta", "tau_eta"]:
    title = str("a"+h)
    func = str("abs("+h+")")
    histos[title] = mc_DFrame.Define(title, func).Histo1D(ROOT.RDF.TH1DModel(title, title, 128, np.zeros(129, float)), title)

histos["tau_n"] = mc_DFrame.Histo1D(ROOT.RDF.TH1DModel("tau_n", "tau_n", 128, 0, 2), "tau_n")

histos["tau_n"].Print()

#rest of tau histos
for h in ["tau_pt", "tau_phi", "tau_E", "tau_charge", "tau_nTracks",
          "tau_BDTid", "ditau_m"]:
    histos[h] = mc_DFrame.Histo1D(ROOT.RDF.TH1DModel(h, h, 128, np.zeros(129, float)), h)

#Other histos
mc_DFrame = mc_DFrame.Define("one_tau", "tau_nTracks[tau_nTracks == 1]")
mc_DFrame = mc_DFrame.Define("three_tau", "tau_nTracks[tau_nTracks == 3]")
mc_DFrame = mc_DFrame.Define("o_tau", "tau_nTracks[tau_nTracks > 3]")
histos["1_tau"] = mc_DFrame.Histo1D("one_tau")
histos["3_tau"] = mc_DFrame.Histo1D("three_tau")
histos["o_tau"] = mc_DFrame.Histo1D("o_tau")
##



#Draw histograms
c1 = ROOT.TCanvas("c1", "Histograms", 1200, 750)

#
upperpad = ROOT.TPad("upperpad", "Jet Histograms", 0.0, 0.75, 1.0, 1.0)
upperpad.Divide(3,1)
upperpad.Draw()

#
lowerpad = ROOT.TPad("lowerpad", "tau Histograms", 0.0, 0.0, 1.0, 0.75)
lowerpad.Divide(3,3)
lowerpad.Draw()

#Draw jet histograms to upperpad
jet_toDraw = ["jet_pt", "ajet_eta", "jet_E"]
for i in range(0, len(jet_toDraw)):
    print(i)
    draw = jet_toDraw[i]
    upperpad.cd(i+1)
    histos[draw].Draw()

#Draw tau histograms to lowerpad
tau_toDraw = ["tau_n", "tau_pt", "atau_eta", "tau_phi", "tau_E", "tau_charge", "tau_nTracks",
              "tau_BDTid", "ditau_m"]

for i in range(0, len(tau_toDraw)):
    draw = tau_toDraw[i]
    lowerpad.cd(i+1)
    histos[draw].Draw()

#
c1.Update()

#Save Canvas as PDF
c1.SaveAs(str("jet_tau_histograms_041121"+TFile_name+".pdf"))

input("return to stop histos ->")