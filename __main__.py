import numpy as np
from array import array
import ROOT
from ROOT import gROOT

ROOT.ROOT.EnableImplicitMT()
ROOT.gStyle.SetOptStat(0)

c1 = ROOT.TCanvas("c1", "Title", 900, 700)
c1.SetGrid()

mc_file = ROOT.TFile.Open("1lep1tau/MC/mc_341123.ggH125_tautaulh.1lep1tau.root", "READ")
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
Make some histograms of the following values:
jet_pt and tau_pt (rescaled)
absolute values of eta for jets and tau (rescaled for tau)
tau_nTracks
computed invariant mass for tau (rescaled)
tau_pt (1 track) and tau_pt (3 tracks) and tau_pt (>3)
jet_E and tau_E (rescaled) 
"""

ROOT.gInterpreter.Declare(
"""
using Vec_t = const ROOT::VecOps::RVec<float>;
Vec_t& ComputeInvariantMass(Vec_t& pt, Vec_t& eta, Vec_t& phi, Vec_t& e) {
    auto fourVecs = ROOT::VecOps::Construct<ROOT::Math::PtEtaPhiE4D<float>>(pt, eta, phi, e);
    auto M = [](ROOT::Math::PtEtaPhiE4D<float> x) { return x.M();};
    auto m = ROOT::VecOps::Map(fourVecs, M);
    return m;
}
""")

mc_DFrame = mc_DFrame.Define("tau_m", "ComputeInvariantMass(tau_pt, tau_eta, tau_phi, tau_E)")

histos = {}

histos["jet_pt"] = mc_DFrame.Histo1D("jet_pt")
histos["tau_pt"] = mc_DFrame.Histo1D("tau_pt", "weight")
histos["jet_eta"] = mc_DFrame.Define("ajet_eta", "abs(jet_eta)").Histo1D("ajet_eta")
histos["tau_eta"] = mc_DFrame.Define("atau_eta", "abs(tau_eta)").Histo1D("atau_eta", "weight")
histos["tau_nTrack"] = mc_DFrame.Histo1D("tau_nTracks")
histos["tau_m"] = mc_DFrame.Histo1D("tau_m", "weight")
histos["jet_e"] = mc_DFrame.Histo1D("jet_E")
histos["tau_e"] = mc_DFrame.Histo1D("tau_E", "weight")

mc_DFrame = mc_DFrame.Define("one_tau", "tau_nTracks[tau_nTracks == 1]")
mc_DFrame = mc_DFrame.Define("three_tau", "tau_nTracks[tau_nTracks == 3]")
mc_DFrame = mc_DFrame.Define("o_tau", "tau_nTracks[tau_nTracks > 3]")

histos["1_tau"] = mc_DFrame.Histo1D("one_tau")
histos["3_tau"] = mc_DFrame.Histo1D("three_tau")
histos["o_tau"] = mc_DFrame.Histo1D("o_tau")

d = mc_DFrame.Filter("tau_m").Take("tau_m")
print(d.Print())

c1.Update()

input("return to stop histos ->")