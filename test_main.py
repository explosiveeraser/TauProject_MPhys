import numpy as np
import modin.pandas as pd
from DataSet_Reader import Dataset
import ROOT
from ROOT import gROOT
import numba
from numba import jit, jit_module
import os, os.path
from tqdm import tqdm, trange

"""
Load delphes shared library located in 
delphes install library directory
"""
ROOT.gSystem.Load("install/lib/libDelphes")

ROOT.ROOT.EnableImplicitMT()
ROOT.gStyle.SetOptStat("ne")

back_dir = "Delphes_Background/"
sig_dir = "Delphes_Signal/"

sig_data = Dataset(sig_dir)
back_data = Dataset(back_dir)


canvases = {}
canvases["Canvas_0"] = ROOT.TCanvas("Canvas_0", "Canvas_0")
canvases["Canvas_0"].Divide(3,3)
canvases["Canvas_0"].cd(0)

text = ROOT.TText(.5, .5, "Plots")
c = 0
j = 1
legend={}

def Get_User_Ranges(hist_1, hist_2):
    ymax1 = hist_1.GetBinContent(hist_1.GetMaximumBin())
    ymax2 = hist_2.GetBinContent(hist_2.GetMaximumBin())
    ymax = max(ymax2, ymax1)
    return ymax

for branch in tqdm(back_data.Histograms):
    for leaf in back_data.Histograms[branch]:
        canvases["Canvas_{}".format(c)].cd(j)
        ymax = Get_User_Ranges(back_data.Tau_Histograms[branch][leaf], back_data.Histograms[branch][leaf])
        back_data.Histograms[branch][leaf].GetYaxis().SetRangeUser(0, ymax*1.2)
        back_data.Tau_Histograms[branch][leaf].GetYaxis().SetRangeUser(0, ymax*1.2)
        back_data.Histograms[branch][leaf].Draw("HIST")
        back_data.Tau_Histograms[branch][leaf].SetLineColor(ROOT.kRed)
        back_data.Tau_Histograms[branch][leaf].Draw("HIST SAMES0")
        if back_data.Histograms[branch][leaf].Integral() == 0 or back_data.Tau_Histograms[branch][leaf].Integral() == 0:
            chi_test = back_data.Histograms[branch][leaf].Chi2Test(back_data.Tau_Histograms[branch][leaf], option="WW NORM")
            text.DrawTextNDC(.0, .0, "Chi2Test: {}".format(chi_test))
        else:
            k_test = back_data.Histograms[branch][leaf].KolmogorovTest(back_data.Tau_Histograms[branch][leaf])
            text.DrawTextNDC(.0, .0, "Kolmogorov Test: {}".format(k_test))
        legend["legend_{}_{}".format(c, j)] = ROOT.TLegend(0.05, 0.85, 0.2, 0.95)
        legend["legend_{}_{}".format(c, j)].SetHeader("Histogram Colors:")
        legend["legend_{}_{}".format(c, j)].AddEntry(back_data.Histograms[branch][leaf], "Non-Tau Tagged Background Data", "L")
        legend["legend_{}_{}".format(c, j)].AddEntry(back_data.Tau_Histograms[branch][leaf], "Tau Tagged Brackground Data", "L")
        legend["legend_{}_{}".format(c, j)].Draw()
        j += 1
        if j > 9:
            canvases["Canvas_{}".format(c)].Update()
            c += 1
            canvases["Canvas_{}".format(c)] = ROOT.TCanvas("Canvas_{}".format(c), "Canvas_{}".format(c))
            canvases["Canvas_{}".format(c)].Divide(3,3)
            canvases["Canvas_{}".format(c)].cd(0)
            j = 1

j = 1
c += 1
canvases["Canvas_{}".format(c)] = ROOT.TCanvas("Canvas_{}".format(c), "Canvas_{}".format(c))
canvases["Canvas_{}".format(c)].Divide(3,3)
canvases["Canvas_{}".format(c)].cd(0)

for branch in tqdm(sig_data.Histograms):
    for leaf in sig_data.Histograms[branch]:
        canvases["Canvas_{}".format(c)].cd(j)
        ymax = Get_User_Ranges(sig_data.Tau_Histograms[branch][leaf], sig_data.Histograms[branch][leaf])
        sig_data.Histograms[branch][leaf].GetYaxis().SetRangeUser(0, ymax*1.2)
        sig_data.Tau_Histograms[branch][leaf].GetYaxis().SetRangeUser(0, ymax*1.2)
        sig_data.Histograms[branch][leaf].Draw("HIST")
        sig_data.Tau_Histograms[branch][leaf].SetLineColor(ROOT.kRed)
        sig_data.Tau_Histograms[branch][leaf].Draw("HIST SAMES0")
        if sig_data.Histograms[branch][leaf].Integral() == 0 or sig_data.Histograms[branch][leaf].Integral() == 0:
            chi_test = sig_data.Histograms[branch][leaf].Chi2Test(sig_data.Tau_Histograms[branch][leaf], option="WW NORM")
            text.DrawTextNDC(.0, .0, "Chi2Test: {}".format(chi_test))
        else:
            k_test = sig_data.Histograms[branch][leaf].KolmogorovTest(sig_data.Tau_Histograms[branch][leaf])
            text.DrawTextNDC(.0, .0, "Kolmogorov Test: {}".format(k_test))
        legend["legend_{}_{}".format(c, j)] = ROOT.TLegend(0.05, 0.85, 0.2, 0.95)
        legend["legend_{}_{}".format(c, j)].SetHeader("Histogram Colors:")
        legend["legend_{}_{}".format(c, j)].AddEntry(back_data.Histograms[branch][leaf], "Non-Tau-Tagged Signal Data", "L")
        legend["legend_{}_{}".format(c, j)].AddEntry(back_data.Tau_Histograms[branch][leaf], "Tau-Tagged Signal Data", "L")
        legend["legend_{}_{}".format(c, j)].Draw()
        j += 1
        if j > 9:
            canvases["Canvas_{}".format(c)].Update()
            c += 1
            canvases["Canvas_{}".format(c)] = ROOT.TCanvas("Canvas_{}".format(c), "Canvas_{}".format(c))
            canvases["Canvas_{}".format(c)].Divide(3,3)
            canvases["Canvas_{}".format(c)].cd(0)
            j = 1


back_data.print_num_of_each_object()
print("back data number of tau jets: {}".format(back_data.num_tau_jets))
sig_data.print_num_of_each_object()
print("sig data number of tau jets: {}".format(sig_data.num_tau_jets))

i = 0
for canvas in canvases.keys():
    canvases[canvas].Print("Canvas_{}.pdf".format(i))
    i+=1

back_data.print_test_arrays(back_data.JetTestArray)
back_data.print_test_arrays(back_data.TJetTestArray)
sig_data.print_test_arrays(sig_data.JetTestArray)
sig_data.print_test_arrays(sig_data.TJetTestArray)

input("Enter to quit")

