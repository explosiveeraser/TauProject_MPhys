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
sig_dir = "Delphes_Signal_wPU/"


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
        ymax = Get_User_Ranges(sig_data.Histograms[branch][leaf], back_data.Histograms[branch][leaf])
        back_data.Histograms[branch][leaf].GetYaxis().SetRangeUser(0, ymax*1.2)
        sig_data.Histograms[branch][leaf].GetYaxis().SetRangeUser(0, ymax*1.2)
        back_data.Histograms[branch][leaf].Draw("HIST")
        sig_data.Histograms[branch][leaf].SetLineColor(ROOT.kRed)
        sig_data.Histograms[branch][leaf].Draw("HIST SAMES0")
        if back_data.Histograms[branch][leaf].Integral() == 0 or sig_data.Histograms[branch][leaf].Integral() == 0:
            chi_test = back_data.Histograms[branch][leaf].Chi2Test(sig_data.Histograms[branch][leaf], option="WW NORM")
            text.DrawTextNDC(.0, .0, "Chi2Test: {}".format(chi_test))
        else:
            k_test = back_data.Histograms[branch][leaf].KolmogorovTest(sig_data.Histograms[branch][leaf])
            text.DrawTextNDC(.0, .0, "Kolmogorov Test: {}".format(k_test))
        legend["legend_{}_{}".format(c, j)] = ROOT.TLegend(0.05, 0.85, 0.2, 0.95)
        legend["legend_{}_{}".format(c, j)].SetHeader("Histogram Colors:")
        legend["legend_{}_{}".format(c, j)].AddEntry(back_data.Histograms[branch][leaf], "Background Data", "L")
        legend["legend_{}_{}".format(c, j)].AddEntry(sig_data.Histograms[branch][leaf], "Signal Data", "L")
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
sig_data.print_num_of_each_object()

i = 0
for canvas in canvases.keys():
    canvases[canvas].Print("Canvas_{}.pdf".format(i))
    i+=1

input("Enter to quit")

