import numpy as np
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

# ROOT.ROOT.EnableImplicitMT()
ROOT.gStyle.SetOptStat(0)

back_dir = "Delphes_Background/"
sig_dir = "Delphes_Signal/"

back_data = Dataset(back_dir)
sig_data = Dataset(sig_dir)

canvases = {}
canvases["Canvas_0"] = ROOT.TCanvas("Canvas_0", "Canvas_0")
canvases["Canvas_0"].Divide(3,3)
canvases["Canvas_0"].cd(0)

text = ROOT.TText(.5, .5, "Plots")
c = 0
j = 1

for branch in tqdm(back_data.Histograms):
    for leaf in back_data.Histograms[branch]:
        if back_data.Histograms[branch][leaf] != None and sig_data.Histograms[branch][leaf] != None:
            back_max = back_data.Histograms[branch][leaf].GetMaximum()
            sig_max = sig_data.Histograms[branch][leaf].GetMaximum()
            back_min = back_data.Histograms[branch][leaf].GetMinimum()
            sig_min = sig_data.Histograms[branch][leaf].GetMinimum()
            canvases["Canvas_{}".format(c)].cd(j)
            back_data.Histograms[branch][leaf].Draw()
            sig_data.Histograms[branch][leaf].SetLineColor(ROOT.kRed)
            sig_data.Histograms[branch][leaf].Draw("SAME")
            k_test = back_data.Histograms[branch][leaf].KolmogorovTest(sig_data.Histograms[branch][leaf])
            if k_test == 0:
                k_test = "Unknown"
            text.DrawTextNDC(.5, .05, "Kolmogorov Test: {}".format(k_test))
            j += 1
            del k_test
            if j == 10:
                canvases["Canvas_{}".format(c)].Update()
                c += 1
                canvases["Canvas_{}".format(c)] = ROOT.TCanvas("Canvas_{}".format(c), "Canvas_{}".format(c))
                canvases["Canvas_{}".format(c)].Divide(3,3)
                canvases["Canvas_{}".format(c)].cd(0)
                j = 1


#gROOT.GetListOfCanvases().Draw()
input("Enter to quit")