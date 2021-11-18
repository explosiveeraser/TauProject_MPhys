import numpy as np
from ROOT import *
from ROOT import gROOT
import numba
from numba import jit, jit_module
import os, os.path
from event_gen.process_MCdata import *
from DataReader import Dataset
from tqdm import tqdm

"""
Load delphes shared library located in 
delphes install library directory
"""
ROOT.gSystem.Load("install/lib/libDelphes")

# ROOT.ROOT.EnableImplicitMT()
ROOT.gStyle.SetOptStat(0)

background = Dataset("Delphes_Background/")
signal = Dataset("Delphes_Signal/")

def build_stacks(Histos1, Histos2):
    Histos_Stacks = {}
    no_hists = 0
    for phys_obj in Histos1:
        for prop in Histos1[phys_obj]:
            name = str(phys_obj+prop)
            Histos_Stacks[phys_obj][prop] = {"Histo": ROOT.THStack(name, name), "Kolmogorov": 0.}
            Histos_Stacks[phys_obj][prop].Add(Histos1[phys_obj][prop], Histos2[phys_obj][prop])
            K_test =  Histos1[phys_obj][prop].KolmogorovTest(Histos2[phys_obj][prop])
            Histos_Stacks[phys_obj][prop]["Kolmogorov"] = K_test
            no_hists += 1
    return Histos_Stacks, no_hists

def canvas_builder(no_obj, div=[3, 3]):
    no_canvases = np.ceil(no_obj/(div[0]*div[1]))
    canvases = {}
    for i in range(no_canvases):
        canvases["Canvas_"+str(i)] = ROOT.TCanvas("Canvas_"+str(i), "Canvas_"+str(i))
        canvases["Canvas_"+str(i)].Divide(div[0], div[1])
    return  canvases




Histo_Stacks, no_hists = build_stacks(background.Histos, signal.Histos)

has_plotted = False
canvases = canvas_builder(no_hists)
text = ROOT.TText()
text.SetTextFont(25)
text.SetTextAlign(21)

hists_on_canvas = 0
current_canvas = 1
for obj in tqdm(Histo_Stacks):
    for prop in Histo_Stacks[obj]:
        if hists_on_canvas < 9:
            hists_on_canvas += 1
            canvases["Canvas_"+str(current_canvas)].cd(hists_on_canvas)
            text.DrawTextNDC(.5, .95, str(Histo_Stacks[obj][prop]["Kolmogorov"]))
            Histo_Stacks[obj][prop]["Histo"].Draw()
        else:
            hists_on_canvas += 1
            canvases["Canvas_" + str(current_canvas)].cd(hists_on_canvas)
            text.DrawTextNDC(.5, .95, str(Histo_Stacks[obj][prop]["Kolmogorov"]))
            Histo_Stacks[obj][prop]["Histo"].Draw()
            hists_on_canvas = 0
            current_canvas += 1

input("blob")



