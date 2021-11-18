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

background = Dataset("Delphes_Background/", get_Histos=True)
signal = Dataset("Delphes_Signal/", get_Histos=True)

def build_stacks(Histos1, Histos2):
    Histos_Stacks = {}
    no_hists = 0
    for phys_obj in Histos1:
        Histos_Stacks[phys_obj] = {}
        for prop in Histos1[phys_obj]:
            name = str(phys_obj+prop)
            Histos_Stacks[phys_obj][prop] = {"Histo": ROOT.THStack(name, name), "Kolmogorov": 0.}
            Histos_Stacks[phys_obj][prop]["Histo"].Add(Histos1[phys_obj][prop])
            Histos_Stacks[phys_obj][prop]["Histo"].Add(Histos2[phys_obj][prop])
            print(Histos1[phys_obj][prop].GetMaximumBin())
            if Histos1[phys_obj][prop].Integral() != 0.:
                K_test =  Histos1[phys_obj][prop].KolmogorovTest(Histos2[phys_obj][prop])
            else:
                K_test = 0.
            Histos_Stacks[phys_obj][prop]["Kolmogorov"] = K_test
            no_hists += 1
    return Histos_Stacks, no_hists

def canvas_builder(no_obj, div=[3, 3]):
    print(no_obj)
    no_canvases = np.ceil(no_obj/(div[0]*div[1]))#
    canvases = {}
    for i in range(0, np.int(no_canvases)):
        canvases["Canvas_"+str(i)] = ROOT.TCanvas("Canvas_"+str(i), "Canvas_"+str(i))
        canvases["Canvas_"+str(i)].Divide(div[0], div[1])
        print("Canvas_"+str(i))
    return canvases




Histo_Stacks, no_hists = build_stacks(background.Histos, signal.Histos)

has_plotted = False
canvases = canvas_builder(no_hists)
print(canvases.keys())
text = ROOT.TText()
text.SetTextFont(25)
text.SetTextAlign(21)

hists_on_canvas = 1
current_canvas = 0
for obj in tqdm(Histo_Stacks):
    canvases["Canvas_" + str(current_canvas)].cd(0)
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
    canvases["Canvas_"+str(current_canvas)].Update()

input("blob")



