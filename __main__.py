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

back_data = Dataset("Delphes_Background/")
sig_data = Dataset("Delphes_Signal/")

def Stacker(Hist_1, Hist_2):
    Stack = {}
    i = 0
    for branch in Hist_1.keys():
        for leaf in Hist_1[branch].keys():
            name = branch +"."+leaf
            Stack[name] = ROOT.THStack(name, name)
            Stack[name].Add(Hist_1[branch][leaf])
            Stack[name].Add(Hist_2[branch][leaf])
            i += 1
    return Stack, i

def Canvas_Maker(no_stacks, div=[3,3]):
    num_iC = div[0]*div[1]
    num_canvases = no_stacks//num_iC
    num_left = no_stacks % num_iC
    canvases = {}
    for canvas_no in trange(num_canvases):
        name = "Canvas_"+str(canvas_no)
        canvases[name] = ROOT.TCanvas(name, name)
        canvases[name].Divide(div)
    last_name = "Canvas_"+ str(num_canvases+1)
    canvases[last_name] = ROOT.TCanvas(last_name, last_name)
    canvases[last_name].Divide(1, num_left)
    return canvases, num_iC+1, num_left

stacks, num = Stacker(back_data.Histograms, sig_data.Histograms)
canvases, last, in_last = Canvas_Maker(num)

text = ROOT.TText(.5, .5, "Plots")

stack_it = iter(stacks)
for canvas in canvases.keys():
    canvases[canvas].cd(0)
    if canvas != last:
        for i in range(1, 9):
            canvases[canvas].cd(i)
            stack_it.Draw()
            next(stack_it)
    else:
        for i in range(1, in_last):
            canvases[canvas].cd[i]
            stack_it.Draw()
            next(stack_it)

input("Enter to quit")