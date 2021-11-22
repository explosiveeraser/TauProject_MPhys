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

back_data = Dataset(back_dir)
sig_data = Dataset(sig_dir)


canvases = {}
canvases["Canvas_0"] = ROOT.TCanvas("Canvas_0", "Canvas_0")
canvases["Canvas_0"].Divide(4,4)
canvases["Canvas_0"].cd(0)

text = ROOT.TText(.5, .5, "Plots")
c = 0
j = 1
legend={}

'''
ROOT.gInterpreter.Declare("""
float k_test(UInt_t size1, Double_t* a1, UInt_t size2, Double_t* a2) {
    auto test = ROOT::Math::GoFTest(size1, a1, size2, a2);
    float k = test.KolmogorovSmirnov2SamplesTest();
    return k;
}
""")
'''

def k_test(sample_1Size, sample_1, sample_2Size, sample_2):
    size1 = sample_1Size
    size2 = sample_2Size
    array1 = np.array(sample_1, dtype=float)
    array2 = np.array(sample_2, dtype=float)
    print("array1: {}".format(array1))
    test = ROOT.Math.GoFTest(size1, array1, size2, array2)
    k = test.KolmogorovSmirnov2SamplesTest()
    return k

for branch in tqdm(back_data.Histograms):
    for leaf in back_data.Histograms[branch]:
        if True:
            canvases["Canvas_{}".format(c)].cd(j)
            back_data.Histograms[branch][leaf].Draw("HIST")
            canvases["Canvas_{}".format(c)].cd(j+1)
            sig_data.Histograms[branch][leaf].SetLineColor(ROOT.kRed)
            sig_data.Histograms[branch][leaf].Draw("HIST")
            #sample_1Size, sample_1 = back_data.get_sample_for_k_test(branch, leaf)
            #sample_2Size, sample_2 = sig_data.get_sample_for_k_test(branch, leaf)
            #k_num = k_test(sample_1Size, sample_1, sample_2Size, sample_2)
            #k_num = np.round(k_num, 4)
            #text.DrawTextNDC(.0, .0, "Kolmogorov Test: {}".format(k_num))
            #legend["legend_{}_{}".format(c, j)] = ROOT.TLegend(0.05, 0.85, 0.2, 0.95)
            #legend["legend_{}_{}".format(c, j)].SetHeader("Histogram Colors:")
            #legend["legend_{}_{}".format(c, j)].AddEntry(back_data.Histograms[branch][leaf], "Background Data", "L")
            #legend["legend_{}_{}".format(c, j)].AddEntry(sig_data.Histograms[branch][leaf], "Signal Data", "L")
            #legend["legend_{}_{}".format(c, j)].Draw()
            j += 2
            #del k_num
            if j > 17:
                canvases["Canvas_{}".format(c)].Update()
                c += 1
                canvases["Canvas_{}".format(c)] = ROOT.TCanvas("Canvas_{}".format(c), "Canvas_{}".format(c))
                canvases["Canvas_{}".format(c)].Divide(4,4)
                canvases["Canvas_{}".format(c)].cd(0)
                j = 1


#gROOT.GetListOfCanvases().Draw()
back_data.print_num_of_each_object()
sig_data.print_num_of_each_object()

i = 0
for canvas in canvases.keys():
    canvases[canvas].Print("Canvas_{}.pdf".format(i))
    i+=1

input("Enter to quit")

