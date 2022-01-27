import numpy as np
import modin.pandas as pd
from DataSet_Reader import Dataset
import ROOT
from ROOT import gROOT
import numba
from numba import jit, jit_module
import os, os.path
from tqdm import tqdm, trange
from Background_DataReader import Background
from Signal_DataReader import Signal
"""
Load delphes shared library located in 
delphes install library directory
"""


#Convention Signal First and Background Second
class DataProcessing():

    def __init__(self, SigDir, BackDir, print_hists=True):
        self.signal = Signal(SigDir, print_hist=False)
        self.background = Background(BackDir, print_hist=False)
        self.signal.write_taucan_ttree("signal_tree")
        self.background.write_taucan_ttree("background_tree")
        self.canvases = {}
        self.legend = {}
        self.Hist_started = False
    
    def _get_user_ranges(self, hist_1, hist_2):
        ymax1 = hist_1.GetBinContent(hist_1.GetMaximumBin())
        ymax2 = hist_2.GetBinContent(hist_2.GetMaximumBin())
        ymax = max(ymax2, ymax1)
        return ymax

    def Sig_Hist_Tau(self):
        if not self.Hist_started:
            self.canvases["TauCan_0"] = ROOT.TCanvas("TauCan_0", "TauCan_0")
            self.canvases["TauCan_0"].Divide(3, 3)
            self.canvases["TauCan_0"].cd(0)
            self.text = ROOT.TText(.5, .5, "Plots")
            c = 0
            j = 1
            self.Hist_started = True
        else:
            j = 1
            c = len(self.canvases) + 1
            self.canvases["TauCan_{}".format(c)] = ROOT.TCanvas("TauCan_{}".format(c), "TauCan_{}".format(c))
            self.canvases["TauCan_{}".format(c)].Divide(3, 3)
            self.canvases["TauCan_{}".format(c)].cd(0)
        for branch in tqdm(self.signal.Histograms):
            for leaf in self.signal.Histograms[branch]:
                self.canvases["TauCan_{}".format(c)].cd(j)
                ymax = self._get_user_ranges(self.signal.Tau_Histograms[branch][leaf], self.signal.Histograms[branch][leaf])
                self.signal.Histograms[branch][leaf].GetYaxis().SetRangeUser(0, ymax * 1.2)
                self.signal.Tau_Histograms[branch][leaf].GetYaxis().SetRangeUser(0, ymax * 1.2)
                self.signal.Histograms[branch][leaf].Draw("HIST")
                self.signal.Tau_Histograms[branch][leaf].SetLineColor(ROOT.kRed)
                self.signal.Tau_Histograms[branch][leaf].Draw("HIST SAMES0")
                if self.signal.Histograms[branch][leaf].Integral() == 0 or self.signal.Histograms[branch][
                    leaf].Integral() == 0:
                    chi_test = self.signal.Histograms[branch][leaf].Chi2Test(self.signal.Tau_Histograms[branch][leaf],
                                                                          option="WW NORM")
                    self.text.DrawTextNDC(.0, .0, "Chi2Test: {}".format(chi_test))
                else:
                    k_test = self.signal.Histograms[branch][leaf].KolmogorovTest(self.signal.Tau_Histograms[branch][leaf])
                    self.text.DrawTextNDC(.0, .0, "Kolmogorov Test: {}".format(k_test))
                self.legend["legend_{}_{}".format(c, j)] = ROOT.TLegend(0.05, 0.85, 0.2, 0.95)
                self.legend["legend_{}_{}".format(c, j)].SetHeader("Histogram Colors:")
                self.legend["legend_{}_{}".format(c, j)].AddEntry(self.signal.Histograms[branch][leaf],
                                                             "Non-Tau-Tagged Signal Data", "L")
                self.legend["legend_{}_{}".format(c, j)].AddEntry(self.signal.Tau_Histograms[branch][leaf],
                                                             "Tau-Tagged Signal Data", "L")
                self.legend["legend_{}_{}".format(c, j)].Draw()
                j += 1
                if j > 9:
                    self.canvases["TauCan_{}".format(c)].Update()
                    c += 1
                    self.canvases["TauCan_{}".format(c)] = ROOT.TCanvas("TauCan_{}".format(c), "TauCan_{}".format(c))
                    self.canvases["TauCan_{}".format(c)].Divide(3, 3)
                    self.canvases["TauCan_{}".format(c)].cd(0)
                    j = 1

    def Back_Hist_Tau(self):
        if not self.Hist_started:
            self.canvases["TauCan_0"] = ROOT.TCanvas("TauCan_0", "TauCan_0")
            self.canvases["TauCan_0"].Divide(3, 3)
            self.canvases["TauCan_0"].cd(0)
            self.text = ROOT.TText(.5, .5, "Plots")
            c = 0
            j = 1
            self.Hist_started = True
        else:
            j = 1
            c = len(self.canvases) + 1
            self.canvases["TauCan_{}".format(c)] = ROOT.TCanvas("TauCan_{}".format(c), "TauCan_{}".format(c))
            self.canvases["TauCan_{}".format(c)].Divide(3, 3)
            self.canvases["TauCan_{}".format(c)].cd(0)
        for branch in tqdm(self.background.Histograms):
            for leaf in self.background.Histograms[branch]:
                self.canvases["TauCan_{}".format(c)].cd(j)
                ymax = self._get_user_ranges(self.background.Tau_Histograms[branch][leaf], self.background.Histograms[branch][leaf])
                self.background.Histograms[branch][leaf].GetYaxis().SetRangeUser(0, ymax * 1.2)
                self.background.Tau_Histograms[branch][leaf].GetYaxis().SetRangeUser(0, ymax * 1.2)
                self.background.Histograms[branch][leaf].Draw("HIST")
                self.background.Tau_Histograms[branch][leaf].SetLineColor(ROOT.kRed)
                self.background.Tau_Histograms[branch][leaf].Draw("HIST SAMES0")
                if self.background.Histograms[branch][leaf].Integral() == 0 or self.background.Tau_Histograms[branch][
                    leaf].Integral() == 0:
                    chi_test = self.background.Histograms[branch][leaf].Chi2Test(self.background.Tau_Histograms[branch][leaf],
                                                                           option="WW NORM")
                    self.text.DrawTextNDC(.0, .0, "Chi2Test: {}".format(chi_test))
                else:
                    k_test = self.background.Histograms[branch][leaf].KolmogorovTest(self.background.Tau_Histograms[branch][leaf])
                    self.text.DrawTextNDC(.0, .0, "Kolmogorov Test: {}".format(k_test))
                self.legend["legend_{}_{}".format(c, j)] = ROOT.TLegend(0.05, 0.85, 0.2, 0.95)
                self.legend["legend_{}_{}".format(c, j)].SetHeader("Histogram Colors:")
                self.legend["legend_{}_{}".format(c, j)].AddEntry(self.background.Histograms[branch][leaf],
                                                             "Non-Tau Tagged Background Data", "L")
                self.legend["legend_{}_{}".format(c, j)].AddEntry(self.background.Tau_Histograms[branch][leaf],
                                                             "Tau Tagged Brackground Data", "L")
                self.legend["legend_{}_{}".format(c, j)].Draw()
                j += 1
                if j > 9:
                    self.canvases["TauCan_{}".format(c)].Update()
                    c += 1
                    self.canvases["TauCan_{}".format(c)] = ROOT.TCanvas("TauCan_{}".format(c), "TauCan_{}".format(c))
                    self.canvases["TauCan_{}".format(c)].Divide(3, 3)
                    self.canvases["TauCan_{}".format(c)].cd(0)
                    j = 1

    def Sig_Back_Hist(self):
        if not self.Hist_started:
            self.canvases["TauCan_0"] = ROOT.TCanvas("TauCan_0", "TauCan_0")
            self.canvases["TauCan_0"].Divide(3, 3)
            self.canvases["TauCan_0"].cd(0)
            self.text = ROOT.TText(.5, .5, "Plots")
            c = 0
            j = 1
            self.Hist_started = True
        else:
            j = 1
            c = len(self.canvases) + 1
            self.canvases["TauCan_{}".format(c)] = ROOT.TCanvas("TauCan_{}".format(c), "TauCan_{}".format(c))
            self.canvases["TauCan_{}".format(c)].Divide(3, 3)
            self.canvases["TauCan_{}".format(c)].cd(0)
        for branch in tqdm(self.background.Histograms):
            for leaf in self.background.Histograms[branch]:
                self.canvases["TauCan_{}".format(c)].cd(j)
                ymax = self._get_user_ranges(self.signal.Histograms[branch][leaf], self.background.Histograms[branch][leaf])
                self.background.Histograms[branch][leaf].GetYaxis().SetRangeUser(0, ymax * 1.2)
                self.signal.Histograms[branch][leaf].GetYaxis().SetRangeUser(0, ymax * 1.2)
                self.background.Histograms[branch][leaf].Draw("HIST")
                self.signal.Histograms[branch][leaf].SetLineColor(ROOT.kRed)
                self.signal.Histograms[branch][leaf].Draw("HIST SAMES0")
                if self.background.Histograms[branch][leaf].Integral() == 0 or self.signal.Histograms[branch][
                    leaf].Integral() == 0:
                    chi_test = self.background.Histograms[branch][leaf].Chi2Test(self.signal.Histograms[branch][leaf],
                                                                           option="WW NORM")
                    self.text.DrawTextNDC(.0, .0, "Chi2Test: {}".format(chi_test))
                else:
                    k_test = self.background.Histograms[branch][leaf].KolmogorovTest(self.signal.Histograms[branch][leaf])
                    self.text.DrawTextNDC(.0, .0, "Kolmogorov Test: {}".format(k_test))
                self.legend["legend_{}_{}".format(c, j)] = ROOT.TLegend(0.05, 0.85, 0.2, 0.95)
                self.legend["legend_{}_{}".format(c, j)].SetHeader("Histogram Colors:")
                self.legend["legend_{}_{}".format(c, j)].AddEntry(self.background.Histograms[branch][leaf],
                                                             "Background Data", "L")
                self.legend["legend_{}_{}".format(c, j)].AddEntry(self.signal.Histograms[branch][leaf],
                                                             "Signal Data", "L")
                self.legend["legend_{}_{}".format(c, j)].Draw()
                j += 1
                if j > 9:
                    self.canvases["TauCan_{}".format(c)].Update()
                    c += 1
                    self.canvases["TauCan_{}".format(c)] = ROOT.TCanvas("TauCan_{}".format(c), "TauCan_{}".format(c))
                    self.canvases["TauCan_{}".format(c)].Divide(3, 3)
                    self.canvases["TauCan_{}".format(c)].cd(0)
                    j = 1

    def Tau_Sig_Back_Hist(self):
        if not self.Hist_started:
            self.canvases["TauCan_0"] = ROOT.TCanvas("TauCan_0", "TauCan_0")
            self.canvases["TauCan_0"].Divide(3, 3)
            self.canvases["TauCan_0"].cd(0)
            self.text = ROOT.TText(.5, .5, "Plots")
            c = 0
            j = 1
            self.Hist_started = True
        else:
            j = 1
            c = len(self.canvases) + 1
            self.canvases["TauCan_{}".format(c)] = ROOT.TCanvas("TauCan_{}".format(c), "TauCan_{}".format(c))
            self.canvases["TauCan_{}".format(c)].Divide(3, 3)
            self.canvases["TauCan_{}".format(c)].cd(0)
        for branch in tqdm(self.background.Tau_Histograms):
            for leaf in self.background.Tau_Histograms[branch]:
                self.canvases["TauCan_{}".format(c)].cd(j)
                ymax = self._get_user_ranges(self.background.Tau_Histograms[branch][leaf], self.signal.Tau_Histograms[branch][leaf])
                self.background.Tau_Histograms[branch][leaf].GetYaxis().SetRangeUser(0, ymax * 1.2)
                self.signal.Tau_Histograms[branch][leaf].GetYaxis().SetRangeUser(0, ymax * 1.2)
                self.background.Tau_Histograms[branch][leaf].Draw("HIST")
                self.signal.Tau_Histograms[branch][leaf].SetLineColor(ROOT.kRed)
                self.signal.Tau_Histograms[branch][leaf].Draw("HIST SAMES0")
                if self.background.Tau_Histograms[branch][leaf].Integral() == 0 or self.signal.Tau_Histograms[branch][
                    leaf].Integral() == 0:
                    chi_test = self.background.Tau_Histograms[branch][leaf].Chi2Test(self.signal.Tau_Histograms[branch][leaf],
                                                                           option="WW NORM")
                    self.text.DrawTextNDC(.0, .0, "Chi2Test: {}".format(chi_test))
                else:
                    k_test = self.background.Tau_Histograms[branch][leaf].KolmogorovTest(self.signal.Tau_Histograms[branch][leaf])
                    self.text.DrawTextNDC(.0, .0, "Kolmogorov Test: {}".format(k_test))
                self.legend["legend_{}_{}".format(c, j)] = ROOT.TLegend(0.05, 0.85, 0.2, 0.95)
                self.legend["legend_{}_{}".format(c, j)].SetHeader("Histogram Colors:")
                self.legend["legend_{}_{}".format(c, j)].AddEntry(self.background.Tau_Histograms[branch][leaf],
                                                             "Tau Tagged Background Data", "L")
                self.legend["legend_{}_{}".format(c, j)].AddEntry(self.signal.Tau_Histograms[branch][leaf],
                                                             "Tau Tagged Signal Data", "L")
                self.legend["legend_{}_{}".format(c, j)].Draw()
                j += 1
                if j > 9:
                    self.canvases["TauCan_{}".format(c)].Update()
                    c += 1
                    self.canvases["TauCan_{}".format(c)] = ROOT.TCanvas("TauCan_{}".format(c), "TauCan_{}".format(c))
                    self.canvases["TauCan_{}".format(c)].Divide(3, 3)
                    self.canvases["TauCan_{}".format(c)].cd(0)
                    j = 1

    def Print_Test(self):
        self.signal.print_test_arrays(self.signal.JetArray)
        self.background.print_test_arrays(self.background.JetArray)

    def Print_Num_of_Tau(self):
        self.background.print_num_of_each_object()
        print("back data number of tau jets: {}".format(self.background.num_tau_jets))
        self.signal.print_num_of_each_object()
        print("sig data number of tau jets: {}".format(self.signal.num_tau_jets))

    def Print_Canvases(self):
        i = 0
        for canvas in self.canvases.keys():
            self.canvases[canvas].Print("Histograms/Original_Tree_Properties/TauCan_{}.pdf".format(i))
            i += 1
