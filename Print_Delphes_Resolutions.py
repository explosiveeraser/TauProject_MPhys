import array
import gc
import math

import numpy as np
import ROOT
import pandas as pd
from ROOT import gROOT
import numba
from array import array
from numba import jit, jit_module
import os, os.path
from tqdm import tqdm, trange
from DataSet_Reader import Dataset
from Jet import Jet_
from Track import Track_
from Tower import Tower_
from Particle import Particle_
from ROOT import addressof
import ctypes
from functools import partial
import math
import pickle
import keras_preprocessing.sequence
import matplotlib.pyplot as plt
import numpy as np
from ROOT import TMVA, TFile, TTree, TCut
import ROOT
from subprocess import call
from os.path import isfile
import pandas as pd
import random
import os

from tensorflow.python import pywrap_tensorflow as _pywrap_tensorflow
from tensorflow.python.eager import context
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Concatenate
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers import TimeDistributed
import tensorflow as tf
from keras.utils.vis_utils import plot_model

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.timeseries import timeseries_dataset_from_array
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm, trange
import root_numpy as rn
from root_numpy import tree2array
import matplotlib as mpl

def mpl_setup(scale=0.49, aspect_ratio=8.0 / 6.0,
              pad_left=0.16, pad_bottom=0.18,
              pad_right=0.95, pad_top=0.95):
    mpl.rcParams["font.sans-serif"] = ["Liberation Sans", "helvetica",
                                       "Helvetica", "Arial"]
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.size"] = 8
    mpl.rcParams["mathtext.default"] = "regular"

    # LaTeX \the\textwidth
    text_width_pt = 451.58598
    inches_per_pt = 1.0 / 72.27
    fig_width = text_width_pt * inches_per_pt * scale
    fig_height = fig_width / aspect_ratio

    mpl.rcParams["figure.figsize"] = [fig_width, fig_height]

    mpl.rcParams["figure.subplot.left"] = pad_left
    mpl.rcParams["figure.subplot.bottom"] = pad_bottom
    mpl.rcParams["figure.subplot.top"] = pad_top
    mpl.rcParams["figure.subplot.right"] = pad_right

    mpl.rcParams["axes.xmargin"] = 0.0
    mpl.rcParams["axes.ymargin"] = 0.0

    mpl.rcParams["axes.labelsize"] = 10
    mpl.rcParams["axes.linewidth"] = 0.6

    mpl.rcParams["xtick.major.size"] = 6.0
    mpl.rcParams["xtick.major.width"] = 0.6
    mpl.rcParams["xtick.minor.size"] = 3.0
    mpl.rcParams["xtick.minor.width"] = 0.6
    mpl.rcParams["xtick.minor.visible"] = True
    mpl.rcParams["xtick.top"] = True
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["xtick.labelsize"] = 8

    mpl.rcParams["ytick.major.size"] = 6.0
    mpl.rcParams["ytick.major.width"] = 0.6
    mpl.rcParams["ytick.minor.size"] = 3.0
    mpl.rcParams["ytick.minor.width"] = 0.6
    mpl.rcParams["ytick.minor.visible"] = True
    mpl.rcParams["ytick.right"] = True
    mpl.rcParams["ytick.direction"] = "in"
    mpl.rcParams["ytick.labelsize"] = 8

    mpl.rcParams["legend.frameon"] = False

    mpl.rcParams["lines.linewidth"] = 1.1
    mpl.rcParams["lines.markersize"] = 3.0

mpl_setup()

ROOT.gSystem.Load("../Delphes-3.5.0/build/libDelphes.so")

try:
  ROOT.gInterpreter.Declare('#include "classes/DelphesClasses.h"')
  ROOT.gInterpreter.Declare('#include "external/ExRootAnalysis/ExRootTreeReader.h"')
except:
  pass

colors = {
    "red": "#e41a1c",
    "blue": "#377eb8",
    "green": "#4daf4a",
    "violet": "#984ea3",
    "orange": "#ff7f00",
    "yellow": "#ffff33",
    "brown": "#a65628",
    "pink": "#f781bf",
    "grey": "#999999"
}

colorseq = [
    colors["red"],
    colors["blue"],
    colors["green"],
    colors["violet"],
    colors["orange"],
    colors["yellow"],
    colors["brown"],
    colors["pink"],
    colors["grey"]
]

def plt_2hist(title, xvar_name, sig_data, bck_data, sig_weight, bck_weight, num_bins, bins=False, save_dir="Resolution_Histograms/", log_plot=False, legloc=0, hist_max=None, hist_min=None):
    fig, ax = plt.subplots()

    bin_edges = np.histogram_bin_edges(sig_data, bins=num_bins)
    ax.hist(bck_data, weights=bck_weight,
            color=colors["blue"], label="background {}".format(xvar_name), bins=bin_edges, density=True,
            histtype="step")
    ax.hist(sig_data, weights=sig_weight,
            color=colors["red"], label="signal {}".format(xvar_name), bins=bin_edges, density=True, histtype="step")
    print(bin_edges)
    if hist_max != None:
        bin_edges[-1] = hist_max + 10

    if log_plot:
        ax.set_yscale("log")

    ax.legend(loc=legloc)
    ax.set_xlabel("{}".format(xvar_name), x=1, ha="right")
    ax.set_ylabel("Norm. number of entries", y=1, ha="right")
    ax.autoscale()
    if hist_max and hist_min:
        ax.set_xlim((hist_min, hist_max))
    y_lo, y_hi = ax.get_ylim()
    if log_plot:
        d = 0.5 * (y_hi - y_lo)
    else:
        d = 0.30 * (y_hi - y_lo)
    ax.set_ylim(y_lo - d, y_hi + d)
    #ax.set_ylim(hist_min, hist_max)
    plt.savefig("{}{}.png".format(save_dir,title))
    plt.close("all")
    return fig

def jet_association(jet_eta, jet_phi, eta, phi):
    deltaEta = jet_eta - eta
    deltaPhi = jet_phi - phi
    deltaR = math.sqrt((deltaEta)**2+(deltaPhi)**2)
    if deltaR < 0.4:
        return True
    else:
        return False

def true_jet_association(jet_eta, jet_phi, eta, phi, jet_deltaR):
    deltaEta = jet_eta - eta
    deltaPhi = jet_phi - phi
    deltaR = math.sqrt((deltaEta)**2+(deltaPhi)**2)
    if deltaR <= jet_deltaR:
        return True
    else:
        return False

sig_wPU_dir = "../sdb5/Delphes_Signal_wPU/0_file/"
a_back_wPU_dir = "../sdb5/Delphes_Background_wPU/0_file/"
b_back_wPU_dir = "../sdb5/Delphes_Background_wPU/1_file/"
c_back_wPU_dir = "../sdb5/Delphes_Background_wPU/2_file/"
d_back_wPU_dir = "../sdb5/Delphes_Background_wPU/3_file/"
e_back_wPU_dir = "../sdb5/Delphes_Background_wPU/4_file/"

back_chain = ROOT.TChain("Delphes")
sig_chain = ROOT.TChain("Delphes")

for dir in [a_back_wPU_dir, b_back_wPU_dir, c_back_wPU_dir, d_back_wPU_dir, e_back_wPU_dir]:
    for f in os.listdir(dir):
        back_chain.Add(dir + f)
for dir in [sig_wPU_dir]:
    for f in os.listdir(dir):
        sig_chain.Add(dir + f)

print_vars = {
    "Event" : ["Scale"],
    "Jet" : ["PT", "Eta", "Phi", "Mass", "DeltaEta", "DeltaPhi", "Charge"]
}

incl_branches = ["Event", "Jet", "Track", "Tower"]

back_branches = list(b for b in map(lambda b: b.GetName(), back_chain.GetListOfBranches()))
for branch in back_branches:
    if branch not in incl_branches:
        back_chain.SetBranchStatus(branch, status=0)

sig_branches = list(b for b in map(lambda b: b.GetName(), sig_chain.GetListOfBranches()))
for branch in sig_branches:
    if branch not in incl_branches:
        sig_chain.SetBranchStatus(branch, status=0)

back_reader = ROOT.ExRootTreeReader(back_chain)
sig_reader = ROOT.ExRootTreeReader(sig_chain)


back_nev = back_reader.GetEntries()
sig_nev = sig_reader.GetEntries()

back_cs = tree2array(back_chain, branches=["Event.CrossSection"]).astype(np.float32)
sig_cs = tree2array(sig_chain, branches=["Event.CrossSection"]).astype(np.float32)

maximums = {
    "Eta" : 5.0,
    "Phi" : 3.1,
    "DeltaEta" : 0.6,
    "DeltaPhi" : 0.6
}

legend_locs = {
    "EventScale" : 0,
    "JetPT" : 0,
    "JetEta" : 3,
    "JetPhi" : 0,
    "JetDeltaEta" : 4,
    "JetDeltaPhi" : 4,
    "JetMass" : 0,
    "JetCharge" : 2
}


for b in tqdm(print_vars.keys()):
    text = ""
    for l in print_vars[b]:
        if "DeltaEta" == l:
            text = r"Jet $|\Delta \eta|$"
        elif "DeltaPhi" == l:
            text = r"Jet $|\Delta \phi|$"
        elif "Eta" == l:
            text = r"Jet $|\eta|$"
        elif "Phi" == l:
            text = r"Jet $|\phi|$"
        elif "PT" == l:
            text = r"Jet $p_T$"
        else:
            text = "{} {}".format(b, l)
        back_cross_sec = np.array([])
        sig_cross_sec = np.array([])
        bck_data = tree2array(back_chain, branches=["{}.{}".format(b, l)])
        sig_data = tree2array(sig_chain, branches=["{}.{}".format(b, l)])
        bck_arr = np.array([])
        sig_arr = np.array([])
        idx = 0
        for entry in tqdm(bck_data):
            temp = back_cs[idx]
            idx += 1
            if not isinstance(entry, list) and not isinstance(entry, np.void):
                entry = [np.float32(entry)]
            elif isinstance(entry, np.void):
                entry = entry[0].astype(np.float32)
            bck_arr = np.append(bck_arr, entry, axis=0).astype(np.float32)
            back_cross_sec = np.append(back_cross_sec, np.repeat(temp, len(entry)), axis=0).astype(np.float32)
        idx = 0
        for entry in tqdm(sig_data):
            temp = sig_cs[idx]
            idx += 1
            if not isinstance(entry, list) and not isinstance(entry, np.void):
                entry = [np.float32(entry)]
            elif isinstance(entry, np.void):
                entry = entry[0].astype(np.float32)
            sig_arr = np.append(sig_arr, entry, axis=0).astype(np.float32)
            sig_cross_sec = np.append(sig_cross_sec, np.repeat(temp, len(entry)), axis=0).astype(np.float32)
        if l in ["DeltaEta", "DeltaPhi", "Phi", "Eta"]:
            sig_arr = np.abs(sig_arr)
            bck_arr = np.abs(bck_arr)
        max_hist = np.max(bck_arr)
        if max_hist < np.max(sig_arr):
            max_hist = np.max(sig_arr)
        min_hist = np.min(bck_arr)
        if min_hist > np.min(sig_arr):
            min_hist = np.min(sig_arr)
        if l in maximums.keys():
            max_hist = maximums[l]
        legloc = legend_locs["{}{}".format(b,l)]
        plt_2hist("{}_{}_NoWeighting".format(b, l), text, sig_arr, bck_arr, np.ones_like(sig_arr),
                  np.ones_like(bck_arr), num_bins=50, hist_max=max_hist, hist_min=min_hist, save_dir="Resolution_Histograms/NoLog_NoWeight/")
        plt_2hist("{}_{}_CrossSectionReweighted".format(b, l), text, sig_arr, bck_arr, np.ones_like(sig_cross_sec),
                  back_cross_sec, num_bins=50, hist_max=max_hist, hist_min=min_hist, save_dir="Resolution_Histograms/NoLog_Weight/")
        plt_2hist("{}_{}_Log_NoWeighting".format(b, l), text, sig_arr, bck_arr, np.ones_like(sig_arr),
                  np.ones_like(bck_arr), num_bins=50, hist_max=max_hist, hist_min=min_hist, legloc=legloc, log_plot=True, save_dir="Resolution_Histograms/Log_NoWeight/")
        plt_2hist("{}_{}_Log_CrossSectionReweighted".format(b, l), text, sig_arr, bck_arr, np.ones_like(sig_cross_sec),
                  back_cross_sec, num_bins=50, hist_max=max_hist, hist_min=min_hist, legloc=legloc, log_plot=True, save_dir="Resolution_Histograms/Log_Weight/")
        del bck_data
        del sig_data

# print("Downloading Background Jets...")
# bck_jet_eta = tree2array(back_chain, branches=["Jet.Eta"])
# print("Jet.Eta done.")
# bck_jet_phi = tree2array(back_chain, branches=["Jet.Phi"])
# print("Jet.Phi done.")
# bck_jet_deta = tree2array(back_chain, branches=["Jet.DeltaEta"])
# print("Jet.DeltaEta done.")
# bck_jet_dphi = tree2array(back_chain, branches=["Jet.DeltaPhi"])
# print("Jet.DeltaPhi done.")
# # bck_tower_eta = tree2array(back_chain, branches=["Tower.Eta"])
# # print("Tower.Eta done.")
# # bck_tower_phi = tree2array(back_chain, branches=["Tower.Phi"])
# # print("Tower.Phi done.")
# # bck_track_eta = tree2array(back_chain, branches=["Track.Eta"])
# # print("Track.Eta done.")
# # bck_track_phi = tree2array(back_chain, branches=["Track.Phi"])
# # print("Tower.Eta done.")
# print("Background Jets done.")
#
# bck_jet_ntower = np.array([])
# bck_jet_ntrack = np.array([])
#
# print("Downloading Signal Jets...")
# sig_jet_eta = tree2array(sig_chain, branches=["Jet.Eta"])
# print("Jet.Eta done.")
# sig_jet_phi = tree2array(sig_chain, branches=["Jet.Phi"])
# print("Jet.Phi done.")
# sig_jet_deta = tree2array(sig_chain, branches=["Jet.DeltaEta"])
# print("Jet.DeltaEta done.")
# sig_jet_dphi = tree2array(sig_chain, branches=["Jet.DeltaPhi"])
# print("Jet.DeltaPhi done.")
# # sig_tower_eta = tree2array(sig_chain, branches=["Tower.Eta"])
# # print("Tower.Eta done.")
# # sig_tower_phi = tree2array(sig_chain, branches=["Tower.Phi"])
# # print("Tower.Phi done.")
# # sig_track_eta = tree2array(sig_chain, branches=["Track.Eta"])
# # print("Track.Eta done.")
# # sig_track_phi = tree2array(sig_chain, branches=["Track.Phi"])
# # print("Track.Phi done.")
# print("Signal Jets done.")
#
# sig_jet_ntower = np.array([])
# sig_jet_ntrack = np.array([])
#
# back_cross = np.array([])
# sig_cross = np.array([])
#
# back_reader.UseBranch("Event").GetEntries()
# back_reader.UseBranch("Track").GetEntries()
# back_reader.UseBranch("Tower").GetEntries()
# back_weight = back_reader.UseBranch("Event.CrossSection")
# track_eta = back_reader.UseBranch("Track.Eta")
# track_phi = back_reader.UseBranch("Track.Phi")
#
# for evt in trange(0, back_nev):
#     back_reader.ReadEntry(evt)
#     print(back_weight.At(0))
#     print(track_eta.At(2))
#     #for ji in range(0, len(bck_jet_eta[evt])):


