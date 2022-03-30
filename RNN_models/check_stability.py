# import keras_preprocessing.sequence
import matplotlib.pyplot as plt
import numpy as np
# from ROOT import TMVA, TFile, TTree, TCut
import ROOT
# from subprocess import call
# from os.path import isfile
# import pandas as pd
# import random
#
# from tensorflow.python import pywrap_tensorflow as _pywrap_tensorflow
# from tensorflow.python.eager import context
# from keras.models import Model
# from keras.layers import Input
# from keras.layers import Dense
# from keras.layers import Concatenate
# from keras.layers import LSTM
# from keras.layers import Flatten
# from keras.layers import TimeDistributed
# import tensorflow as tf
# from keras.utils.vis_utils import plot_model
#
# from keras.preprocessing.sequence import pad_sequences
# from keras.preprocessing.timeseries import timeseries_dataset_from_array
# from keras.layers.convolutional import Conv2D
# from keras.layers.pooling import MaxPooling2D
# from keras.layers.merge import concatenate
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Activation
# from tensorflow.keras.optimizers import SGD
# import tensorflow as tf
# from tensorflow.keras.layers.experimental.preprocessing import Normalization
# from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm, trange
import numpy as np
import tqdm



from RNN_model import Tau_Model
from RNN_Data import RNN_Data
import root_numpy as rn
#from rootpy.plotting import Hist
from Plots import Plots

import matplotlib as mpl
from Mu_ATLAS_Utils import colors, colorseq, roc, roc_ratio, \
    binned_efficiency_ci
from flattener import Flattener
from scipy.stats import binned_statistic, binned_statistic_2d
from matplotlib.lines import Line2D

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
####


def plot_1_histogram(name, data, weight, num_bins):
    hist_max = np.max(data)
    hist_min = np.max(data)
    hist = ROOT.TH1D("{}".format(name), "{}".format(name), num_bins, hist_min, hist_max)
    rn.fill_hist(hist, data, weights=weight)
    canvas = ROOT.TCanvas("Hist_Data_{}".format(name), "Hist_Data_{}".format(name))
    canvas.Divide(1, 1)
    canvas.cd(1)
    integral = hist.Integral()
    if integral != 0.:
        hist.Scale(1/integral)
    hist.Draw("HIST")
    legend = ROOT.TLegend(0.05, 0.85, 0.2, 0.95)
    legend.AddEntry(hist, "{} Histogram".format(name))
    legend.Draw()
    canvas.Update()
    canvas.Print("{}.pdf".format(name))

def plot_2_histogram(name, sig_data, bck_data, sig_weight, bck_weight, num_bins, bins=False, hist_max=None, hist_min=None):
    if type(bins) == np.array or type(bins) == list:

        bck_hist = ROOT.TH1D("{}_BACK".format(name), "{}_BACK".format(name), num_bins, bins)
        sig_hist = ROOT.TH1D("{}_SIG".format(name), "{}_SIG".format(name), num_bins, bins)
    else:
        if np.max(bck_data) <= np.max(sig_data) and hist_max == None:
            hist_max = np.max(sig_data)
        elif hist_max == None:
            hist_max = np.max(bck_data)
        if np.min(bck_data) >= np.min(sig_data) and hist_min == None:
            hist_min = np.max(sig_data)
        elif hist_min == None:
            hist_min = np.max(bck_data)
        bck_hist = ROOT.TH1D("{}_BACK".format(name), "{}_BACK".format(name), num_bins, hist_min, hist_max)
        sig_hist = ROOT.TH1D("{}_SIG".format(name), "{}_SIG".format(name), num_bins, hist_min, hist_max)
    rn.fill_hist(bck_hist, bck_data, weights=bck_weight)
    rn.fill_hist(sig_hist, sig_data, weights=sig_weight)
    canvas = ROOT.TCanvas("Hist_Data_{}".format(name), "Hist_Data_{}".format(name))
    canvas.Divide(1, 1)
    canvas.cd(1)
    bck_integral = bck_hist.Integral()
    sig_integral = sig_hist.Integral()
    if bck_integral != 0. and sig_integral != 0.:
        bck_hist.Scale(1/bck_integral)
        sig_hist.Scale(1/sig_integral)
    bck_hist.Draw("HIST")
    sig_hist.SetLineColor(ROOT.kRed)
    sig_hist.Draw("HIST SAMES0")
    legend = ROOT.TLegend(0.05, 0.85, 0.2, 0.95)
    legend.AddEntry(bck_hist, "Back {} Histogram".format(name))
    legend.AddEntry(sig_hist, "Signal {} Histogram".format(name))
    legend.Draw()
    canvas.Update()
    canvas.Print("{}_back_sig.pdf".format(name))


def plt_2hist(name, sig_data, bck_data, sig_weight, bck_weight, num_bins, legend=False, var_name="", add_label="", save_dir=False, bins=False, log_plot=False, hist_max=None, hist_min=None):
    fig, ax = plt.subplots()
    if hist_max != None and hist_min != None:
        bin_edges = np.histogram_bin_edges(np.append(sig_data, bck_data), bins=num_bins, range=(hist_min, hist_max))
    else:
        bin_edges = np.histogram_bin_edges(np.append(sig_data, bck_data), bins=num_bins)
    ax.hist(sig_data, weights=sig_weight,
            color=colors["red"], label="Signal {}".format(add_label), bins=bin_edges, density=True, histtype="step")
    ax.hist(bck_data, weights=bck_weight,
            color=colors["blue"], label="Background {}".format(add_label), bins=bin_edges, density=True, histtype="step")
    if log_plot:
        ax.set_yscale("log")
    ax.autoscale()
    ax.set_ylabel("Norm. number of entries")
    ax.set_xlabel(var_name)
    if hist_max != None and hist_min != None:
        ax.set_xlim((hist_min, hist_max))
    if legend:
        ax.legend()
    y_lo, y_hi = ax.get_ylim()
    d = 0.40 * (y_hi - y_lo)
    ax.set_ylim(y_lo, y_hi + d)
    if save_dir:
        plt.savefig("{}{}.png".format(save_dir, name))
    return fig

def plt_2hist_prongcomparison(name, prong1_data, prong3_data, prong1_weight, prong3_weight, num_bins, legend=False, var_name="", add_label="", save_dir=False, bins=False, log_plot=False, hist_max=None, hist_min=None):
    fig, ax = plt.subplots()
    if hist_max != None and hist_min != None:
        bin_edges = np.histogram_bin_edges(np.append(prong1_data, prong3_data), bins=num_bins, range=(hist_min, hist_max))
    else:
        bin_edges = np.histogram_bin_edges(np.append(prong1_data, prong3_data), bins=num_bins)
    ax.hist(prong1_data, weights=prong1_weight,
            color=colors["red"], label="1-Prong {}".format(add_label), bins=bin_edges, density=True, histtype="step")
    ax.hist(prong3_data, weights=prong3_weight,
            color=colors["blue"], label="3-Prong {}".format(add_label), bins=bin_edges, density=True, histtype="step")
    if log_plot:
        ax.set_yscale("log")
    ax.autoscale()
    if hist_max != None and hist_min != None:
        ax.set_xlim((hist_min, hist_max))
    ax.set_ylabel("Norm. number of entries")
    ax.set_xlabel(var_name)
    if legend:
        ax.legend()
    y_lo, y_hi = ax.get_ylim()
    d = 0.05 * (y_hi - y_lo)
    ax.set_ylim(y_lo - d, y_hi + d)
    if save_dir:
        plt.savefig("{}{}.png".format(save_dir, name))
    return fig

def plot_graph(name, x_data, y_data, n_points):
    graph = ROOT.TGraph(n_points, x_data, y_data)
    canvas = ROOT.TCanvas("Hist_Data_{}".format(name), "Hist_Data_{}".format(name))
    canvas.SetLogy()
    canvas.Divide(1, 1)
    canvas.cd(1)
    graph.Draw("AC*")
    legend = ROOT.TLegend(0.05, 0.85, 0.2, 0.95)
    legend.AddEntry(graph, "{} Graph".format(name))
    legend.Draw()
    canvas.Update()
    canvas.Print("{}.pdf".format(name))

pile_up = True

if pile_up:
    from Mu_ATLAS_Plots import ScorePlot, FlattenerCutmapPlot, FlattenerEfficiencyPlot, EfficiencyPlot, RejectionPlot

else:
    from ATLAS_RNN_Plots import ScorePlot, FlattenerCutmapPlot, FlattenerEfficiencyPlot, EfficiencyPlot, RejectionPlot

#prefix = "NewCondCT_"
prefix = "CoreTrackCond_"

DataP1 = RNN_Data(1, False, "prong1_data", print_hists=False,
                  BacktreeFile=["{}0-1_background_wPU_tree_1-Prong".format(prefix), "{}1-1_background_wPU_tree_1-Prong".format(prefix), "{}2-1_background_wPU_tree_1-Prong".format(prefix),
                                "{}3-1_background_wPU_tree_1-Prong".format(prefix), "{}4-1_background_wPU_tree_1-Prong".format(prefix), "{}0-2_background_wPU_tree_1-Prong".format(prefix),
                                "{}1-2_background_wPU_tree_1-Prong".format(prefix), "{}2-2_background_wPU_tree_1-Prong".format(prefix), "{}3-2_background_wPU_tree_1-Prong".format(prefix),
                                "{}4-2_background_wPU_tree_1-Prong".format(prefix)]
                , BackTreeName=["{}0-1_background_wPU_tree".format(prefix), "{}1-1_background_wPU_tree".format(prefix), "{}2-1_background_wPU_tree".format(prefix),
                                "{}3-1_background_wPU_tree".format(prefix), "{}4-1_background_wPU_tree".format(prefix), "{}0-2_background_wPU_tree".format(prefix),
                                "{}1-2_background_wPU_tree".format(prefix), "{}2-2_background_wPU_tree".format(prefix), "{}3-2_background_wPU_tree".format(prefix),
                                "{}4-2_background_wPU_tree".format(prefix)]
                  , SignaltreeFile=["{}0_signal_wPU_tree_1-Prong".format(prefix), "{}1_signal_wPU_tree_1-Prong".format(prefix)],
                  SignalTreeName=["{}0_signal_wPU_tree".format(prefix), "{}1_signal_wPU_tree".format(prefix)], BackendPartOfTree="", SignalendPartOfTree="")

DataP3 = RNN_Data(3, False, "prong3_data", print_hists=False,
                  BacktreeFile=["{}0-1_background_wPU_tree_3-Prong".format(prefix), "{}1-1_background_wPU_tree_3-Prong".format(prefix), "{}2-1_background_wPU_tree_3-Prong".format(prefix),
                                "{}3-1_background_wPU_tree_3-Prong".format(prefix), "{}4-1_background_wPU_tree_3-Prong".format(prefix), "{}0-2_background_wPU_tree_3-Prong".format(prefix),
                                "{}1-2_background_wPU_tree_3-Prong".format(prefix), "{}2-2_background_wPU_tree_3-Prong".format(prefix), "{}3-2_background_wPU_tree_3-Prong".format(prefix),
                                "{}4-2_background_wPU_tree_3-Prong".format(prefix)]
                , BackTreeName=["{}0-1_background_wPU_tree".format(prefix), "{}1-1_background_wPU_tree".format(prefix), "{}2-1_background_wPU_tree".format(prefix),
                                "{}3-1_background_wPU_tree".format(prefix), "{}4-1_background_wPU_tree".format(prefix), "{}0-2_background_wPU_tree".format(prefix),
                                "{}1-2_background_wPU_tree".format(prefix), "{}2-2_background_wPU_tree".format(prefix), "{}3-2_background_wPU_tree".format(prefix),
                                "{}4-2_background_wPU_tree".format(prefix)]
                  , SignaltreeFile=["{}0_signal_wPU_tree_3-Prong".format(prefix), "{}1_signal_wPU_tree_3-Prong".format(prefix)],
                  SignalTreeName=["{}0_signal_wPU_tree".format(prefix), "{}1_signal_wPU_tree".format(prefix)], BackendPartOfTree="", SignalendPartOfTree="")

####PLOTTING FOR REPORT#####
plot_track_stuff = True

if plot_track_stuff:
    #Signal
    p1sig_core_ntrack = np.array([])
    p1sig_leading_coretrack_pt = np.array([])
    p1sig_weights = DataP1.cross_section[-DataP1.length_sig: -1]

    p3sig_core_ntrack = np.array([])
    p3sig_leading_coretrack_pt = np.array([])
    p3sig_weights = DataP3.cross_section[-DataP3.length_sig: -1]

    #Prong 1 Loop
    p1sig_tracks = [DataP1.untrans_track["track_PT"][-DataP1.length_sig: -1],
                    DataP1.untrans_track["track_deltaEta"][-DataP1.length_sig: -1],
                    DataP1.untrans_track["track_deltaPhi"][-DataP1.length_sig: -1]]

    nOutofBounds = 0
    nZeros = 0
    for idx in trange(0, len(p1sig_tracks[0])):
        ntrack = 0
        leading_pt = 0.
        for jdx in range(0, len(p1sig_tracks[0][idx])):
            deltaR = np.sqrt(p1sig_tracks[1][idx, jdx]**2 + p1sig_tracks[2][idx, jdx]**2)
            if deltaR < 0.2 and not np.isnan(p1sig_tracks[0][idx, jdx]):
                ntrack += 1
                pt = p1sig_tracks[0][idx, jdx]
                if pt > leading_pt:
                    leading_pt = pt
            elif deltaR > 0.6:
                nOutofBounds += 1
        if ntrack == 0 or ntrack > 6:
            nZeros += 1
        p1sig_core_ntrack = np.append(p1sig_core_ntrack, [ntrack], axis=0)
        p1sig_leading_coretrack_pt = np.append(p1sig_leading_coretrack_pt, [leading_pt], axis=0)


    print("Number of Jets with Zero Core Tracks in PRONG-1 SIGNAL is {}".format(nZeros))
    print("Number of tracks out of bounds for PRONG-1 SIGNAL is {}".format(nOutofBounds))


    #Prong 3 Loop
    p3sig_tracks = [DataP3.untrans_track["track_PT"][-DataP3.length_sig: -1],
                    DataP3.untrans_track["track_deltaEta"][-DataP3.length_sig: -1],
                    DataP3.untrans_track["track_deltaPhi"][-DataP3.length_sig: -1]]

    nOutofBounds = 0
    nZeros = 0
    for idx in trange(0, len(p3sig_tracks[0])):
        ntrack = 0
        leading_pt = 0.
        for jdx in range(0, len(p3sig_tracks[0][idx])):
            deltaR = np.sqrt(p3sig_tracks[1][idx, jdx]**2 + p3sig_tracks[2][idx, jdx]**2)
            if deltaR < 0.2 and not np.isnan(p3sig_tracks[0][idx, jdx]):
                ntrack += 1
                pt = p3sig_tracks[0][idx, jdx]
                if pt > leading_pt:
                    leading_pt = pt
            elif deltaR > 0.6:
                nOutofBounds += 1
        if ntrack < 3 or ntrack > 8:
            nZeros += 1
        p3sig_core_ntrack = np.append(p3sig_core_ntrack, [ntrack], axis=0)
        p3sig_leading_coretrack_pt = np.append(p3sig_leading_coretrack_pt, [leading_pt], axis=0)

    print("Number of Jets with Less than 3 Core Tracks in PRONG-3 SIGNAL is {}".format(nZeros))
    print("Number of tracks out of bounds for PRONG-3 SIGNAL is {}".format(nOutofBounds))

    #Background
    p1bck_core_ntrack = np.array([])
    p1bck_leading_coretrack_pt = np.array([])
    p1bck_weight = DataP1.cross_section[0: DataP1.length_bck]

    p3bck_core_ntrack = np.array([])
    p3bck_leading_coretrack_pt = np.array([])
    p3bck_weight = DataP3.cross_section[0: DataP3.length_bck]

    #Prong 1 Loop

    p1bck_tracks = [DataP1.untrans_track["track_PT"][0: DataP1.length_bck],
                    DataP1.untrans_track["track_deltaEta"][0: DataP1.length_bck],
                    DataP1.untrans_track["track_deltaPhi"][0: DataP1.length_bck]]

    p1sig_jetprops = [DataP1.untrans_jet["jet_f_cent"][-DataP1.length_sig:-1],
                      DataP1.untrans_jet["jet_max_deltaR"][-DataP1.length_sig:-1],
                      DataP1.untrans_jet["jet_frac_trEM_pt"][-DataP1.length_sig:-1]]
    p1bck_jetprops = [DataP1.untrans_jet["jet_f_cent"][0:DataP1.length_bck],
                      DataP1.untrans_jet["jet_max_deltaR"][0:DataP1.length_bck],
                      DataP1.untrans_jet["jet_frac_trEM_pt"][0:DataP1.length_bck]]

    p3sig_jetprops = [DataP3.untrans_jet["jet_f_cent"][-DataP3.length_sig:-1],
                      DataP3.untrans_jet["jet_max_deltaR"][-DataP3.length_sig:-1],
                      DataP3.untrans_jet["jet_frac_trEM_pt"][-DataP3.length_sig:-1]]
    p3bck_jetprops = [DataP3.untrans_jet["jet_f_cent"][0:DataP3.length_bck],
                      DataP3.untrans_jet["jet_max_deltaR"][0:DataP3.length_bck],
                      DataP3.untrans_jet["jet_frac_trEM_pt"][0:DataP3.length_bck]]

    nOutofBounds = 0
    nZeros = 0
    for idx in trange(0, len(p1bck_tracks[0])):
        ntrack = 0
        leading_pt = 0.
        for jdx in range(0, len(p1bck_tracks[0][idx])):
            deltaR = np.sqrt(p1bck_tracks[1][idx, jdx]**2 + p1bck_tracks[2][idx, jdx]**2)
            if deltaR < 0.2 and not np.isnan(p1bck_tracks[0][idx, jdx]):
                ntrack += 1
                pt = p1bck_tracks[0][idx, jdx]
                if pt > leading_pt:
                    leading_pt = pt
            elif deltaR > 0.6:
                nOutofBounds += 1
        if ntrack == 0 or ntrack > 6:
            nZeros += 1
        p1bck_core_ntrack = np.append(p1bck_core_ntrack, [ntrack], axis=0)
        p1bck_leading_coretrack_pt = np.append(p1bck_leading_coretrack_pt, [leading_pt], axis=0)

    print("Number of Jets with Zero Core Tracks in PRONG-1 BACKGROUND is {}".format(nZeros))
    print("Number of tracks out of bounds for PRONG-1 BACKGROUND is {}".format(nOutofBounds))

    #Prong 3 Loop
    p3bck_tracks = [DataP3.untrans_track["track_PT"][0: DataP3.length_bck],
                    DataP3.untrans_track["track_deltaEta"][0: DataP3.length_bck],
                    DataP3.untrans_track["track_deltaPhi"][0: DataP3.length_bck]]

    nOutofBounds = 0

    nZeros = 0
    for idx in trange(0, len(p3bck_tracks[0])):
        ntrack = 0
        leading_pt = 0.
        for jdx in range(0, len(p3bck_tracks[0][idx])):
            deltaR = np.sqrt(p3bck_tracks[1][idx, jdx]**2 + p3bck_tracks[2][idx, jdx]**2)
            if deltaR < 0.2 and not np.isnan(p3bck_tracks[0][idx, jdx]):
                ntrack += 1
                pt = p3bck_tracks[0][idx, jdx]
                if pt > leading_pt:
                    leading_pt = pt
            elif deltaR > 0.6:
                nOutofBounds += 1
        if ntrack < 3 or ntrack > 8:
            nZeros += 1
        p3bck_core_ntrack = np.append(p3bck_core_ntrack, [ntrack], axis=0)
        p3bck_leading_coretrack_pt = np.append(p3bck_leading_coretrack_pt, [leading_pt], axis=0)

    print("Number of Jets with Less than 3 Core Tracks in PRONG-3 BACKGROUND is {}".format(nZeros))
    print("Number of tracks out of bounds for PRONG-3 BACKGROUND is {}".format(nOutofBounds))

    print("sig : {} | bck : {}".format(np.shape(p3sig_core_ntrack), np.shape(p3bck_core_ntrack)))
    print("sig w: {} | bck w: {}".format(np.shape(p3sig_weights), np.shape(p3bck_weight)))

    plt_2hist("{}prong1_coretrack_multiplicity".format(prefix), p1sig_core_ntrack, p1bck_core_ntrack, p1sig_weights, p1bck_weight, 15, log_plot=True, hist_min=0, hist_max=20, legend=True, var_name="No. of Tracks within Jet", add_label=r"1-prong $\tau_{had-vis}$", save_dir="report_plots/")
    plt_2hist("{}prong3_coretrack_multiplicity".format(prefix), p3sig_core_ntrack, p3bck_core_ntrack, p3sig_weights, p3bck_weight, 15, log_plot=True, hist_min=0, hist_max=20, legend=True, var_name="No. of Tracks within Jet", add_label=r"3-prong $\tau_{had-vis}$", save_dir="report_plots/")

    plt_2hist("{}prong1_coreleadingtrack_pt".format(prefix), p1sig_leading_coretrack_pt, p1bck_leading_coretrack_pt, p1sig_weights, p1bck_weight, 30, log_plot=True, hist_min=0, hist_max=1000., legend=True, var_name=r"$p_T$ of Leading Core Track (GeV)", add_label=r"1-prong $\tau_{had-vis}$", save_dir="report_plots/")
    plt_2hist("{}prong3_coreleadingtrack_pt".format(prefix), p3sig_leading_coretrack_pt, p3bck_leading_coretrack_pt, p3sig_weights, p3bck_weight, 30, log_plot=True, hist_min=0, hist_max=1000., legend=True, var_name=r"$p_T$ of Leading Core Track (GeV)", add_label=r"3-prong $\tau_{had-vis}$", save_dir="report_plots/")

    #plot prong-1 prong-3 jet_pt comparison

    p1sig_jetpt = DataP1.sig_pt
    p3sig_jetpt = DataP3.sig_pt
    p1sig_ptreweight = DataP1.pt_weights[-DataP1.length_sig:-1]
    p3sig_ptreweight = DataP3.pt_weights[-DataP3.length_sig:-1]

    p1bck_jetpt = DataP1.bck_pt
    p3bck_jetpt = DataP3.bck_pt
    p1bck_ptreweight = DataP1.pt_weights[0:DataP1.length_bck]
    p3bck_ptreweight = DataP3.pt_weights[0:DataP3.length_bck]

    #pt_reweight
    plt_2hist("{}prong1_jetpt".format(prefix), p1sig_jetpt, p1bck_jetpt, np.ones_like(p1sig_weights), np.ones_like(p1bck_weight), 30, hist_min=0, hist_max=1000., legend=True, log_plot=True, var_name=r"Jet $p_T$ (GeV)", add_label=r"1-prong $\tau_{had-vis}$", save_dir="report_plots/")
    plt_2hist("{}prong3_jetpt".format(prefix), p3sig_jetpt, p3bck_jetpt, np.ones_like(p3sig_weights), np.ones_like(p3bck_weight), 30, hist_min=0, hist_max=1000., legend=True, log_plot=True, var_name=r"Jet $p_T$ (GeV)", add_label=r"3-prong $\tau_{had-vis}$", save_dir="report_plots/")

    plt_2hist("{}prong1_jetpt_forptreweight".format(prefix), p1sig_jetpt, p1bck_jetpt, p1sig_ptreweight, p1bck_ptreweight, 30, hist_min=0, hist_max=1000., legend=True, log_plot=True, var_name=r"Jet $p_T$ (GeV)", add_label=r"1-prong $\tau_{had-vis}$", save_dir="report_plots/")
    plt_2hist("{}prong3_jetpt_forptreweight".format(prefix), p3sig_jetpt, p3bck_jetpt, p3sig_ptreweight, p3bck_ptreweight, 30, hist_min=0, hist_max=1000., legend=True, log_plot=True, var_name=r"Jet $p_T$ (GeV)", add_label=r"3-prong $\tau_{had-vis}$", save_dir="report_plots/")

    #hl_var plots
    p1sig_trans_LCpt = DataP1.input_jet[-DataP1.length_sig:-1, 2]
    p3sig_trans_LCpt = DataP3.input_jet[-DataP3.length_sig:-1, 2]
    p1sig_untrans_LCpt = DataP1.untrans_jet["jet_PT_LC_scale"][-DataP1.length_sig:-1]
    p3sig_untrans_LCpt = DataP3.untrans_jet["jet_PT_LC_scale"][-DataP3.length_sig:-1]

    p1bck_trans_LCpt = DataP1.input_jet[0:DataP1.length_bck, 2]
    p3bck_trans_LCpt = DataP3.input_jet[0:DataP3.length_bck, 2]
    p1bck_untrans_LCpt = DataP1.untrans_jet["jet_PT_LC_scale"][0:DataP1.length_bck]
    p3bck_untrans_LCpt = DataP3.untrans_jet["jet_PT_LC_scale"][0:DataP3.length_bck]

    p1sig_trans_fcent = DataP1.input_jet[-DataP1.length_sig:-1, 3]
    p3sig_trans_fcent = DataP3.input_jet[-DataP3.length_sig:-1, 3]
    p1sig_untrans_fcent = DataP1.untrans_jet["jet_f_cent"][-DataP1.length_sig:-1]
    p3sig_untrans_fcent = DataP3.untrans_jet["jet_f_cent"][-DataP3.length_sig:-1]

    p1bck_trans_fcent = DataP1.input_jet[0:DataP1.length_bck, 3]
    p3bck_trans_fcent = DataP3.input_jet[0:DataP3.length_bck, 3]
    p1bck_untrans_fcent = DataP1.untrans_jet["jet_f_cent"][0:DataP1.length_bck]
    p3bck_untrans_fcent = DataP3.untrans_jet["jet_f_cent"][0:DataP3.length_bck]

    p1sig_trans_ftrackiso = DataP1.input_jet[-DataP1.length_sig:-1, 6]
    p3sig_trans_ftrackiso = DataP3.input_jet[-DataP3.length_sig:-1, 6]
    p1sig_untrans_ftrackiso = DataP1.untrans_jet["jet_Ftrack_Iso"][-DataP1.length_sig:-1]
    p3sig_untrans_ftrackiso = DataP3.untrans_jet["jet_Ftrack_Iso"][-DataP3.length_sig:-1]

    p1bck_trans_ftrackiso = DataP1.input_jet[0:DataP1.length_bck, 6]
    p3bck_trans_ftrackiso = DataP3.input_jet[0:DataP3.length_bck, 6]
    p1bck_untrans_ftrackiso = DataP1.untrans_jet["jet_Ftrack_Iso"][0:DataP1.length_bck]
    p3bck_untrans_ftrackiso = DataP3.untrans_jet["jet_Ftrack_Iso"][0:DataP3.length_bck]

    p1sig_trans_leadcoretrack = DataP1.input_track[-DataP1.length_sig:-1, 0, 0]
    p3sig_trans_leadcoretrack = DataP3.input_track[-DataP3.length_sig:-1, 0, 0]

    p1bck_trans_leadcoretrack = DataP1.input_track[0:DataP1.length_bck, 0, 0]
    p3bck_trans_leadcoretrack = DataP3.input_track[0:DataP3.length_bck, 0, 0]

    ### LeadCoreTrack Pt
    plt_2hist("trans_{}prong1_LeadCoreTrack_Pt".format(prefix), p1sig_trans_leadcoretrack, p1bck_trans_leadcoretrack, p1sig_ptreweight, p1bck_ptreweight, 30, log_plot=True, legend=True, var_name=r"Jet Lead Core Track $p^{LC}_T$ (GeV)", add_label=r"1-prong $\tau_{had-vis}$", save_dir="report_plots/")
    plt_2hist("untrans_{}prong1_LeadCoreTrack_Pt".format(prefix), p1sig_leading_coretrack_pt, p1bck_leading_coretrack_pt, p1sig_ptreweight, p1bck_ptreweight, 30, hist_min=0, hist_max=1000., log_plot=True, legend=True, var_name=r"Jet Lead Core Track $p^{LC}_T$ (GeV)", add_label=r"1-prong $\tau_{had-vis}$", save_dir="report_plots/")

    plt_2hist("trans_{}prong3_LeadCoreTrack_Pt".format(prefix), p3sig_trans_leadcoretrack, p3bck_trans_leadcoretrack, p3sig_ptreweight, p3bck_ptreweight, 30, log_plot=True, legend=True, var_name=r"Jet Lead Core Track $p^{LC}_T$ (GeV)", add_label=r"3-prong $\tau_{had-vis}$", save_dir="report_plots/")
    plt_2hist("untrans_{}prong3_LeadCoreTrack_Pt".format(prefix), p3sig_leading_coretrack_pt, p3bck_leading_coretrack_pt, p3sig_ptreweight, p3bck_ptreweight, 30, hist_min=0, hist_max=1000., log_plot=True, legend=True, var_name=r"Jet Lead Core Track $p^{LC}_T$ (GeV)", add_label=r"3-prong $\tau_{had-vis}$", save_dir="report_plots/")


    ### LCpt
    plt_2hist("trans_{}prong1_LCpt".format(prefix), p1sig_trans_LCpt, p1bck_trans_LCpt, p1sig_ptreweight, p1bck_ptreweight, 30, hist_min=-25., hist_max=25., log_plot=True, legend=True, var_name=r"Jet $p^{LC}_T$ (GeV)", add_label=r"1-prong $\tau_{had-vis}$", save_dir="report_plots/")
    plt_2hist("untrans_{}prong1_LCpt".format(prefix), p1sig_untrans_LCpt, p1bck_untrans_LCpt, p1sig_ptreweight, p1bck_ptreweight, 30, hist_min=-25., hist_max=25., log_plot=True, legend=True, var_name=r"Jet $p^{LC}_T$ (GeV)", add_label=r"1-prong $\tau_{had-vis}$", save_dir="report_plots/")

    plt_2hist("trans_{}prong3_LCpt".format(prefix), p3sig_trans_LCpt, p3bck_trans_LCpt, p3sig_ptreweight, p3bck_ptreweight, 30, hist_min=-25., hist_max=25., log_plot=True, legend=True, var_name=r"Jet $p^{LC}_T$ (GeV)", add_label=r"3-prong $\tau_{had-vis}$", save_dir="report_plots/")
    plt_2hist("untrans_{}prong3_LCpt".format(prefix), p3sig_untrans_LCpt, p3bck_untrans_LCpt, p3sig_ptreweight, p3bck_ptreweight, 30, hist_min=-25., hist_max=25., log_plot=True, legend=True, var_name=r"Jet $p^{LC}_T$ (GeV)", add_label=r"3-prong $\tau_{had-vis}$", save_dir="report_plots/")

    ### fcent
    plt_2hist("trans_{}prong1_fcent".format(prefix), p1sig_trans_fcent, p1bck_trans_fcent, p1sig_ptreweight,
              p1bck_ptreweight, 30, log_plot=True, legend=True, var_name=r"Jet $f_{cent}$", add_label=r"1-prong $\tau_{had-vis}$",
              save_dir="report_plots/")
    plt_2hist("untrans_{}prong1_fcent".format(prefix), p1sig_untrans_fcent, p1bck_untrans_fcent, p1sig_ptreweight,
              p1bck_ptreweight, 30, log_plot=True, hist_min=0., hist_max=200., legend=True, var_name=r"Jet $f_{cent}$", add_label=r"1-prong $\tau_{had-vis}$",
              save_dir="report_plots/")

    plt_2hist("trans_{}prong3_fcent".format(prefix), p3sig_trans_fcent, p3bck_trans_fcent, p3sig_ptreweight,
              p3bck_ptreweight, 30, log_plot=True, legend=True, var_name=r"Jet $f_{cent}$", add_label=r"3-prong $\tau_{had-vis}$",
              save_dir="report_plots/")
    plt_2hist("untrans_{}prong3_fcent".format(prefix), p3sig_untrans_fcent, p3bck_untrans_fcent, p3sig_ptreweight,
              p3bck_ptreweight, 30, log_plot=True, hist_min=0., hist_max=200., legend=True, var_name=r"Jet $f_{cent}$", add_label=r"3-prong $\tau_{had-vis}$",
              save_dir="report_plots/")

    ### Ftrack_Iso
    plt_2hist("trans_{}prong1_Ftrack_Iso".format(prefix), p1sig_trans_ftrackiso, p1bck_trans_ftrackiso, p1sig_ptreweight,
              p1bck_ptreweight, 30, log_plot=True, legend=True, var_name=r"Jet $f^{track}_{iso}$", add_label=r"1-prong $\tau_{had-vis}$",
              save_dir="report_plots/")
    plt_2hist("untrans_{}prong1_Ftrack_Iso".format(prefix), p1sig_untrans_ftrackiso, p1bck_untrans_ftrackiso, p1sig_ptreweight,
              p1bck_ptreweight, 30, log_plot=True, legend=True, var_name=r"Jet $f^{track}_{iso}$", add_label=r"1-prong $\tau_{had-vis}$",
              save_dir="report_plots/")

    plt_2hist("trans_{}prong3_Ftrack_Iso".format(prefix), p3sig_trans_ftrackiso, p3bck_trans_ftrackiso, p3sig_ptreweight,
              p3bck_ptreweight, 30, log_plot=True, legend=True, var_name=r"Jet $f^{track}_{iso}$", add_label=r"3-prong $\tau_{had-vis}$",
              save_dir="report_plots/")
    plt_2hist("untrans_{}prong3_Ftrack_Iso".format(prefix), p3sig_untrans_ftrackiso, p3bck_untrans_ftrackiso, p3sig_ptreweight,
              p3bck_ptreweight, 30, log_plot=True, legend=True, var_name=r"Jet $f^{track}_{iso}$", add_label=r"3-prong $\tau_{had-vis}$",
              save_dir="report_plots/")


    #Prongness Comparison
    plt_2hist_prongcomparison("signal_{}jet_ncoretrack_prong_comparison".format(prefix), p1sig_core_ntrack, p3sig_core_ntrack, p1sig_weights, p3sig_weights, 15, log_plot=True, legend=True, var_name=r"Signal No. of Tracks within Jet", add_label=r"Signal $\tau_{had-vis}$", save_dir="report_plots/")
    plt_2hist_prongcomparison("signal_{}jet_f_cent_prong_comparison".format(prefix), p1sig_jetprops[0], p3sig_jetprops[0], p1sig_weights, p3sig_weights, 30, log_plot=True, hist_min=0., hist_max=200., legend=True, var_name=r"Signal Jets $f_{cent}$", add_label=r"Signal $\tau_{had-vis}$", save_dir="report_plots/")
    plt_2hist_prongcomparison("signal_{}jet_max_deltaR_prong_comparison".format(prefix), p1sig_jetprops[1], p3sig_jetprops[1], p1sig_weights, p3sig_weights, 30, log_plot=True, legend=True, var_name=r"Signal Jets $\Delta R_{max}$", add_label=r"Signal $\tau_{had-vis}$", save_dir="report_plots/")
    plt_2hist_prongcomparison("signal_{}jet_frac_trEM_pt_prong_comparison".format(prefix), p1sig_jetprops[2], p3sig_jetprops[2], p1sig_weights, p3sig_weights, 20, log_plot=True, hist_min=0, hist_max=10., legend=True, var_name=r"Signal Jets $p^{EM+track}_{T}/p_{T}$", add_label=r"Signal $\tau_{had-vis}$", save_dir="report_plots/")

    plt_2hist_prongcomparison("background_{}jet_ncoretrack_prong_comparison".format(prefix), p1bck_core_ntrack, p3bck_core_ntrack, p1bck_weight, p3bck_weight, 15, log_plot=True, legend=True, var_name=r"Background No. of Tracks within Jet", add_label=r"Background $\tau_{had-vis}$", save_dir="report_plots/")
    plt_2hist_prongcomparison("background_{}jet_f_cent_prong_comparison".format(prefix), p1bck_jetprops[0], p3bck_jetprops[0], p1bck_weight, p3bck_weight, 30, log_plot=True, hist_min=0., hist_max=200., legend=True, var_name=r"Background Jets $f_{cent}$", add_label=r"Background $\tau_{had-vis}$", save_dir="report_plots/")
    plt_2hist_prongcomparison("background_{}jet_max_deltaR_prong_comparison".format(prefix), p1bck_jetprops[1], p3bck_jetprops[1], p1bck_weight, p3bck_weight, 30, log_plot=True, legend=True, var_name=r"Background Jets $\Delta R_{max}$", add_label=r"Background $\tau_{had-vis}$", save_dir="report_plots/")
    plt_2hist_prongcomparison("background_{}jet_frac_trEM_pt_prong_comparison".format(prefix), p1bck_jetprops[2], p3bck_jetprops[2], p1bck_weight, p3bck_weight, 20, log_plot=True, hist_min=0, hist_max=10., legend=True, var_name=r"Background Jets $p^{EM+track}_{T}/p_{T}$", add_label=r"Background $\tau_{had-vis}$", save_dir="report_plots/")



  #  input("next...")
####TRAIN NETWORKS######

do_same_seed = True
do_random_seed = True
do_prong1 = True
do_prong3 = True

#if print_hist:

#    DataP1.plot_hists()
if do_prong1:
    if do_random_seed:
        Prong1Model = []
        train_real_y_p1 = []
        train_pred_y_p1 = []
        train_weights_p1 = []
        real_y_p1 = []
        pred_y_p1 = []
        jet_pt_p1 = []
        Prong1Plots = []
        prong1_rejvseff = []
        fig, ax = plt.subplots()
        prong1_eval_losses = np.array([])
        for i in trange(0, 40):
            p1_jet_input = np.delete(DataP1.input_jet[:, 1:12], 9, 1)

            Prong1Model.append(Tau_Model(1, [DataP1.input_track[:,0:10,:], DataP1.input_tower[:,0:6,:], p1_jet_input], DataP1.sig_pt, DataP1.bck_pt, DataP1.jet_pt, DataP1.Ytrain, DataP1.new_weights, DataP1.cross_section, DataP1.mu, kinematic_vars={"jet_Eta" : DataP1.input_jet[:, 12], "jet_Phi" : DataP1.input_jet[:, 13]}))
            Prong1Model[i].Model_Fit(256, 100, 0.4, model=Prong1Model[i].RNNmodel, inputs=Prong1Model[i].inputs, addition_savename="{}RandomSeeded{}_".format(prefix,i))
            #Prong1Model[i].load_model("{}RandomSeeded{}_RNN_Model_Prong-1.h5".format(prefix, i))
            Prong1Model[i].evaluate_model(Prong1Model[i].eval_inputs, Prong1Model[i].eval_y, Prong1Model[i].eval_w, Prong1Model[i].RNNmodel)
            tr, tp = Prong1Model[i].get_train_scores(Prong1Model[i].RNNmodel, Prong1Model[i].inputs)
            train_real_y_p1.append(tr)
            train_pred_y_p1.append(tp)
            train_weights_p1.append(Prong1Model[i].w_train)
            Prong1Model[i].evaluate_model(Prong1Model[i].eval_inputs, Prong1Model[i].eval_y, Prong1Model[i].eval_w, Prong1Model[i].RNNmodel)

            prong1_eval_losses = np.append(prong1_eval_losses, [Prong1Model[i].eval_results[0]])

            ry, py, jpt = Prong1Model[i].predict(Prong1Model[i].RNNmodel, Prong1Model[i].eval_inputs)
            real_y_p1.append(ry)
            pred_y_p1.append(py)
            jet_pt_p1.append(jpt)
            Prong1Plots.append(Plots( "{}Prong1Plots_randomseed_{}".format(prefix, i), real_y_p1[i], pred_y_p1[i], Prong1Model[i].eval_w, train_real_y_p1[i], train_pred_y_p1[i], train_weights_p1[i],
                                      [i]))
            prong1_rejvseff.append(Prong1Plots[i].plot_rej_vs_eff("{}ROC_Prong1_randomseed_{}".format(prefix, i), "ROC_curves/", do_train=True))
            plt.close(prong1_rejvseff[i])
            ax.plot(Prong1Plots[i].eff, Prong1Plots[i].rej, "-", label="Test {} (Evaluation Sample)".format(i))
        ax.set_xlim((0., 1.))
        ax.set_ylim((1., 1e4))
        ax.set_yscale("log")
        ax.set_xlabel("Signal efficiency", x=1, ha="right")
        ax.set_ylabel("Background rejection", y=1, ha="right")
        #ax.legend(fontsize='xx-small')
        plt.savefig("ROC_curves/{}P1_Test_RandomSeed_Stability.png".format(prefix))
        #plt.show()
        plt.close(fig)
        fig_train, ax_train = plt.subplots()
        for i in trange(0, 40):
            ax_train.plot(Prong1Plots[i].eff_train, Prong1Plots[i].rej_train, "--", label="Test {} (Training Sample)".format(i))
        ax_train.set_xlim((0., 1.))
        ax_train.set_ylim((1., 1e4))
        ax_train.set_yscale("log")
        ax_train.set_xlabel("Signal efficiency", x=1, ha="right")
        ax_train.set_ylabel("Background rejection", y=1, ha="right")
        plt.savefig("ROC_curves/{}P1_Train_RandomSeed_Stability.png".format(prefix))
        #plt.show()
        plt.close(fig_train)
        fig_all, ax_all = plt.subplots()
        for i in trange(0, 40):
            ax_all.plot(Prong1Plots[i].eff, Prong1Plots[i].rej, "-", color="g", label="Test {} (Evaluation Sample)".format(i))
            ax_all.plot(Prong1Plots[i].eff_train, Prong1Plots[i].rej_train, "-", color="b", label="Test {} (Training Sample)".format(i))
        ax_all.set_xlim((0., 1.))
        ax_all.set_ylim((1., 1e4))
        ax_all.set_yscale("log")
        ax_all.set_xlabel("Signal efficiency", x=1, ha="right")
        ax_all.set_ylabel("Background rejection", y=1, ha="right")
        all_p1_lines = [Line2D([0], [0], color="g", lw=4),
                        Line2D([0], [0], color="b", lw=4)]
        ax_all.legend(all_p1_lines, ["Evaluation ROC Curves", "Training ROC Curves"])
        plt.savefig("ROC_curves/{}P1_ALL_RandomSeed_Stability.png".format(prefix))
        #plt.show()
        plt.close(fig_all)
        del Prong1Model

        loss_fig, loss_ax = plt.subplots()
        loss_ax.hist(prong1_eval_losses, color=colors["red"], label="Loss from Trained Models")
        loss_ax.set_ylabel("Number of Models")
        loss_ax.set_xlabel("Binary Cross-Entropy Loss")
        loss_ax.legend()
        loss_ax.autoscale()
        plt.savefig("report_plots/{}prong1_models_histogrammed_losses.png".format(prefix))
        plt.close(loss_fig)

        del loss_fig, loss_ax


    if do_same_seed:
        p1_jet_input = np.delete(DataP1.input_jet[:, 1:12], 9, 1)

        Prong1Model_1 = Tau_Model(1, [DataP1.input_track[:,0:10,:], DataP1.input_tower[:,0:6,:], p1_jet_input], DataP1.sig_pt, DataP1.bck_pt, DataP1.jet_pt, DataP1.Ytrain, DataP1.new_weights, DataP1.cross_section, DataP1.mu, kinematic_vars={"jet_Eta" : DataP1.input_jet[:, 12], "jet_Phi" : DataP1.input_jet[:, 13]}, shuffle_seed=1521)
        Prong1Model_2 = Tau_Model(1, [DataP1.input_track[:,0:10,:], DataP1.input_tower[:,0:6,:], p1_jet_input], DataP1.sig_pt, DataP1.bck_pt, DataP1.jet_pt, DataP1.Ytrain, DataP1.new_weights, DataP1.cross_section, DataP1.mu, kinematic_vars={"jet_Eta" : DataP1.input_jet[:, 12], "jet_Phi" : DataP1.input_jet[:, 13]}, shuffle_seed=1521)
        Prong1Model_3 = Tau_Model(1, [DataP1.input_track[:,0:10,:], DataP1.input_tower[:,0:6,:], p1_jet_input], DataP1.sig_pt, DataP1.bck_pt, DataP1.jet_pt, DataP1.Ytrain, DataP1.new_weights, DataP1.cross_section, DataP1.mu, kinematic_vars={"jet_Eta" : DataP1.input_jet[:, 12], "jet_Phi" : DataP1.input_jet[:, 13]}, shuffle_seed=1521)
        Prong1Model_4 = Tau_Model(1, [DataP1.input_track[:,0:10,:], DataP1.input_tower[:,0:6,:], p1_jet_input], DataP1.sig_pt, DataP1.bck_pt, DataP1.jet_pt, DataP1.Ytrain, DataP1.new_weights, DataP1.cross_section, DataP1.mu, kinematic_vars={"jet_Eta" : DataP1.input_jet[:, 12], "jet_Phi" : DataP1.input_jet[:, 13]}, shuffle_seed=1521)
        Prong1Model_5 = Tau_Model(1, [DataP1.input_track[:,0:10,:], DataP1.input_tower[:,0:6,:], p1_jet_input], DataP1.sig_pt, DataP1.bck_pt, DataP1.jet_pt, DataP1.Ytrain, DataP1.new_weights, DataP1.cross_section, DataP1.mu, kinematic_vars={"jet_Eta" : DataP1.input_jet[:, 12], "jet_Phi" : DataP1.input_jet[:, 13]}, shuffle_seed=1521)

        Prong1Model_1.Model_Fit(256, 100, 0.4, model=Prong1Model_1.RNNmodel, inputs=Prong1Model_1.inputs, addition_savename="{}1_".format(prefix))
        Prong1Model_2.Model_Fit(256, 100, 0.4, model=Prong1Model_2.RNNmodel, inputs=Prong1Model_2.inputs, addition_savename="{}2_".format(prefix))
        Prong1Model_3.Model_Fit(256, 100, 0.4, model=Prong1Model_3.RNNmodel, inputs=Prong1Model_3.inputs, addition_savename="{}3_".format(prefix))
        Prong1Model_4.Model_Fit(256, 100, 0.4, model=Prong1Model_4.RNNmodel, inputs=Prong1Model_4.inputs, addition_savename="{}4_".format(prefix))
        Prong1Model_5.Model_Fit(256, 100, 0.4, model=Prong1Model_5.RNNmodel, inputs=Prong1Model_5.inputs, addition_savename="{}5_".format(prefix))

        #Prong1Model_1.load_model("{}RNN_Model_Prong-1.h5".format("{}1_".format(prefix)))
        #Prong1Model_2.load_model("{}RNN_Model_Prong-1.h5".format("{}2_".format(prefix)))
        #Prong1Model_3.load_model("{}RNN_Model_Prong-1.h5".format("{}3_".format(prefix)))
        #Prong1Model_4.load_model("{}RNN_Model_Prong-1.h5".format("{}4_".format(prefix)))
        #Prong1Model_5.load_model("{}RNN_Model_Prong-1.h5".format("{}5_".format(prefix)))

        Prong1Model_1.evaluate_model(Prong1Model_1.eval_inputs, Prong1Model_1.eval_y, Prong1Model_1.eval_w, Prong1Model_1.RNNmodel)
        Prong1Model_2.evaluate_model(Prong1Model_2.eval_inputs, Prong1Model_2.eval_y, Prong1Model_2.eval_w, Prong1Model_2.RNNmodel)
        Prong1Model_3.evaluate_model(Prong1Model_3.eval_inputs, Prong1Model_3.eval_y, Prong1Model_3.eval_w, Prong1Model_3.RNNmodel)
        Prong1Model_4.evaluate_model(Prong1Model_4.eval_inputs, Prong1Model_4.eval_y, Prong1Model_4.eval_w, Prong1Model_4.RNNmodel)
        Prong1Model_5.evaluate_model(Prong1Model_5.eval_inputs, Prong1Model_5.eval_y, Prong1Model_5.eval_w, Prong1Model_5.RNNmodel)

        train_real_y_p1_1, train_pred_y_p1_1 = Prong1Model_1.get_train_scores(Prong1Model_1.RNNmodel, Prong1Model_1.inputs)
        train_real_y_p1_2, train_pred_y_p1_2 = Prong1Model_2.get_train_scores(Prong1Model_2.RNNmodel, Prong1Model_2.inputs)
        train_real_y_p1_3, train_pred_y_p1_3 = Prong1Model_3.get_train_scores(Prong1Model_3.RNNmodel, Prong1Model_3.inputs)
        train_real_y_p1_4, train_pred_y_p1_4 = Prong1Model_4.get_train_scores(Prong1Model_4.RNNmodel, Prong1Model_4.inputs)
        train_real_y_p1_5, train_pred_y_p1_5 = Prong1Model_5.get_train_scores(Prong1Model_5.RNNmodel, Prong1Model_5.inputs)

        train_weights_p1_1 = Prong1Model_1.w_train
        train_weights_p1_2 = Prong1Model_2.w_train
        train_weights_p1_3 = Prong1Model_3.w_train
        train_weights_p1_4 = Prong1Model_4.w_train
        train_weights_p1_5 = Prong1Model_5.w_train

        Prong1Model_1.evaluate_model(Prong1Model_1.eval_inputs, Prong1Model_1.eval_y, Prong1Model_1.eval_w, Prong1Model_1.RNNmodel)
        Prong1Model_2.evaluate_model(Prong1Model_2.eval_inputs, Prong1Model_2.eval_y, Prong1Model_2.eval_w, Prong1Model_2.RNNmodel)
        Prong1Model_3.evaluate_model(Prong1Model_3.eval_inputs, Prong1Model_3.eval_y, Prong1Model_3.eval_w, Prong1Model_3.RNNmodel)
        Prong1Model_4.evaluate_model(Prong1Model_4.eval_inputs, Prong1Model_4.eval_y, Prong1Model_4.eval_w, Prong1Model_4.RNNmodel)
        Prong1Model_5.evaluate_model(Prong1Model_5.eval_inputs, Prong1Model_5.eval_y, Prong1Model_5.eval_w, Prong1Model_5.RNNmodel)

        real_y_p1_1, pred_y_p1_1, jet_pt_p1_1 = Prong1Model_1.predict(Prong1Model_1.RNNmodel, Prong1Model_1.eval_inputs)
        real_y_p1_2, pred_y_p1_2, jet_pt_p1_2 = Prong1Model_2.predict(Prong1Model_2.RNNmodel, Prong1Model_2.eval_inputs)
        real_y_p1_3, pred_y_p1_3, jet_pt_p1_3 = Prong1Model_3.predict(Prong1Model_3.RNNmodel, Prong1Model_3.eval_inputs)
        real_y_p1_4, pred_y_p1_4, jet_pt_p1_4 = Prong1Model_4.predict(Prong1Model_4.RNNmodel, Prong1Model_4.eval_inputs)
        real_y_p1_5, pred_y_p1_5, jet_pt_p1_5 = Prong1Model_5.predict(Prong1Model_5.RNNmodel, Prong1Model_5.eval_inputs)



        Prong1Plots_1 = Plots( "{}Prong1Plots_1".format(prefix), real_y_p1_1, pred_y_p1_1, Prong1Model_1.eval_w, train_real_y_p1_1, train_pred_y_p1_1, train_weights_p1_1, jet_pt_p1_1)
        Prong1Plots_2 = Plots( "{}Prong1Plots_2".format(prefix), real_y_p1_2, pred_y_p1_2, Prong1Model_2.eval_w, train_real_y_p1_2, train_pred_y_p1_2, train_weights_p1_2, jet_pt_p1_2)
        Prong1Plots_3 = Plots( "{}Prong1Plots_3".format(prefix), real_y_p1_3, pred_y_p1_3, Prong1Model_3.eval_w, train_real_y_p1_3, train_pred_y_p1_3, train_weights_p1_3, jet_pt_p1_3)
        Prong1Plots_4 = Plots( "{}Prong1Plots_4".format(prefix), real_y_p1_4, pred_y_p1_4, Prong1Model_4.eval_w, train_real_y_p1_4, train_pred_y_p1_4, train_weights_p1_4, jet_pt_p1_4)
        Prong1Plots_5 = Plots( "{}Prong1Plots_5".format(prefix), real_y_p1_5, pred_y_p1_5, Prong1Model_5.eval_w, train_real_y_p1_5, train_pred_y_p1_5, train_weights_p1_5, jet_pt_p1_5)

        prong1_rejveff_1 = Prong1Plots_1.plot_rej_vs_eff("{}ROC_Prong1_1".format(prefix), "ROC_curves/", do_train=True)
        prong1_rejveff_2 = Prong1Plots_2.plot_rej_vs_eff("{}ROC_Prong1_2".format(prefix), "ROC_curves/", do_train=True)
        prong1_rejveff_3 = Prong1Plots_3.plot_rej_vs_eff("{}ROC_Prong1_3".format(prefix), "ROC_curves/", do_train=True)
        prong1_rejveff_4 = Prong1Plots_4.plot_rej_vs_eff("{}ROC_Prong1_4".format(prefix), "ROC_curves/", do_train=True)
        prong1_rejveff_5 = Prong1Plots_5.plot_rej_vs_eff("{}ROC_Prong1_5".format(prefix), "ROC_curves/", do_train=True)

        #plt.show()

        fig, ax = plt.subplots()
        ax.plot(Prong1Plots_1.eff, Prong1Plots_1.rej, "-", color='r', label="First Test (Evaluation Sample)")
        ax.plot(Prong1Plots_1.eff_train, Prong1Plots_1.rej_train, "--", color='r', label="First Test (Training Sample)")

        ax.plot(Prong1Plots_2.eff, Prong1Plots_2.rej, "-", color='b', label="Second Test (Evaluation Sample)")
        ax.plot(Prong1Plots_2.eff_train, Prong1Plots_2.rej_train, "--", color='b', label="Second Test (Training Sample)")

        ax.plot(Prong1Plots_3.eff, Prong1Plots_3.rej, "-", color='g', label="Third Test (Evaluation Sample)")
        ax.plot(Prong1Plots_3.eff_train, Prong1Plots_3.rej_train, "--", color='g', label="Third Test (Training Sample)")

        ax.plot(Prong1Plots_4.eff, Prong1Plots_4.rej, "-", color='y', label="Fourth Test (Evaluation Sample)")
        ax.plot(Prong1Plots_4.eff_train, Prong1Plots_4.rej_train, "--", color='y', label="Fourth Test (Training Sample)")

        ax.plot(Prong1Plots_5.eff, Prong1Plots_5.rej, "-", color='m', label="Fifth Test (Evaluation Sample)")
        ax.plot(Prong1Plots_5.eff_train, Prong1Plots_5.rej_train, "--", color='m', label="Fifth Test (Training Sample)")

        ax.set_xlim((0., 1.))
        ax.set_ylim((1., 1e4))
        ax.set_yscale("log")
        ax.set_ylabel("Signal efficiency", x=1, ha="right")
        ax.set_xlabel("Background rejection", y=1, ha="right")
        #ax.legend()
        plt.savefig("ROC_curves/{}P1_Stability_Test.png".format(prefix))
        del Prong1Model_1, Prong1Model_2, Prong1Model_3, Prong1Model_4, Prong1Model_5

if do_prong3:
    if do_random_seed:
        Prong3Model = []
        train_real_y_p3 = []
        train_pred_y_p3 = []
        train_weights_p3 = []
        real_y_p3 = []
        pred_y_p3 = []
        jet_pt_p3 = []
        Prong3Plots = []
        Prong3_rejvseff = []
        fig, ax = plt.subplots()

        prong3_eval_losses = np.array([])

        for i in trange(0, 40):
            p3_jet_input = np.delete(DataP3.input_jet[:, 1:12], 10, 1)

            Prong3Model.append(Tau_Model(3, [DataP3.input_track[:,0:10,:], DataP3.input_tower[:,0:6,:], p3_jet_input], DataP3.sig_pt, DataP3.bck_pt, DataP3.jet_pt, DataP3.Ytrain, DataP3.new_weights, DataP3.cross_section, DataP3.mu, kinematic_vars={"jet_Eta" : DataP3.input_jet[:, 12], "jet_Phi" : DataP3.input_jet[:, 13]}))
            Prong3Model[i].Model_Fit(256, 100, 0.4, model=Prong3Model[i].RNNmodel, inputs=Prong3Model[i].inputs, addition_savename="{}RandomSeeded{}_".format(prefix, i))
            #Prong3Model[i].load_model("{}RandomSeeded{}_RNN_Model_Prong-3.h5".format(prefix, i))
            Prong3Model[i].evaluate_model(Prong3Model[i].eval_inputs, Prong3Model[i].eval_y, Prong3Model[i].eval_w, Prong3Model[i].RNNmodel)
            tr, tp = Prong3Model[i].get_train_scores(Prong3Model[i].RNNmodel, Prong3Model[i].inputs)
            train_real_y_p3.append(tr)
            train_pred_y_p3.append(tp)
            train_weights_p3.append(Prong3Model[i].w_train)
            Prong3Model[i].evaluate_model(Prong3Model[i].eval_inputs, Prong3Model[i].eval_y, Prong3Model[i].eval_w, Prong3Model[i].RNNmodel)

            prong3_eval_losses = np.append(prong3_eval_losses, [Prong3Model[i].eval_results[0]])

            ry, py, jpt = Prong3Model[i].predict(Prong3Model[i].RNNmodel, Prong3Model[i].eval_inputs)
            real_y_p3.append(ry)
            pred_y_p3.append(py)
            jet_pt_p3.append(jpt)
            Prong3Plots.append(Plots( "{}Prong3Plots_randomseed_{}".format(prefix, i), real_y_p3[i], pred_y_p3[i], Prong3Model[i].eval_w, train_real_y_p3[i], train_pred_y_p3[i], train_weights_p3[i], jet_pt_p3[i]))
            Prong3_rejvseff.append(Prong3Plots[i].plot_rej_vs_eff("{}ROC_Prong3_randomseed_{}".format(prefix, i), "ROC_curves/", do_train=True))
            plt.close(Prong3_rejvseff[i])
            ax.plot(Prong3Plots[i].eff, Prong3Plots[i].rej, "-", label="Test {} (Evaluation Sample)".format(i))
        ax.set_xlim((0., 1.))
        ax.set_ylim((1., 1e4))
        ax.set_yscale("log")
        ax.set_xlabel("Signal efficiency", x=1, ha="right")
        ax.set_ylabel("Background rejection", y=1, ha="right")
        #ax.legend(fontsize='xx-small')
        plt.savefig("ROC_curves/{}P3_Test_RandomSeed_Stability.png".format(prefix))
        #plt.show()
        plt.close(fig)
        fig_train, ax_train = plt.subplots()
        for i in trange(0, 40):
            ax_train.plot(Prong3Plots[i].eff_train, Prong3Plots[i].rej_train, "--", label="Test {} (Training Sample)".format(i))
        ax_train.set_xlim((0., 1.))
        ax_train.set_ylim((1., 1e4))
        ax_train.set_yscale("log")
        ax_train.set_xlabel("Signal efficiency", x=1, ha="right")
        ax_train.set_ylabel("Background rejection", y=1, ha="right")
        plt.savefig("ROC_curves/{}P3_Train_RandomSeed_Stability.png".format(prefix))
        #plt.show()
        plt.close(fig_train)
        fig_all, ax_all = plt.subplots()
        for i in trange(0, 40):
            ax_all.plot(Prong3Plots[i].eff, Prong3Plots[i].rej, "-", color="g", label="Test {} (Evaluation Sample)".format(i))
            ax_all.plot(Prong3Plots[i].eff_train, Prong3Plots[i].rej_train, "-", color="b", label="Test {} (Training Sample)".format(i))
        ax_all.set_xlim((0., 1.))
        ax_all.set_ylim((1., 1e4))
        ax_all.set_yscale("log")
        ax_all.set_xlabel("Signal efficiency", x=1, ha="right")
        ax_all.set_ylabel("Background rejection", y=1, ha="right")
        all_p1_lines = [Line2D([0], [0], color="g", lw=4),
                        Line2D([0], [0], color="b", lw=4)]
        ax_all.legend(all_p1_lines, ["Training ROC Curves", "Evaluation ROC Curves"])
        plt.savefig("ROC_curves/{}P3_ALL_RandomSeed_Stability.png".format(prefix))
        #plt.show()
        plt.close(fig_all)
        del Prong3Model

        loss_fig, loss_ax = plt.subplots()
        loss_ax.hist(prong3_eval_losses, color=colors["red"], label="Loss from Trained Models")
        loss_ax.set_xlabel("Number of Models")
        loss_ax.set_ylabel("Binary Cross-Entropy Loss")
        loss_ax.legend()
        loss_ax.autoscale()
        plt.savefig("report_plots/{}prong3_models_histogrammed_losses.png".format(prefix))
        plt.close(loss_fig)

        del loss_fig, loss_ax

    if do_same_seed:
        p3_jet_input = np.delete(DataP3.input_jet[:, 1:12], 10, 1)

        Prong3Model_1 = Tau_Model(3, [DataP3.input_track[:,0:10,:], DataP3.input_tower[:,0:6,:], p3_jet_input], DataP3.sig_pt, DataP3.bck_pt, DataP3.jet_pt, DataP3.Ytrain, DataP3.new_weights, DataP3.cross_section, DataP3.mu, kinematic_vars={"jet_Eta" : DataP3.input_jet[:, 12], "jet_Phi" : DataP3.input_jet[:, 13]}, shuffle_seed=1521)
        Prong3Model_2 = Tau_Model(3, [DataP3.input_track[:,0:10,:], DataP3.input_tower[:,0:6,:], p3_jet_input], DataP3.sig_pt, DataP3.bck_pt, DataP3.jet_pt, DataP3.Ytrain, DataP3.new_weights, DataP3.cross_section, DataP3.mu, kinematic_vars={"jet_Eta" : DataP3.input_jet[:, 12], "jet_Phi" : DataP3.input_jet[:, 13]}, shuffle_seed=1521)
        Prong3Model_3 = Tau_Model(3, [DataP3.input_track[:,0:10,:], DataP3.input_tower[:,0:6,:], p3_jet_input], DataP3.sig_pt, DataP3.bck_pt, DataP3.jet_pt, DataP3.Ytrain, DataP3.new_weights, DataP3.cross_section, DataP3.mu, kinematic_vars={"jet_Eta" : DataP3.input_jet[:, 12], "jet_Phi" : DataP3.input_jet[:, 13]}, shuffle_seed=1521)
        Prong3Model_4 = Tau_Model(3, [DataP3.input_track[:,0:10,:], DataP3.input_tower[:,0:6,:], p3_jet_input], DataP3.sig_pt, DataP3.bck_pt, DataP3.jet_pt, DataP3.Ytrain, DataP3.new_weights, DataP3.cross_section, DataP3.mu, kinematic_vars={"jet_Eta" : DataP3.input_jet[:, 12], "jet_Phi" : DataP3.input_jet[:, 13]}, shuffle_seed=1521)
        Prong3Model_5 = Tau_Model(3, [DataP3.input_track[:,0:10,:], DataP3.input_tower[:,0:6,:], p3_jet_input], DataP3.sig_pt, DataP3.bck_pt, DataP3.jet_pt, DataP3.Ytrain, DataP3.new_weights, DataP3.cross_section, DataP3.mu, kinematic_vars={"jet_Eta" : DataP3.input_jet[:, 12], "jet_Phi" : DataP3.input_jet[:, 13]}, shuffle_seed=1521)

        Prong3Model_1.Model_Fit(256, 100, 0.4, model=Prong3Model_1.RNNmodel, inputs=Prong3Model_1.inputs, addition_savename="{}1_".format(prefix))
        Prong3Model_2.Model_Fit(256, 100, 0.4, model=Prong3Model_2.RNNmodel, inputs=Prong3Model_2.inputs, addition_savename="{}2_".format(prefix))
        Prong3Model_3.Model_Fit(256, 100, 0.4, model=Prong3Model_3.RNNmodel, inputs=Prong3Model_3.inputs, addition_savename="{}3_".format(prefix))
        Prong3Model_4.Model_Fit(256, 100, 0.4, model=Prong3Model_4.RNNmodel, inputs=Prong3Model_4.inputs, addition_savename="{}4_".format(prefix))
        Prong3Model_5.Model_Fit(256, 100, 0.4, model=Prong3Model_5.RNNmodel, inputs=Prong3Model_5.inputs, addition_savename="{}5_".format(prefix))

        #Prong3Model_1.load_model("{}RNN_Model_Prong-3.h5".format("{}1_".format(prefix)))
        #Prong3Model_2.load_model("{}RNN_Model_Prong-3.h5".format("{}2_".format(prefix)))
        #Prong3Model_3.load_model("{}RNN_Model_Prong-3.h5".format("{}3_".format(prefix)))
        #Prong3Model_4.load_model("{}RNN_Model_Prong-3.h5".format("{}4_".format(prefix)))
        #Prong3Model_5.load_model("{}RNN_Model_Prong-3.h5".format("{}5_".format(prefix)))

        Prong3Model_1.evaluate_model(Prong3Model_1.eval_inputs, Prong3Model_1.eval_y, Prong3Model_1.eval_w, Prong3Model_1.RNNmodel)
        Prong3Model_2.evaluate_model(Prong3Model_2.eval_inputs, Prong3Model_2.eval_y, Prong3Model_2.eval_w, Prong3Model_2.RNNmodel)
        Prong3Model_3.evaluate_model(Prong3Model_3.eval_inputs, Prong3Model_3.eval_y, Prong3Model_3.eval_w, Prong3Model_3.RNNmodel)
        Prong3Model_4.evaluate_model(Prong3Model_4.eval_inputs, Prong3Model_4.eval_y, Prong3Model_4.eval_w, Prong3Model_4.RNNmodel)
        Prong3Model_5.evaluate_model(Prong3Model_5.eval_inputs, Prong3Model_5.eval_y, Prong3Model_5.eval_w, Prong3Model_5.RNNmodel)

        train_real_y_p3_1, train_pred_y_p3_1 = Prong3Model_1.get_train_scores(Prong3Model_1.RNNmodel, Prong3Model_1.inputs)
        train_real_y_p3_2, train_pred_y_p3_2 = Prong3Model_2.get_train_scores(Prong3Model_2.RNNmodel, Prong3Model_2.inputs)
        train_real_y_p3_3, train_pred_y_p3_3 = Prong3Model_3.get_train_scores(Prong3Model_3.RNNmodel, Prong3Model_3.inputs)
        train_real_y_p3_4, train_pred_y_p3_4 = Prong3Model_4.get_train_scores(Prong3Model_4.RNNmodel, Prong3Model_4.inputs)
        train_real_y_p3_5, train_pred_y_p3_5 = Prong3Model_5.get_train_scores(Prong3Model_5.RNNmodel, Prong3Model_5.inputs)

        train_weights_p3_1 = Prong3Model_1.w_train
        train_weights_p3_2 = Prong3Model_2.w_train
        train_weights_p3_3 = Prong3Model_3.w_train
        train_weights_p3_4 = Prong3Model_4.w_train
        train_weights_p3_5 = Prong3Model_5.w_train

        Prong3Model_1.evaluate_model(Prong3Model_1.eval_inputs, Prong3Model_1.eval_y, Prong3Model_1.eval_w, Prong3Model_1.RNNmodel)
        Prong3Model_2.evaluate_model(Prong3Model_2.eval_inputs, Prong3Model_2.eval_y, Prong3Model_2.eval_w, Prong3Model_2.RNNmodel)
        Prong3Model_3.evaluate_model(Prong3Model_3.eval_inputs, Prong3Model_3.eval_y, Prong3Model_3.eval_w, Prong3Model_3.RNNmodel)
        Prong3Model_4.evaluate_model(Prong3Model_4.eval_inputs, Prong3Model_4.eval_y, Prong3Model_4.eval_w, Prong3Model_4.RNNmodel)
        Prong3Model_5.evaluate_model(Prong3Model_5.eval_inputs, Prong3Model_5.eval_y, Prong3Model_5.eval_w, Prong3Model_5.RNNmodel)

        real_y_p3_1, pred_y_p3_1, jet_pt_p3_1 = Prong3Model_1.predict(Prong3Model_1.RNNmodel, Prong3Model_1.eval_inputs)
        real_y_p3_2, pred_y_p3_2, jet_pt_p3_2 = Prong3Model_2.predict(Prong3Model_2.RNNmodel, Prong3Model_2.eval_inputs)
        real_y_p3_3, pred_y_p3_3, jet_pt_p3_3 = Prong3Model_3.predict(Prong3Model_3.RNNmodel, Prong3Model_3.eval_inputs)
        real_y_p3_4, pred_y_p3_4, jet_pt_p3_4 = Prong3Model_4.predict(Prong3Model_4.RNNmodel, Prong3Model_4.eval_inputs)
        real_y_p3_5, pred_y_p3_5, jet_pt_p3_5 = Prong3Model_5.predict(Prong3Model_5.RNNmodel, Prong3Model_5.eval_inputs)



        Prong3Plots_1 = Plots( "{}Prong3Plots_1".format(prefix), real_y_p3_1, pred_y_p3_1, Prong3Model_1.eval_w, train_real_y_p3_1, train_pred_y_p3_1, train_weights_p3_1, jet_pt_p3_1)
        Prong3Plots_2 = Plots( "{}Prong3Plots_2".format(prefix), real_y_p3_2, pred_y_p3_2, Prong3Model_2.eval_w, train_real_y_p3_2, train_pred_y_p3_2, train_weights_p3_2, jet_pt_p3_2)
        Prong3Plots_3 = Plots( "{}Prong3Plots_3".format(prefix), real_y_p3_3, pred_y_p3_3, Prong3Model_3.eval_w, train_real_y_p3_3, train_pred_y_p3_3, train_weights_p3_3, jet_pt_p3_3)
        Prong3Plots_4 = Plots( "{}Prong3Plots_4".format(prefix), real_y_p3_4, pred_y_p3_4, Prong3Model_4.eval_w, train_real_y_p3_4, train_pred_y_p3_4, train_weights_p3_4, jet_pt_p3_4)
        Prong3Plots_5 = Plots( "{}Prong3Plots_5".format(prefix), real_y_p3_5, pred_y_p3_5, Prong3Model_5.eval_w, train_real_y_p3_5, train_pred_y_p3_5, train_weights_p3_5, jet_pt_p3_5)

        Prong3_rejveff_1 = Prong3Plots_1.plot_rej_vs_eff("{}ROC_Prong3_1".format(prefix), "ROC_curves/", do_train=True)
        Prong3_rejveff_2 = Prong3Plots_2.plot_rej_vs_eff("{}ROC_Prong3_2".format(prefix), "ROC_curves/", do_train=True)
        Prong3_rejveff_3 = Prong3Plots_3.plot_rej_vs_eff("{}ROC_Prong3_3".format(prefix), "ROC_curves/", do_train=True)
        Prong3_rejveff_4 = Prong3Plots_4.plot_rej_vs_eff("{}ROC_Prong3_4".format(prefix), "ROC_curves/", do_train=True)
        Prong3_rejveff_5 = Prong3Plots_5.plot_rej_vs_eff("{}ROC_Prong3_5".format(prefix), "ROC_curves/", do_train=True)

        #plt.show()

        fig, ax = plt.subplots()
        ax.plot(Prong3Plots_1.eff, Prong3Plots_1.rej, "-", color='r', label="First Test (Evaluation Sample)")
        ax.plot(Prong3Plots_1.eff_train, Prong3Plots_1.rej_train, "--", color='r', label="First Test (Training Sample)")

        ax.plot(Prong3Plots_2.eff, Prong3Plots_2.rej, "-", color='b', label="Second Test (Evaluation Sample)")
        ax.plot(Prong3Plots_2.eff_train, Prong3Plots_2.rej_train, "--", color='b', label="Second Test (Training Sample)")

        ax.plot(Prong3Plots_3.eff, Prong3Plots_3.rej, "-", color='g', label="Third Test (Evaluation Sample)")
        ax.plot(Prong3Plots_3.eff_train, Prong3Plots_3.rej_train, "--", color='g', label="Third Test (Training Sample)")

        ax.plot(Prong3Plots_4.eff, Prong3Plots_4.rej, "-", color='y', label="Fourth Test (Evaluation Sample)")
        ax.plot(Prong3Plots_4.eff_train, Prong3Plots_4.rej_train, "--", color='y', label="Fourth Test (Training Sample)")

        ax.plot(Prong3Plots_5.eff, Prong3Plots_5.rej, "-", color='m', label="Fifth Test (Evaluation Sample)")
        ax.plot(Prong3Plots_5.eff_train, Prong3Plots_5.rej_train, "--", color='m', label="Fifth Test (Training Sample)")

        ax.set_xlim((0., 1.))
        ax.set_ylim((1., 1e4))
        ax.set_yscale("log")
        ax.set_xlabel("Signal efficiency", x=1, ha="right")
        ax.set_ylabel("Background rejection", y=1, ha="right")
        ax.legend()
        plt.savefig("ROC_curves/{}P3_Stability_Test.png".format(prefix))
        del Prong3Model_1, Prong3Model_2, Prong3Model_3, Prong3Model_4, Prong3Model_5

prefix_1 = "CoreTrackCond_"

CTCDataP1 = RNN_Data(1, False, "prong1_data", print_hists=False,
                  BacktreeFile=["{}0-1_background_wPU_tree_1-Prong".format(prefix_1), "{}1-1_background_wPU_tree_1-Prong".format(prefix_1), "{}2-1_background_wPU_tree_1-Prong".format(prefix_1),
                                "{}3-1_background_wPU_tree_1-Prong".format(prefix_1), "{}4-1_background_wPU_tree_1-Prong".format(prefix_1), "{}0-2_background_wPU_tree_1-Prong".format(prefix_1),
                                "{}1-2_background_wPU_tree_1-Prong".format(prefix_1), "{}2-2_background_wPU_tree_1-Prong".format(prefix_1), "{}3-2_background_wPU_tree_1-Prong".format(prefix_1),
                                "{}4-2_background_wPU_tree_1-Prong".format(prefix_1)]
                , BackTreeName=["{}0-1_background_wPU_tree".format(prefix_1), "{}1-1_background_wPU_tree".format(prefix_1), "{}2-1_background_wPU_tree".format(prefix_1),
                                "{}3-1_background_wPU_tree".format(prefix_1), "{}4-1_background_wPU_tree".format(prefix_1), "{}0-2_background_wPU_tree".format(prefix_1),
                                "{}1-2_background_wPU_tree".format(prefix_1), "{}2-2_background_wPU_tree".format(prefix_1), "{}3-2_background_wPU_tree".format(prefix_1),
                                "{}4-2_background_wPU_tree".format(prefix_1)]
                  , SignaltreeFile=["{}0_signal_wPU_tree_1-Prong".format(prefix_1), "{}1_signal_wPU_tree_1-Prong".format(prefix_1)],
                  SignalTreeName=["{}0_signal_wPU_tree".format(prefix_1), "{}1_signal_wPU_tree".format(prefix_1)], BackendPartOfTree="", SignalendPartOfTree="")

CTCDataP3 = RNN_Data(3, False, "prong3_data", print_hists=False,
                  BacktreeFile=["{}0-1_background_wPU_tree_3-Prong".format(prefix), "{}1-1_background_wPU_tree_3-Prong".format(prefix), "{}2-1_background_wPU_tree_3-Prong".format(prefix),
                                "{}3-1_background_wPU_tree_3-Prong".format(prefix), "{}4-1_background_wPU_tree_3-Prong".format(prefix), "{}0-2_background_wPU_tree_3-Prong".format(prefix),
                                "{}1-2_background_wPU_tree_3-Prong".format(prefix), "{}2-2_background_wPU_tree_3-Prong".format(prefix), "{}3-2_background_wPU_tree_3-Prong".format(prefix),
                                "{}4-2_background_wPU_tree_3-Prong".format(prefix)]
                , BackTreeName=["{}0-1_background_wPU_tree".format(prefix), "{}1-1_background_wPU_tree".format(prefix), "{}2-1_background_wPU_tree".format(prefix),
                                "{}3-1_background_wPU_tree".format(prefix), "{}4-1_background_wPU_tree".format(prefix), "{}0-2_background_wPU_tree".format(prefix),
                                "{}1-2_background_wPU_tree".format(prefix), "{}2-2_background_wPU_tree".format(prefix), "{}3-2_background_wPU_tree".format(prefix),
                                "{}4-2_background_wPU_tree".format(prefix)]
                  , SignaltreeFile=["{}0_signal_wPU_tree_3-Prong".format(prefix), "{}1_signal_wPU_tree_3-Prong".format(prefix)],
                  SignalTreeName=["{}0_signal_wPU_tree".format(prefix), "{}1_signal_wPU_tree".format(prefix)], BackendPartOfTree="", SignalendPartOfTree="")

do_comparison = True

if do_comparison:
    if do_random_seed:
        CTCProng1Model = []
        CTCtrain_real_y_p1 = []
        CTCtrain_pred_y_p1 = []
        CTCtrain_weights_p1 = []
        CTCreal_y_p1 = []
        CTCpred_y_p1 = []
        CTCjet_pt_p1 = []
        CTCProng1Plots = []
        CTCprong1_rejvseff = []
        fig, ax = plt.subplots()
        CTCprong1_eval_losses = np.array([])
        for i in trange(0, 40):
            CTCp1_jet_input = np.delete(CTCDataP1.input_jet[:, 1:12], 9, 1)

            CTCProng1Model.append(Tau_Model(1, [CTCDataP1.input_track[:,0:10,:], CTCDataP1.input_tower[:,0:6,:], CTCp1_jet_input], CTCDataP1.sig_pt, CTCDataP1.bck_pt, CTCDataP1.jet_pt, CTCDataP1.Ytrain, CTCDataP1.new_weights, CTCDataP1.cross_section, CTCDataP1.mu, kinematic_vars={"jet_Eta" : CTCDataP1.input_jet[:, 12], "jet_Phi" : CTCDataP1.input_jet[:, 13]}))
            #Prong1Model[i].Model_Fit(256, 100, 0.2, model=Prong1Model[i].RNNmodel, inputs=Prong1Model[i].inputs, addition_savename="{}RandomSeeded{}_".format(prefix,i))
            CTCProng1Model[i].load_model("{}RandomSeeded{}_RNN_Model_Prong-1.h5".format(prefix_1, i))

            CTCProng1Model[i].evaluate_model(CTCProng1Model[i].eval_inputs, CTCProng1Model[i].eval_y, CTCProng1Model[i].eval_w, CTCProng1Model[i].RNNmodel)
            tr, tp = CTCProng1Model[i].get_train_scores(CTCProng1Model[i].RNNmodel, CTCProng1Model[i].inputs)
            CTCtrain_real_y_p1.append(tr)
            CTCtrain_pred_y_p1.append(tp)
            CTCtrain_weights_p1.append(CTCProng1Model[i].w_train)
            CTCProng1Model[i].evaluate_model(CTCProng1Model[i].eval_inputs, CTCProng1Model[i].eval_y, CTCProng1Model[i].eval_w, CTCProng1Model[i].RNNmodel)

            CTCprong1_eval_losses = np.append(CTCprong1_eval_losses, [CTCProng1Model[i].eval_results[0]])

            ry, py, jpt = CTCProng1Model[i].predict(CTCProng1Model[i].RNNmodel, CTCProng1Model[i].eval_inputs)
            CTCreal_y_p1.append(ry)
            CTCpred_y_p1.append(py)
            CTCjet_pt_p1.append(jpt)
            CTCProng1Plots.append(Plots( "{}Prong1Plots_randomseed_{}".format(prefix_1, i), CTCreal_y_p1[i], CTCpred_y_p1[i], CTCProng1Model[i].eval_w, CTCtrain_real_y_p1[i], CTCtrain_pred_y_p1[i], CTCtrain_weights_p1[i],
                                      [i]))
            CTCprong1_rejvseff.append(CTCProng1Plots[i].plot_rej_vs_eff("{}ROC_Prong1_randomseed_{}".format(prefix_1, i), "ROC_curves/", do_train=True))
            plt.close(CTCprong1_rejvseff[i])
            ax.plot(CTCProng1Plots[i].eff, CTCProng1Plots[i].rej, "-", label="Test {} (Evaluation Sample)".format(i))
        fig_comp, ax_comp = plt.subplots()
        for i in trange(0, 40):
            ax_comp.plot(CTCProng1Plots[i].eff, CTCProng1Plots[i].rej, "-", color="cyan", label="Test {} (Training Sample)".format(i))
            ax_comp.plot(Prong1Plots[i].eff, Prong1Plots[i].rej, "-", color="lime", label="Test {} (Training Sample)".format(i))
        ax_comp.set_xlim((0., 1.))
        ax_comp.set_ylim((1., 1e4))
        ax_comp.set_yscale("log")
        ax_comp.set_xlabel("Signal efficiency", x=1, ha="right")
        ax_comp.set_ylabel("Background rejection", y=1, ha="right")
        all_p1_lines = [Line2D([0], [0], color="lime", lw=4),
                        Line2D([0], [0], color="cyan", lw=4)]
        ax_comp.legend(all_p1_lines, ["More than 1 Core Track", "1 and 6 Core Tracks"])
        plt.savefig("ROC_curves/{}_CTCandNewCOnd_P1_ALL_RandomSeed_Stability.png".format(prefix))
        #plt.show()
        plt.close(fig_comp)
        del CTCProng1Model

if do_prong3:
    if do_random_seed:
        CTCProng3Model = []
        CTCtrain_real_y_p3 = []
        CTCtrain_pred_y_p3 = []
        CTCtrain_weights_p3 = []
        CTCreal_y_p3 = []
        CTCpred_y_p3 = []
        CTCjet_pt_p3 = []
        CTCProng3Plots = []
        CTCProng3_rejvseff = []
        CTCfig, CTCax = plt.subplots()

        CTCprong3_eval_losses = np.array([])

        for i in trange(0, 40):
            CTCp3_jet_input = np.delete(DataP3.input_jet[:, 1:12], 10, 1)

            CTCProng3Model.append(Tau_Model(3, [CTCDataP3.input_track[:,0:10,:], CTCDataP3.input_tower[:,0:6,:], CTCp3_jet_input], CTCDataP3.sig_pt, CTCDataP3.bck_pt, CTCDataP3.jet_pt, CTCDataP3.Ytrain, CTCDataP3.new_weights, CTCDataP3.cross_section, CTCDataP3.mu, kinematic_vars={"jet_Eta" : CTCDataP3.input_jet[:, 12], "jet_Phi" : CTCDataP3.input_jet[:, 13]}))
            #Prong3Model[i].Model_Fit(256, 100, 0.2, model=Prong3Model[i].RNNmodel, inputs=Prong3Model[i].inputs, addition_savename="{}RandomSeeded{}_".format(prefix, i))
            CTCProng3Model[i].load_model("{}RandomSeeded{}_RNN_Model_Prong-3.h5".format(prefix_1, i))
            CTCProng3Model[i].evaluate_model(CTCProng3Model[i].eval_inputs, CTCProng3Model[i].eval_y, CTCProng3Model[i].eval_w, CTCProng3Model[i].RNNmodel)
            tr, tp = CTCProng3Model[i].get_train_scores(CTCProng3Model[i].RNNmodel, CTCProng3Model[i].inputs)
            CTCtrain_real_y_p3.append(tr)
            CTCtrain_pred_y_p3.append(tp)
            CTCtrain_weights_p3.append(CTCProng3Model[i].w_train)
            CTCProng3Model[i].evaluate_model(CTCProng3Model[i].eval_inputs, CTCProng3Model[i].eval_y, CTCProng3Model[i].eval_w, CTCProng3Model[i].RNNmodel)

            CTCprong3_eval_losses = np.append(CTCprong3_eval_losses, [CTCProng3Model[i].eval_results[0]])

            ry, py, jpt = CTCProng3Model[i].predict(CTCProng3Model[i].RNNmodel, CTCProng3Model[i].eval_inputs)
            CTCreal_y_p3.append(ry)
            CTCpred_y_p3.append(py)
            CTCjet_pt_p3.append(jpt)
            CTCProng3Plots.append(Plots( "{}Prong3Plots_randomseed_{}".format(prefix_1, i), CTCreal_y_p3[i], CTCpred_y_p3[i], CTCProng3Model[i].eval_w, CTCtrain_real_y_p3[i], CTCtrain_pred_y_p3[i], CTCtrain_weights_p3[i], CTCjet_pt_p3[i]))
            CTCProng3_rejvseff.append(CTCProng3Plots[i].plot_rej_vs_eff("{}ROC_Prong3_randomseed_{}".format(prefix_1, i), "ROC_curves/", do_train=True))
            plt.close(CTCProng3_rejvseff[i])
            CTCax.plot(CTCProng3Plots[i].eff, CTCProng3Plots[i].rej, "-", label="Test {} (Evaluation Sample)".format(i))
        fig_comp3, ax_comp3 = plt.subplots()
        for i in trange(0, 40):
            ax_comp3.plot(CTCProng3Plots[i].eff, CTCProng3Plots[i].rej, "-", color="cyan", label="Test {} (Training Sample)".format(i))
            ax_comp3.plot(Prong3Plots[i].eff, Prong3Plots[i].rej, "-", color="lime", label="Test {} (Training Sample)".format(i))
        ax_comp3.set_xlim((0., 1.))
        ax_comp3.set_ylim((1., 1e4))
        ax_comp3.set_yscale("log")
        ax_comp3.set_xlabel("Signal efficiency", x=1, ha="right")
        ax_comp3.set_ylabel("Background rejection", y=1, ha="right")
        all_p3_lines = [Line2D([0], [0], color="lime", lw=4),
                        Line2D([0], [0], color="cyan", lw=4)]
        ax_comp3.legend(all_p3_lines, ["More than 3 Core Track", "3 and 8 Core Tracks"])
        plt.savefig("ROC_curves/{}_CTCandNewCOnd_P3_ALL_RandomSeed_Stability.png".format(prefix_1))
        #plt.show()
        plt.close(fig_comp3)
        del CTCProng3Model
