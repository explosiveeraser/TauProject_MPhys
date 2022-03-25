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
# from tqdm import tqdm, trange
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


def plt_2hist(name, sig_data, bck_data, sig_weight, bck_weight, num_bins, save_dir=False, bins=False, log_plot=False, hist_max=None, hist_min=None):
    fig, ax = plt.subplots()
    ax.hist(sig_data, weights=sig_weight,
            color=colors["red"], label="signal", bins=num_bins, density=True, histtype="step")
    ax.hist(bck_data, weights=bck_weight,
            color=colors["blue"], label="background", bins=num_bins, density=True, histtype="step")
    if log_plot:
        ax.set_yscale("log")
    ax.autoscale()
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


DataP1 = RNN_Data(1, False, "prong1_data", print_hists=False,
                  BacktreeFile=["1600pt_0-1_background_wPU_tree_1-Prong", "1600pt_1-1_background_wPU_tree_1-Prong", "1600pt_2-1_background_wPU_tree_1-Prong",
                                "1600pt_3-1_background_wPU_tree_1-Prong", "1600pt_4-1_background_wPU_tree_1-Prong", "1600pt_0-2_background_wPU_tree_1-Prong",
                                "1600pt_1-2_background_wPU_tree_1-Prong", "1600pt_2-2_background_wPU_tree_1-Prong", "1600pt_3-2_background_wPU_tree_1-Prong",
                                "1600pt_4-2_background_wPU_tree_1-Prong"]
                , BackTreeName=["1600pt_0-1_background_wPU_tree", "1600pt_1-1_background_wPU_tree", "1600pt_2-1_background_wPU_tree",
                                "1600pt_3-1_background_wPU_tree", "1600pt_4-1_background_wPU_tree", "1600pt_0-2_background_wPU_tree",
                                "1600pt_1-2_background_wPU_tree", "1600pt_2-2_background_wPU_tree", "1600pt_3-2_background_wPU_tree",
                                "1600pt_4-2_background_wPU_tree"]
                  , SignaltreeFile=["1600pt_0_signal_wPU_tree_1-Prong", "1600pt_1_signal_wPU_tree_1-Prong"],
                  SignalTreeName=["1600pt_0_signal_wPU_tree", "1600pt_1_signal_wPU_tree"], BackendPartOfTree="", SignalendPartOfTree="")

#print_hist = True

do_RNN = True
prong3 = True

#if print_hist:

#    DataP1.plot_hists()


if do_RNN:
    Prong1Model = Tau_Model(1, [DataP1.input_track[:,0:6,:], DataP1.input_tower[:,0:10,:], DataP1.input_jet[:, 1:12]], DataP1.sig_pt, DataP1.bck_pt, DataP1.jet_pt, DataP1.Ytrain, DataP1.new_weights, DataP1.cross_section, DataP1.mu, kinematic_vars={"jet_Eta" : DataP1.input_jet[:, 12], "jet_Phi" : DataP1.input_jet[:, 13]})
    plot_2_histogram("Train_weights-Eval_weights", Prong1Model.w_train, Prong1Model.eval_w, np.ones_like(Prong1Model.w_train), np.ones_like(Prong1Model.eval_w), 70)

    all_sig_pt = Prong1Model.jet_pt[Prong1Model.index_of_sig_bck == "s"]
    all_bgk_pt = Prong1Model.jet_pt[Prong1Model.index_of_sig_bck == "b"]
    all_sig_w = Prong1Model.w[Prong1Model.index_of_sig_bck == "s"]
    all_bgk_w = Prong1Model.w[Prong1Model.index_of_sig_bck == "b"]
    bck_weight = Prong1Model.w[Prong1Model.index_of_sig_bck == "b"]
    plot_2_histogram("All_Jet_PT", all_sig_pt, all_bgk_pt, all_sig_w, all_bgk_w, 50, hist_min=20., hist_max=3300.0)


    plt_2hist("All_Jet_PT", all_sig_pt, all_bgk_pt, all_sig_w, all_bgk_w, 50, hist_min=20., hist_max=710.0, log_plot=True)



    training_sig_pt = Prong1Model.train_jet_pt[Prong1Model.train_sigbck_index == "s"]
    print("{}--------------".format(len(training_sig_pt)))
    training_bck_pt = Prong1Model.train_jet_pt[Prong1Model.train_sigbck_index == "b"]
    train_s_w = Prong1Model.w_train[Prong1Model.train_sigbck_index == "s"]
    train_b_w = Prong1Model.w_train[Prong1Model.train_sigbck_index == "b"]
    plt_2hist("Training_Jet_PT", training_sig_pt, training_bck_pt, train_s_w, train_b_w, 50, hist_min=20., hist_max=710.0, log_plot=True)


    eval_sig_pt = Prong1Model.eval_jet_pt[Prong1Model.eval_sigbck_index == "s"]
    eval_bck_pt = Prong1Model.eval_jet_pt[Prong1Model.eval_sigbck_index == "b"]
    eval_s_w = Prong1Model.eval_w[Prong1Model.eval_sigbck_index == "s"]
    eval_b_w = Prong1Model.eval_w[Prong1Model.eval_sigbck_index == "b"]
    plt_2hist("Evaluation_Jet_PT", eval_sig_pt, eval_bck_pt, eval_s_w, eval_b_w, 50, hist_min=20., hist_max=710.0, log_plot=True)

    #plt.show()

    sig_pt = DataP1.sig_pt
    bck_pt = DataP1.bck_pt

    s_w = DataP1.cross_section[-DataP1.length_sig:-1]
    b_w = DataP1.cross_section[0:DataP1.length_bck]

    plot_2_histogram("dataproc_jet_pt", sig_pt, bck_pt, s_w, b_w, 50, hist_min=20., hist_max=3300.0)


    bck_weight = DataP1.new_weights[0:DataP1.length_bck]
    sig_weight = DataP1.new_weights[-DataP1.length_sig:-1]

    plot_2_histogram("new_weights", sig_weight, bck_weight, np.ones_like(sig_weight), np.ones_like(bck_weight), 75)

    plt.show()

    Prong1Model.Model_Fit(256, 100, 0.2, model=Prong1Model.RNNmodel, inputs=Prong1Model.inputs)
    Prong1Model.plot_loss()


    train_real_y, train_pred_y = Prong1Model.get_train_scores(Prong1Model.RNNmodel, Prong1Model.inputs)
    train_weights = Prong1Model.w_train

    Prong1Model.evaluate_model(Prong1Model.eval_inputs, Prong1Model.eval_y, Prong1Model.eval_w, Prong1Model.RNNmodel)

    real_y, pred_y, jet_pt = Prong1Model.predict(Prong1Model.RNNmodel, Prong1Model.eval_inputs)

    back_real_y, back_pred_y, back_jet_pt = Prong1Model.predict_back(Prong1Model.RNNmodel, Prong1Model.eval_back_inputs)

    Prong1Plots = Plots( "Prong1Plots", real_y, pred_y, Prong1Model.eval_w, train_real_y, train_pred_y, train_weights, jet_pt)

    back_plots = Plots("P1_backplots", back_real_y, back_pred_y, Prong1Model.eval_back_w, train_real_y, train_pred_y, train_weights, back_jet_pt)

    Prong1Plots.plot_raw_score_vs_jetPT()
    Prong1Plots.histogram_RNN_score()

    back_plots.histogram_RNN_score()

    prong1_rejveff = Prong1Plots.plot_rej_vs_eff("ROC_Prong1", "ROC_curves/", do_train=True)

    plt.draw()
    plt.show()

    sig_train_pt = Prong1Model.train_jet_pt[Prong1Model.train_sigbck_index == "s"]
    sig_train_weight = Prong1Model.w_train[Prong1Model.train_sigbck_index == "s"]
    if pile_up:
        sig_train_mu = Prong1Model.mu_train[Prong1Model.train_sigbck_index == "s"]

    sig_train_score = Prong1Model.RNNmodel.predict([Prong1Model.inputs[0][Prong1Model.train_sigbck_index == "s"], Prong1Model.inputs[1][Prong1Model.train_sigbck_index == "s"], Prong1Model.inputs[2][Prong1Model.train_sigbck_index == "s"]])
    sig_train_score.flatten()

    bkg_train_score = Prong1Model.RNNmodel.predict([Prong1Model.inputs[0][Prong1Model.train_sigbck_index == "b"], Prong1Model.inputs[1][Prong1Model.train_sigbck_index == "b"], Prong1Model.inputs[2][Prong1Model.train_sigbck_index == "b"]])
    bkg_train_score.flatten()
    bkg_train_weight = Prong1Model.w_train[Prong1Model.train_sigbck_index == "b"].flatten()
    if pile_up:
        bkg_train_mu = Prong1Model.mu_train[Prong1Model.train_sigbck_index == "b"]

    sig_test_pt = Prong1Model.eval_jet_pt[Prong1Model.eval_sigbck_index == "s"]
    sig_test_weight = Prong1Model.eval_w[Prong1Model.eval_sigbck_index == "s"]

    sig_test_score = pred_y[Prong1Model.eval_sigbck_index == "s"]
    sig_test_score.flatten()
    if pile_up:
        sig_test_mu = Prong1Model.eval_mu[Prong1Model.eval_sigbck_index == "s"]
    bkg_test_pt = Prong1Model.eval_jet_pt[Prong1Model.eval_sigbck_index == "b"]
    bkg_test_weight = Prong1Model.eval_w[Prong1Model.eval_sigbck_index == "b"]

    bkg_test_score = pred_y[Prong1Model.eval_sigbck_index == "b"]
    bkg_test_score.flatten()
    if pile_up:
        bkg_test_mu = Prong1Model.eval_mu[Prong1Model.eval_sigbck_index == "b"]

    #ATLAS Score Plot
    rnn_score = ScorePlot(test=True, train=True, log_y=True)
    rnn_score.plot(sig_train_score, bkg_train_score, sig_train_weight, bkg_train_weight,
                   sig_test_score, bkg_test_score, sig_test_weight, bkg_test_weight, "Prong1", "score_plots/")
    plt.show()
  #  input("Enter..")

    #Test ATLAS plotting algorithm
    if pile_up:
        FEP_tight = FlattenerEfficiencyPlot(sig_train_pt, sig_train_score, sig_train_mu, sig_test_pt, sig_test_score, sig_test_mu, 60 / 100)
        FEP_fig = FEP_tight.plot()
        plt.show()
        EPs = []
        RPs = []
        plots = []
        plots_idx = 0
        # PT
        sig_test_xvar = Prong1Model.eval_jet_pt[Prong1Model.eval_sigbck_index == "s"]
        bkg_test_xvar = Prong1Model.eval_jet_pt[Prong1Model.eval_sigbck_index == "b"]
        pt_max_sig = np.max(sig_test_xvar)
        pt_min_sig = np.min(sig_test_xvar)

#        stest_bins_pt = np.percentile(sig_test_xvar, np.linspace(20.0, 400.0, 16))
        stest_bins_pt = np.linspace(20.0, 400.0, 38)
 #       btest_bins_pt = np.percentile(bkg_test_xvar, np.linspace(np.log10(20.), np.log10(250.0), 9))
        btest_bins_pt = 10 ** np.linspace(np.log10(20.0), np.log10(400.0), 8)

        stest_bins_mu = np.linspace(0, 70, 38)
        btest_bins_mu = np.linspace(0, 70, 6)

        stest_bins_eta = np.linspace(0, 2.5, 24)
        btest_bins_eta = np.linspace(0, 2.5, 12)
        sig_test_eta = Prong1Model.eval_kinematic_vars["jet_Eta"][Prong1Model.eval_sigbck_index == "s"]
        bck_test_eta = Prong1Model.eval_kinematic_vars["jet_Eta"][Prong1Model.eval_sigbck_index == "b"]

        stest_bins_phi = np.linspace(0, np.pi, 24)
        btest_bins_phi = np.linspace(0, np.pi, 12)
        sig_test_phi = Prong1Model.eval_kinematic_vars["jet_Phi"][Prong1Model.eval_sigbck_index == "s"]
        bck_test_phi = Prong1Model.eval_kinematic_vars["jet_Phi"][Prong1Model.eval_sigbck_index == "b"]

        #efficiency = [95, 85, 75, 60, 45]
        eff = [0.95, 0.85, 0.75, 0.60]
        colours = ["red", "blue", "green", "violet"]
       # for eff in efficiency:
        pt_efficiency = EfficiencyPlot(eff, colours, bins=stest_bins_pt)
        pt_eff_plot = pt_efficiency.plot(sig_train_pt, sig_train_score, sig_train_mu, sig_test_pt, sig_test_score, sig_test_mu, "1prong_Jet_PT_eff",
                                    sig_test_xvar, sig_test_weight, "efficiencies/")
        pt_rejection = RejectionPlot(eff , colours, bins=btest_bins_pt)
        pt_rej_plot = pt_rejection.plot(sig_train_pt, sig_train_score, sig_train_mu, sig_test_pt, sig_test_weight, bkg_test_pt,
                                             bkg_test_weight, bkg_test_score, bkg_test_mu, bkg_test_xvar, "1prong_Jet_PT_rej", "rejection/")

        mu_efficiency = EfficiencyPlot(eff , colours, bins=stest_bins_mu)
        mu_eff_plot = mu_efficiency.plot(sig_train_pt, sig_train_score, sig_train_mu, sig_test_pt, sig_test_score,
                                         sig_test_mu, "1prong_Mu_eff",
                                         sig_test_mu, sig_test_weight, "efficiencies/")
        mu_rejection = RejectionPlot(eff , colours, bins=btest_bins_mu)
        mu_rej_plot = mu_rejection.plot(sig_train_pt, sig_train_score, sig_train_mu, sig_test_pt, sig_test_weight,
                                        bkg_test_pt,
                                        bkg_test_weight, bkg_test_score, bkg_test_mu, bkg_test_mu,
                                        "1prong_Mu_eff", "rejection/")
        eta_efficiency = EfficiencyPlot(eff , colours, bins=stest_bins_eta)
        eta_eff_plot = eta_efficiency.plot(sig_train_pt, sig_train_score, sig_train_mu, sig_test_pt, sig_test_score,
                                         sig_test_mu, "1prong_jet_Eta_eff",
                                         np.abs(sig_test_eta), sig_test_weight, "efficiencies/")
        eta_rejection = RejectionPlot(eff , colours, bins=btest_bins_eta)
        eta_rej_plot = eta_rejection.plot(sig_train_pt, sig_train_score, sig_train_mu, sig_test_pt, sig_test_weight,
                                        bkg_test_pt,
                                        bkg_test_weight, bkg_test_score, bkg_test_mu, np.abs(bck_test_eta),
                                        "1prong_jet_Eta_rej", "rejection/")
        phi_efficiency = EfficiencyPlot(eff , colours, bins=stest_bins_phi)
        phi_eff_plot = phi_efficiency.plot(sig_train_pt, sig_train_score, sig_train_mu, sig_test_pt, sig_test_score,
                                          sig_test_mu, "1prong_jet_Phi_eff",
                                          np.abs(sig_test_phi), sig_test_weight, "efficiencies/")
        phi_rejection = RejectionPlot(eff , colours, bins=btest_bins_phi)
        phi_rej_plot = phi_rejection.plot(sig_train_pt, sig_train_score, sig_train_mu, sig_test_pt, sig_test_weight,
                                         bkg_test_pt,
                                         bkg_test_weight, bkg_test_score, bkg_test_mu, np.abs(bck_test_phi),
                                         "1prong_jet_Phi_rej", "rejection/")
        #plt.show()
        input("enter...")
    else:
        FEP_tight = FlattenerEfficiencyPlot(sig_train_pt, sig_train_score, sig_test_pt, sig_test_score, 60/100)
        FEP_fig = FEP_tight.plot()
        plt.show()
        EPs = []
        RPs = []
        plots = []
        plots_idx = 0
        for idx in tqdm.trange(0, len(Prong1Model.inputs[2][0,1:12])):
            sig_test_xvar = Prong1Model.eval_inputs[2][Prong1Model.eval_sigbck_index == "s", idx]
            bkg_test_xvar = Prong1Model.eval_inputs[2][Prong1Model.eval_sigbck_index == "b", idx]
            EPs.append(EfficiencyPlot(60/100))
            plots.append(EPs[plots_idx].plot(sig_train_pt, sig_train_score, sig_test_pt, sig_test_score, "xvar_{}".format(idx),
                                             sig_test_xvar))
            RPs.append(RejectionPlot(60/100))
            plots.append(RPs[plots_idx].plot(sig_train_pt, sig_train_score, sig_test_pt, sig_test_weight, bkg_test_pt,
                                             bkg_test_weight, bkg_test_score, bkg_test_xvar, "xvar_{}".format(idx)))
        plt.show()

######################################
if prong3:
    DataP3 = RNN_Data(3, False, "prong3_data", print_hists=False,
                      BacktreeFile=["0-1_background_wPU_tree_3-Prong", "1-1_background_wPU_tree_3-Prong",
                                    "2-1_background_wPU_tree_3-Prong", "3-1_background_wPU_tree_3-Prong",
                                    "4-1_background_wPU_tree_3-Prong",
                                    "0-2_background_wPU_tree_3-Prong", "1-2_background_wPU_tree_3-Prong",
                                    "2-2_background_wPU_tree_3-Prong", "3-2_background_wPU_tree_3-Prong",
                                    "4-2_background_wPU_tree_3-Prong"]
                      , BackTreeName=["0-1_background_wPU_tree", "1-1_background_wPU_tree", "2-1_background_wPU_tree",
                                      "3-1_background_wPU_tree", "4-1_background_wPU_tree",
                                      "0-2_background_wPU_tree", "1-2_background_wPU_tree", "2-2_background_wPU_tree",
                                      "3-2_background_wPU_tree", "4-2_background_wPU_tree"]
                      , SignaltreeFile=["0_signal_wPU_tree_3-Prong", "1_signal_wPU_tree_3-Prong"],
                      SignalTreeName=["0_signal_wPU_tree", "1_signal_wPU_tree"], BackendPartOfTree="",
                      SignalendPartOfTree="")

    Prong3Model = Tau_Model(3,
                            [DataP3.input_track[:, 0:6, :], DataP3.input_tower[:, 0:10, :], DataP3.input_jet[:, 1:12]],
                            DataP3.sig_pt, DataP3.bck_pt, DataP3.jet_pt, DataP3.Ytrain, DataP3.new_weights,
                            DataP3.cross_section, DataP3.mu,
                            kinematic_vars={"jet_Eta": DataP3.input_jet[:, 12], "jet_Phi": DataP3.input_jet[:, 13]})
    plot_2_histogram("Train_weights-Eval_weights_3Prong", Prong3Model.w_train, Prong3Model.eval_w,
                     np.ones_like(Prong3Model.w_train), np.ones_like(Prong3Model.eval_w), 70)

    all_sig_pt = Prong3Model.jet_pt[Prong3Model.index_of_sig_bck == "s"]
    all_bgk_pt = Prong3Model.jet_pt[Prong3Model.index_of_sig_bck == "b"]
    all_sig_w = Prong3Model.w[Prong3Model.index_of_sig_bck == "s"]
    all_bgk_w = Prong3Model.w[Prong3Model.index_of_sig_bck == "b"]
    bck_weight = Prong3Model.w[Prong3Model.index_of_sig_bck == "b"]
    plot_2_histogram("All_Jet_PT_3Prong", all_sig_pt, all_bgk_pt, all_sig_w, all_bgk_w, 50, hist_min=20., hist_max=3300.0)

    plt_2hist("All_Jet_PT_3Prong", all_sig_pt, all_bgk_pt, all_sig_w, all_bgk_w, 50, hist_min=20., hist_max=710.0,
              log_plot=True)

    training_sig_pt = Prong3Model.train_jet_pt[Prong3Model.train_sigbck_index == "s"]
    print("{}--------------".format(len(training_sig_pt)))
    training_bck_pt = Prong3Model.train_jet_pt[Prong3Model.train_sigbck_index == "b"]
    train_s_w = Prong3Model.w_train[Prong3Model.train_sigbck_index == "s"]
    train_b_w = Prong3Model.w_train[Prong3Model.train_sigbck_index == "b"]
    plt_2hist("Training_Jet_PT_3Prong", training_sig_pt, training_bck_pt, train_s_w, train_b_w, 50, hist_min=20.,
              hist_max=710.0, log_plot=True)

    eval_sig_pt = Prong3Model.eval_jet_pt[Prong3Model.eval_sigbck_index == "s"]
    eval_bck_pt = Prong3Model.eval_jet_pt[Prong3Model.eval_sigbck_index == "b"]
    eval_s_w = Prong3Model.eval_w[Prong3Model.eval_sigbck_index == "s"]
    eval_b_w = Prong3Model.eval_w[Prong3Model.eval_sigbck_index == "b"]
    plt_2hist("Evaluation_Jet_PT_3Prong", eval_sig_pt, eval_bck_pt, eval_s_w, eval_b_w, 50, hist_min=20., hist_max=710.0,
              log_plot=True)

    plt.show()

    sig_pt = DataP3.sig_pt
    bck_pt = DataP3.bck_pt

    s_w = DataP3.cross_section[-DataP3.length_sig:-1]
    b_w = DataP3.cross_section[0:DataP3.length_bck]

    plot_2_histogram("dataproc_jet_pt_3Prong", sig_pt, bck_pt, s_w, b_w, 50, hist_min=20., hist_max=3300.0)

    bck_weight = DataP3.new_weights[0:DataP3.length_bck]
    sig_weight = DataP3.new_weights[-DataP3.length_sig:-1]

    plot_2_histogram("new_weights_3Prong", sig_weight, bck_weight, np.ones_like(sig_weight), np.ones_like(bck_weight), 75)

    Prong3Model.Model_Fit(256, 100, 0.2, model=Prong3Model.RNNmodel, inputs=Prong3Model.inputs)
    Prong3Model.plot_loss()


    train_real_y, train_pred_y = Prong3Model.get_train_scores(Prong3Model.RNNmodel, Prong3Model.inputs)
    train_weights = Prong3Model.w_train

    Prong3Model.evaluate_model(Prong3Model.eval_inputs, Prong3Model.eval_y, Prong3Model.eval_w, Prong3Model.RNNmodel)

    real_y, pred_y, jet_pt = Prong3Model.predict(Prong3Model.RNNmodel, Prong3Model.eval_inputs)

    back_real_y, back_pred_y, back_jet_pt = Prong3Model.predict_back(Prong3Model.RNNmodel, Prong3Model.eval_back_inputs)

    Prong3Plots = Plots("Prong3Plots", real_y, pred_y, Prong3Model.eval_w, train_real_y, train_pred_y, train_weights,
                        jet_pt)

    back_plots = Plots("P3_backplots", back_real_y, back_pred_y, Prong3Model.eval_back_w, train_real_y, train_pred_y,
                       train_weights, back_jet_pt)

    Prong3Plots.plot_raw_score_vs_jetPT()
    Prong3Plots.histogram_RNN_score()

    back_plots.histogram_RNN_score()

    prong3_rejveff = Prong3Plots.plot_rej_vs_eff("ROC_Prong3", "ROC_curves/", do_train=True)

    plt.draw()
    plt.show()

    sig_train_pt = Prong3Model.train_jet_pt[Prong3Model.train_sigbck_index == "s"]
    sig_train_weight = Prong3Model.w_train[Prong3Model.train_sigbck_index == "s"]
    if pile_up:
        sig_train_mu = Prong3Model.mu_train[Prong3Model.train_sigbck_index == "s"]

    sig_train_score = Prong3Model.RNNmodel.predict([Prong3Model.inputs[0][Prong3Model.train_sigbck_index == "s"],
                                                    Prong3Model.inputs[1][Prong3Model.train_sigbck_index == "s"],
                                                    Prong3Model.inputs[2][Prong3Model.train_sigbck_index == "s"]])
    sig_train_score.flatten()

    bkg_train_score = Prong3Model.RNNmodel.predict([Prong3Model.inputs[0][Prong3Model.train_sigbck_index == "b"],
                                                    Prong3Model.inputs[1][Prong3Model.train_sigbck_index == "b"],
                                                    Prong3Model.inputs[2][Prong3Model.train_sigbck_index == "b"]])
    bkg_train_score.flatten()
    bkg_train_weight = Prong3Model.w_train[Prong3Model.train_sigbck_index == "b"].flatten()
    if pile_up:
        bkg_train_mu = Prong3Model.mu_train[Prong3Model.train_sigbck_index == "b"]

    sig_test_pt = Prong3Model.eval_jet_pt[Prong3Model.eval_sigbck_index == "s"]
    sig_test_weight = Prong3Model.eval_w[Prong3Model.eval_sigbck_index == "s"]

    sig_test_score = pred_y[Prong3Model.eval_sigbck_index == "s"]
    sig_test_score.flatten()
    if pile_up:
        sig_test_mu = Prong3Model.eval_mu[Prong3Model.eval_sigbck_index == "s"]
    bkg_test_pt = Prong3Model.eval_jet_pt[Prong3Model.eval_sigbck_index == "b"]
    bkg_test_weight = Prong3Model.eval_w[Prong3Model.eval_sigbck_index == "b"]

    bkg_test_score = pred_y[Prong3Model.eval_sigbck_index == "b"]
    bkg_test_score.flatten()
    if pile_up:
        bkg_test_mu = Prong3Model.eval_mu[Prong3Model.eval_sigbck_index == "b"]

    # ATLAS Score Plot
    rnn_score = ScorePlot(test=True, train=True, log_y=True)
    rnn_score.plot(sig_train_score, bkg_train_score, sig_train_weight, bkg_train_weight,
                   sig_test_score, bkg_test_score, sig_test_weight, bkg_test_weight, "Prong3", "score_plots/")
    plt.show()
    #  input("Enter..")

    # Test ATLAS plotting algorithm
    if pile_up:
        FEP_tight = FlattenerEfficiencyPlot(sig_train_pt, sig_train_score, sig_train_mu, sig_test_pt, sig_test_score,
                                            sig_test_mu, 60 / 100)
        FEP_fig = FEP_tight.plot()
        plt.show()
        EPs = []
        RPs = []
        plots = []
        plots_idx = 0
        # PT
        sig_test_xvar = Prong3Model.eval_jet_pt[Prong3Model.eval_sigbck_index == "s"]
        bkg_test_xvar = Prong3Model.eval_jet_pt[Prong3Model.eval_sigbck_index == "b"]
        pt_max_sig = np.max(sig_test_xvar)
        pt_min_sig = np.min(sig_test_xvar)

        #        stest_bins_pt = np.percentile(sig_test_xvar, np.linspace(20.0, 400.0, 16))
        stest_bins_pt = np.linspace(20.0, 400.0, 16)
        #       btest_bins_pt = np.percentile(bkg_test_xvar, np.linspace(np.log10(20.), np.log10(250.0), 9))
        btest_bins_pt = 10 ** np.linspace(np.log10(20.0), np.log10(400.0), 9)

        stest_bins_mu = np.linspace(0, 70, 9)
        btest_bins_mu = np.linspace(0, 70, 9)

        stest_bins_eta = np.linspace(0, 2.5, 12)
        btest_bins_eta = np.linspace(0, 2.5, 12)
        sig_test_eta = Prong3Model.eval_kinematic_vars["jet_Eta"][Prong3Model.eval_sigbck_index == "s"]
        bck_test_eta = Prong3Model.eval_kinematic_vars["jet_Eta"][Prong3Model.eval_sigbck_index == "b"]

        stest_bins_phi = np.linspace(0, np.pi, 12)
        btest_bins_phi = np.linspace(0, np.pi, 12)
        sig_test_phi = Prong3Model.eval_kinematic_vars["jet_Phi"][Prong3Model.eval_sigbck_index == "s"]
        bck_test_phi = Prong3Model.eval_kinematic_vars["jet_Phi"][Prong3Model.eval_sigbck_index == "b"]

        eff = [0.95, 0.75, 0.60, 0.45]
        colours = ["red", "green", "violet", "yellow"]
        # for eff in efficiency:
        pt_efficiency = EfficiencyPlot(eff, colours, bins=stest_bins_pt)
        pt_eff_plot = pt_efficiency.plot(sig_train_pt, sig_train_score, sig_train_mu, sig_test_pt, sig_test_score,
                                         sig_test_mu, "3prong_Jet_PT_eff",
                                         sig_test_xvar, sig_test_weight, "efficiencies/")
        pt_rejection = RejectionPlot(eff, colours, bins=btest_bins_pt, ylim=(0, 600))
        pt_rej_plot = pt_rejection.plot(sig_train_pt, sig_train_score, sig_train_mu, sig_test_pt, sig_test_weight,
                                        bkg_test_pt,
                                        bkg_test_weight, bkg_test_score, bkg_test_mu, bkg_test_xvar,
                                        "3prong_Jet_PT_rej", "rejection/")

        mu_efficiency = EfficiencyPlot(eff, colours, bins=stest_bins_mu)
        mu_eff_plot = mu_efficiency.plot(sig_train_pt, sig_train_score, sig_train_mu, sig_test_pt, sig_test_score,
                                         sig_test_mu, "3prong_Mu_eff",
                                         sig_test_mu, sig_test_weight, "efficiencies/")
        mu_rejection = RejectionPlot(eff, colours, bins=btest_bins_mu, ylim=(0, 400))
        mu_rej_plot = mu_rejection.plot(sig_train_pt, sig_train_score, sig_train_mu, sig_test_pt, sig_test_weight,
                                        bkg_test_pt,
                                        bkg_test_weight, bkg_test_score, bkg_test_mu, bkg_test_mu,
                                        "3prong_Mu_eff", "rejection/")
        eta_efficiency = EfficiencyPlot(eff, colours, bins=stest_bins_eta)
        eta_eff_plot = eta_efficiency.plot(sig_train_pt, sig_train_score, sig_train_mu, sig_test_pt, sig_test_score,
                                           sig_test_mu, "3prong_jet_Eta_eff",
                                           sig_test_eta, sig_test_weight, "efficiencies/")
        eta_rejection = RejectionPlot(eff, colours, bins=btest_bins_eta, ylim=(0, 2500))
        eta_rej_plot = eta_rejection.plot(sig_train_pt, sig_train_score, sig_train_mu, sig_test_pt, sig_test_weight,
                                          bkg_test_pt,
                                          bkg_test_weight, bkg_test_score, bkg_test_mu, bck_test_eta,
                                          "3prong_jet_Eta_rej", "rejection/")
        phi_efficiency = EfficiencyPlot(eff, colours, bins=stest_bins_phi)
        phi_eff_plot = phi_efficiency.plot(sig_train_pt, sig_train_score, sig_train_mu, sig_test_pt, sig_test_score,
                                           sig_test_mu, "3prong_jet_Phi_eff",
                                           sig_test_phi, sig_test_weight, "efficiencies/")
        phi_rejection = RejectionPlot(eff, colours, bins=btest_bins_phi, ylim=(0, 3000))
        phi_rej_plot = phi_rejection.plot(sig_train_pt, sig_train_score, sig_train_mu, sig_test_pt, sig_test_weight,
                                          bkg_test_pt,
                                          bkg_test_weight, bkg_test_score, bkg_test_mu, bck_test_phi,
                                          "3prong_jet_Phi_rej", "rejection/")
        plt.show()
        input("enter...")
    else:
        FEP_tight = FlattenerEfficiencyPlot(sig_train_pt, sig_train_score, sig_test_pt, sig_test_score, 60 / 100)
        FEP_fig = FEP_tight.plot()
        plt.show()
        EPs = []
        RPs = []
        plots = []
        plots_idx = 0
        for idx in tqdm.trange(0, len(Prong3Model.inputs[2][0, 1:12])):
            sig_test_xvar = Prong3Model.eval_inputs[2][Prong3Model.eval_sigbck_index == "s", idx]
            bkg_test_xvar = Prong3Model.eval_inputs[2][Prong3Model.eval_sigbck_index == "b", idx]
            EPs.append(EfficiencyPlot(60 / 100))
            plots.append(
                EPs[plots_idx].plot(sig_train_pt, sig_train_score, sig_test_pt, sig_test_score, "xvar_{}".format(idx),
                                    sig_test_xvar))
            RPs.append(RejectionPlot(60 / 100))
            plots.append(RPs[plots_idx].plot(sig_train_pt, sig_train_score, sig_test_pt, sig_test_weight, bkg_test_pt,
                                             bkg_test_weight, bkg_test_score, bkg_test_xvar, "xvar_{}".format(idx)))
        #plt.show()
    input("Press enter to exit")

