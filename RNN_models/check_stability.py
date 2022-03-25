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
                  BacktreeFile=["0-1_background_wPU_tree_1-Prong", "1-1_background_wPU_tree_1-Prong", "2-1_background_wPU_tree_1-Prong",
                                "3-1_background_wPU_tree_1-Prong", "4-1_background_wPU_tree_1-Prong", "0-2_background_wPU_tree_1-Prong",
                                "1-2_background_wPU_tree_1-Prong", "2-2_background_wPU_tree_1-Prong", "3-2_background_wPU_tree_1-Prong",
                                "4-2_background_wPU_tree_1-Prong"]
                , BackTreeName=["0-1_background_wPU_tree", "1-1_background_wPU_tree", "2-1_background_wPU_tree",
                                "3-1_background_wPU_tree", "4-1_background_wPU_tree", "0-2_background_wPU_tree",
                                "1-2_background_wPU_tree", "2-2_background_wPU_tree", "3-2_background_wPU_tree",
                                "4-2_background_wPU_tree"]
                  , SignaltreeFile=["0_signal_wPU_tree_1-Prong", "1_signal_wPU_tree_1-Prong"],
                  SignalTreeName=["0_signal_wPU_tree", "1_signal_wPU_tree"], BackendPartOfTree="", SignalendPartOfTree="")

DataP3 = RNN_Data(3, False, "prong3_data", print_hists=False,
                  BacktreeFile=["0-1_background_wPU_tree_3-Prong", "1-1_background_wPU_tree_3-Prong", "2-1_background_wPU_tree_3-Prong",
                                "3-1_background_wPU_tree_3-Prong", "4-1_background_wPU_tree_3-Prong", "0-2_background_wPU_tree_3-Prong",
                                "1-2_background_wPU_tree_3-Prong", "2-2_background_wPU_tree_3-Prong", "3-2_background_wPU_tree_3-Prong",
                                "4-2_background_wPU_tree_3-Prong"]
                , BackTreeName=["0-1_background_wPU_tree", "1-1_background_wPU_tree", "2-1_background_wPU_tree",
                                "3-1_background_wPU_tree", "4-1_background_wPU_tree", "0-2_background_wPU_tree",
                                "1-2_background_wPU_tree", "2-2_background_wPU_tree", "3-2_background_wPU_tree",
                                "4-2_background_wPU_tree"]
                  , SignaltreeFile=["0_signal_wPU_tree_3-Prong", "1_signal_wPU_tree_3-Prong"],
                  SignalTreeName=["0_signal_wPU_tree", "1_signal_wPU_tree"], BackendPartOfTree="", SignalendPartOfTree="")

do_same_seed = True
do_random_seed = True
do_prong1 = False
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

        for i in range(0, 40):
            Prong1Model.append(Tau_Model(1, [DataP1.input_track[:,0:6,:], DataP1.input_tower[:,0:10,:], DataP1.input_jet[:, 1:12]], DataP1.sig_pt, DataP1.bck_pt, DataP1.jet_pt, DataP1.Ytrain, DataP1.new_weights, DataP1.cross_section, DataP1.mu, kinematic_vars={"jet_Eta" : DataP1.input_jet[:, 12], "jet_Phi" : DataP1.input_jet[:, 13]}))
            Prong1Model[i].Model_Fit(256, 100, 0.2, model=Prong1Model[i].RNNmodel, inputs=Prong1Model[i].inputs, addition_savename="RandomSeeded{}_".format(i))
            #Prong1Model[i].load_model("RandomSeeded{}_RNN_Model_Prong-1.h5".format(i))
            Prong1Model[i].evaluate_model(Prong1Model[i].eval_inputs, Prong1Model[i].eval_y, Prong1Model[i].eval_w, Prong1Model[i].RNNmodel)
            tr, tp = Prong1Model[i].get_train_scores(Prong1Model[i].RNNmodel, Prong1Model[i].inputs)
            train_real_y_p1.append(tr)
            train_pred_y_p1.append(tp)
            train_weights_p1.append(Prong1Model[i].w_train)
            Prong1Model[i].evaluate_model(Prong1Model[i].eval_inputs, Prong1Model[i].eval_y, Prong1Model[i].eval_w, Prong1Model[i].RNNmodel)
            ry, py, jpt = Prong1Model[i].predict(Prong1Model[i].RNNmodel, Prong1Model[i].eval_inputs)
            real_y_p1.append(ry)
            pred_y_p1.append(py)
            jet_pt_p1.append(jpt)
            Prong1Plots.append(Plots( "Prong1Plots_randomseed_{}".format(i), real_y_p1[i], pred_y_p1[i], Prong1Model[i].eval_w, train_real_y_p1[i], train_pred_y_p1[i], train_weights_p1[i],
                                      [i]))
            prong1_rejvseff.append(Prong1Plots[i].plot_rej_vs_eff("ROC_Prong1_randomseed_{}".format(i), "ROC_curves/", do_train=True))
            plt.close(prong1_rejvseff[i])
            ax.plot(Prong1Plots[i].eff, Prong1Plots[i].rej, "-", label="Test {} (Evaluation Sample)".format(i))
        ax.set_xlim((0., 1.))
        ax.set_ylim((1., 1e4))
        ax.set_yscale("log")
        ax.set_xlabel("Signal efficiency", x=1, ha="right")
        ax.set_ylabel("Background rejection", y=1, ha="right")
        #ax.legend(fontsize='xx-small')
        plt.savefig("ROC_curves/P1_Test_RandomSeed_Stability.png")
        #plt.show()
        plt.close(fig)
        fig_train, ax_train = plt.subplots()
        for i in range(0, 40):
            ax_train.plot(Prong1Plots[i].eff_train, Prong1Plots[i].rej_train, "--", label="Test {} (Training Sample)".format(i))
        ax_train.set_xlim((0., 1.))
        ax_train.set_ylim((1., 1e4))
        ax_train.set_yscale("log")
        ax_train.set_xlabel("Signal efficiency", x=1, ha="right")
        ax_train.set_ylabel("Background rejection", y=1, ha="right")
        plt.savefig("ROC_curves/P1_Train_RandomSeed_Stability.png")
        #plt.show()
        plt.close(fig_train)
        fig_all, ax_all = plt.subplots()
        for i in range(0, 40):
            ax_all.plot(Prong1Plots[i].eff, Prong1Plots[i].rej, "-", color="g", label="Test {} (Training Sample)".format(i))
            ax_all.plot(Prong1Plots[i].eff_train, Prong1Plots[i].rej_train, "-", color="b", label="Test {} (Training Sample)".format(i))
        ax_all.set_xlim((0., 1.))
        ax_all.set_ylim((1., 1e4))
        ax_all.set_yscale("log")
        ax_all.set_xlabel("Signal efficiency", x=1, ha="right")
        ax_all.set_ylabel("Background rejection", y=1, ha="right")
        plt.savefig("ROC_curves/P1_ALL_RandomSeed_Stability.png")
        #plt.show()
        plt.close(fig_all)

    if do_same_seed:
        Prong1Model_1 = Tau_Model(1, [DataP1.input_track[:,0:6,:], DataP1.input_tower[:,0:10,:], DataP1.input_jet[:, 1:12]], DataP1.sig_pt, DataP1.bck_pt, DataP1.jet_pt, DataP1.Ytrain, DataP1.new_weights, DataP1.cross_section, DataP1.mu, kinematic_vars={"jet_Eta" : DataP1.input_jet[:, 12], "jet_Phi" : DataP1.input_jet[:, 13]}, shuffle_seed=1521)
        Prong1Model_2 = Tau_Model(1, [DataP1.input_track[:,0:6,:], DataP1.input_tower[:,0:10,:], DataP1.input_jet[:, 1:12]], DataP1.sig_pt, DataP1.bck_pt, DataP1.jet_pt, DataP1.Ytrain, DataP1.new_weights, DataP1.cross_section, DataP1.mu, kinematic_vars={"jet_Eta" : DataP1.input_jet[:, 12], "jet_Phi" : DataP1.input_jet[:, 13]}, shuffle_seed=1521)
        Prong1Model_3 = Tau_Model(1, [DataP1.input_track[:,0:6,:], DataP1.input_tower[:,0:10,:], DataP1.input_jet[:, 1:12]], DataP1.sig_pt, DataP1.bck_pt, DataP1.jet_pt, DataP1.Ytrain, DataP1.new_weights, DataP1.cross_section, DataP1.mu, kinematic_vars={"jet_Eta" : DataP1.input_jet[:, 12], "jet_Phi" : DataP1.input_jet[:, 13]}, shuffle_seed=1521)
        Prong1Model_4 = Tau_Model(1, [DataP1.input_track[:,0:6,:], DataP1.input_tower[:,0:10,:], DataP1.input_jet[:, 1:12]], DataP1.sig_pt, DataP1.bck_pt, DataP1.jet_pt, DataP1.Ytrain, DataP1.new_weights, DataP1.cross_section, DataP1.mu, kinematic_vars={"jet_Eta" : DataP1.input_jet[:, 12], "jet_Phi" : DataP1.input_jet[:, 13]}, shuffle_seed=1521)
        Prong1Model_5 = Tau_Model(1, [DataP1.input_track[:,0:6,:], DataP1.input_tower[:,0:10,:], DataP1.input_jet[:, 1:12]], DataP1.sig_pt, DataP1.bck_pt, DataP1.jet_pt, DataP1.Ytrain, DataP1.new_weights, DataP1.cross_section, DataP1.mu, kinematic_vars={"jet_Eta" : DataP1.input_jet[:, 12], "jet_Phi" : DataP1.input_jet[:, 13]}, shuffle_seed=1521)

        Prong1Model_1.Model_Fit(256, 100, 0.2, model=Prong1Model_1.RNNmodel, inputs=Prong1Model_1.inputs, addition_savename="1_")
        Prong1Model_2.Model_Fit(256, 100, 0.2, model=Prong1Model_2.RNNmodel, inputs=Prong1Model_2.inputs, addition_savename="2_")
        Prong1Model_3.Model_Fit(256, 100, 0.2, model=Prong1Model_3.RNNmodel, inputs=Prong1Model_3.inputs, addition_savename="3_")
        Prong1Model_4.Model_Fit(256, 100, 0.2, model=Prong1Model_4.RNNmodel, inputs=Prong1Model_4.inputs, addition_savename="4_")
        Prong1Model_5.Model_Fit(256, 100, 0.2, model=Prong1Model_5.RNNmodel, inputs=Prong1Model_5.inputs, addition_savename="5_")

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



        Prong1Plots_1 = Plots( "Prong1Plots_1", real_y_p1_1, pred_y_p1_1, Prong1Model_1.eval_w, train_real_y_p1_1, train_pred_y_p1_1, train_weights_p1_1, jet_pt_p1_1)
        Prong1Plots_2 = Plots( "Prong1Plots_2", real_y_p1_2, pred_y_p1_2, Prong1Model_2.eval_w, train_real_y_p1_2, train_pred_y_p1_2, train_weights_p1_2, jet_pt_p1_2)
        Prong1Plots_3 = Plots( "Prong1Plots_3", real_y_p1_3, pred_y_p1_3, Prong1Model_3.eval_w, train_real_y_p1_3, train_pred_y_p1_3, train_weights_p1_3, jet_pt_p1_3)
        Prong1Plots_4 = Plots( "Prong1Plots_4", real_y_p1_4, pred_y_p1_4, Prong1Model_4.eval_w, train_real_y_p1_4, train_pred_y_p1_4, train_weights_p1_4, jet_pt_p1_4)
        Prong1Plots_5 = Plots( "Prong1Plots_5", real_y_p1_5, pred_y_p1_5, Prong1Model_5.eval_w, train_real_y_p1_5, train_pred_y_p1_5, train_weights_p1_5, jet_pt_p1_5)

        prong1_rejveff_1 = Prong1Plots_1.plot_rej_vs_eff("ROC_Prong1_1", "ROC_curves/", do_train=True)
        prong1_rejveff_2 = Prong1Plots_2.plot_rej_vs_eff("ROC_Prong1_2", "ROC_curves/", do_train=True)
        prong1_rejveff_3 = Prong1Plots_3.plot_rej_vs_eff("ROC_Prong1_3", "ROC_curves/", do_train=True)
        prong1_rejveff_4 = Prong1Plots_4.plot_rej_vs_eff("ROC_Prong1_4", "ROC_curves/", do_train=True)
        prong1_rejveff_5 = Prong1Plots_5.plot_rej_vs_eff("ROC_Prong1_5", "ROC_curves/", do_train=True)

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
        ax.set_xlabel("Signal efficiency", x=1, ha="right")
        ax.set_ylabel("Background rejection", y=1, ha="right")
        ax.legend()
        plt.savefig("ROC_curves/P1_Stability_Test.png")

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

        for i in range(0, 40):
            Prong3Model.append(Tau_Model(3, [DataP3.input_track[:,0:6,:], DataP3.input_tower[:,0:10,:], DataP3.input_jet[:, 1:12]], DataP3.sig_pt, DataP3.bck_pt, DataP3.jet_pt, DataP3.Ytrain, DataP3.new_weights, DataP3.cross_section, DataP3.mu, kinematic_vars={"jet_Eta" : DataP3.input_jet[:, 12], "jet_Phi" : DataP3.input_jet[:, 13]}))
            Prong3Model[i].Model_Fit(256, 100, 0.2, model=Prong3Model[i].RNNmodel, inputs=Prong3Model[i].inputs, addition_savename="RandomSeeded{}_".format(i))
            #Prong3Model[i].load_model("RandomSeeded{}_RNN_Model_Prong-3.h5".format(i))
            Prong3Model[i].evaluate_model(Prong3Model[i].eval_inputs, Prong3Model[i].eval_y, Prong3Model[i].eval_w, Prong3Model[i].RNNmodel)
            tr, tp = Prong3Model[i].get_train_scores(Prong3Model[i].RNNmodel, Prong3Model[i].inputs)
            train_real_y_p3.append(tr)
            train_pred_y_p3.append(tp)
            train_weights_p3.append(Prong3Model[i].w_train)
            Prong3Model[i].evaluate_model(Prong3Model[i].eval_inputs, Prong3Model[i].eval_y, Prong3Model[i].eval_w, Prong3Model[i].RNNmodel)
            ry, py, jpt = Prong3Model[i].predict(Prong3Model[i].RNNmodel, Prong3Model[i].eval_inputs)
            real_y_p3.append(ry)
            pred_y_p3.append(py)
            jet_pt_p3.append(jpt)
            Prong3Plots.append(Plots( "Prong3Plots_randomseed_{}".format(i), real_y_p3[i], pred_y_p3[i], Prong3Model[i].eval_w, train_real_y_p3[i], train_pred_y_p3[i], train_weights_p3[i], jet_pt_p3[i]))
            Prong3_rejvseff.append(Prong3Plots[i].plot_rej_vs_eff("ROC_Prong3_randomseed_{}".format(i), "ROC_curves/", do_train=True))
            plt.close(Prong3_rejvseff[i])
            ax.plot(Prong3Plots[i].eff, Prong3Plots[i].rej, "-", label="Test {} (Evaluation Sample)".format(i))
        ax.set_xlim((0., 1.))
        ax.set_ylim((1., 1e4))
        ax.set_yscale("log")
        ax.set_xlabel("Signal efficiency", x=1, ha="right")
        ax.set_ylabel("Background rejection", y=1, ha="right")
        #ax.legend(fontsize='xx-small')
        plt.savefig("ROC_curves/P3_Test_RandomSeed_Stability.png")
        #plt.show()
        plt.close(fig)
        fig_train, ax_train = plt.subplots()
        for i in range(0, 40):
            ax_train.plot(Prong3Plots[i].eff_train, Prong3Plots[i].rej_train, "--", label="Test {} (Training Sample)".format(i))
        ax_train.set_xlim((0., 1.))
        ax_train.set_ylim((1., 1e4))
        ax_train.set_yscale("log")
        ax_train.set_xlabel("Signal efficiency", x=1, ha="right")
        ax_train.set_ylabel("Background rejection", y=1, ha="right")
        plt.savefig("ROC_curves/P3_Train_RandomSeed_Stability.png")
        #plt.show()
        plt.close(fig_train)
        fig_all, ax_all = plt.subplots()
        for i in range(0, 40):
            ax_all.plot(Prong3Plots[i].eff, Prong3Plots[i].rej, "-", color="g", label="Test {} (Training Sample)".format(i))
            ax_all.plot(Prong3Plots[i].eff_train, Prong3Plots[i].rej_train, "-", color="b", label="Test {} (Training Sample)".format(i))
        ax_all.set_xlim((0., 1.))
        ax_all.set_ylim((1., 1e4))
        ax_all.set_yscale("log")
        ax_all.set_xlabel("Signal efficiency", x=1, ha="right")
        ax_all.set_ylabel("Background rejection", y=1, ha="right")
        plt.savefig("ROC_curves/P3_ALL_RandomSeed_Stability.png")
        #plt.show()
        plt.close(fig_all)

    if do_same_seed:
        Prong3Model_1 = Tau_Model(3, [DataP3.input_track[:,0:6,:], DataP3.input_tower[:,0:10,:], DataP3.input_jet[:, 1:12]], DataP3.sig_pt, DataP3.bck_pt, DataP3.jet_pt, DataP3.Ytrain, DataP3.new_weights, DataP3.cross_section, DataP3.mu, kinematic_vars={"jet_Eta" : DataP3.input_jet[:, 12], "jet_Phi" : DataP3.input_jet[:, 13]}, shuffle_seed=1521)
        Prong3Model_2 = Tau_Model(3, [DataP3.input_track[:,0:6,:], DataP3.input_tower[:,0:10,:], DataP3.input_jet[:, 1:12]], DataP3.sig_pt, DataP3.bck_pt, DataP3.jet_pt, DataP3.Ytrain, DataP3.new_weights, DataP3.cross_section, DataP3.mu, kinematic_vars={"jet_Eta" : DataP3.input_jet[:, 12], "jet_Phi" : DataP3.input_jet[:, 13]}, shuffle_seed=1521)
        Prong3Model_3 = Tau_Model(3, [DataP3.input_track[:,0:6,:], DataP3.input_tower[:,0:10,:], DataP3.input_jet[:, 1:12]], DataP3.sig_pt, DataP3.bck_pt, DataP3.jet_pt, DataP3.Ytrain, DataP3.new_weights, DataP3.cross_section, DataP3.mu, kinematic_vars={"jet_Eta" : DataP3.input_jet[:, 12], "jet_Phi" : DataP3.input_jet[:, 13]}, shuffle_seed=1521)
        Prong3Model_4 = Tau_Model(3, [DataP3.input_track[:,0:6,:], DataP3.input_tower[:,0:10,:], DataP3.input_jet[:, 1:12]], DataP3.sig_pt, DataP3.bck_pt, DataP3.jet_pt, DataP3.Ytrain, DataP3.new_weights, DataP3.cross_section, DataP3.mu, kinematic_vars={"jet_Eta" : DataP3.input_jet[:, 12], "jet_Phi" : DataP3.input_jet[:, 13]}, shuffle_seed=1521)
        Prong3Model_5 = Tau_Model(3, [DataP3.input_track[:,0:6,:], DataP3.input_tower[:,0:10,:], DataP3.input_jet[:, 1:12]], DataP3.sig_pt, DataP3.bck_pt, DataP3.jet_pt, DataP3.Ytrain, DataP3.new_weights, DataP3.cross_section, DataP3.mu, kinematic_vars={"jet_Eta" : DataP3.input_jet[:, 12], "jet_Phi" : DataP3.input_jet[:, 13]}, shuffle_seed=1521)

        Prong3Model_1.Model_Fit(256, 100, 0.2, model=Prong3Model_1.RNNmodel, inputs=Prong3Model_1.inputs, addition_savename="1_")
        Prong3Model_2.Model_Fit(256, 100, 0.2, model=Prong3Model_2.RNNmodel, inputs=Prong3Model_2.inputs, addition_savename="2_")
        Prong3Model_3.Model_Fit(256, 100, 0.2, model=Prong3Model_3.RNNmodel, inputs=Prong3Model_3.inputs, addition_savename="3_")
        Prong3Model_4.Model_Fit(256, 100, 0.2, model=Prong3Model_4.RNNmodel, inputs=Prong3Model_4.inputs, addition_savename="4_")
        Prong3Model_5.Model_Fit(256, 100, 0.2, model=Prong3Model_5.RNNmodel, inputs=Prong3Model_5.inputs, addition_savename="5_")

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



        Prong3Plots_1 = Plots( "Prong3Plots_1", real_y_p3_1, pred_y_p3_1, Prong3Model_1.eval_w, train_real_y_p3_1, train_pred_y_p3_1, train_weights_p3_1, jet_pt_p3_1)
        Prong3Plots_2 = Plots( "Prong3Plots_2", real_y_p3_2, pred_y_p3_2, Prong3Model_2.eval_w, train_real_y_p3_2, train_pred_y_p3_2, train_weights_p3_2, jet_pt_p3_2)
        Prong3Plots_3 = Plots( "Prong3Plots_3", real_y_p3_3, pred_y_p3_3, Prong3Model_3.eval_w, train_real_y_p3_3, train_pred_y_p3_3, train_weights_p3_3, jet_pt_p3_3)
        Prong3Plots_4 = Plots( "Prong3Plots_4", real_y_p3_4, pred_y_p3_4, Prong3Model_4.eval_w, train_real_y_p3_4, train_pred_y_p3_4, train_weights_p3_4, jet_pt_p3_4)
        Prong3Plots_5 = Plots( "Prong3Plots_5", real_y_p3_5, pred_y_p3_5, Prong3Model_5.eval_w, train_real_y_p3_5, train_pred_y_p3_5, train_weights_p3_5, jet_pt_p3_5)

        Prong3_rejveff_1 = Prong3Plots_1.plot_rej_vs_eff("ROC_Prong3_1", "ROC_curves/", do_train=True)
        Prong3_rejveff_2 = Prong3Plots_2.plot_rej_vs_eff("ROC_Prong3_2", "ROC_curves/", do_train=True)
        Prong3_rejveff_3 = Prong3Plots_3.plot_rej_vs_eff("ROC_Prong3_3", "ROC_curves/", do_train=True)
        Prong3_rejveff_4 = Prong3Plots_4.plot_rej_vs_eff("ROC_Prong3_4", "ROC_curves/", do_train=True)
        Prong3_rejveff_5 = Prong3Plots_5.plot_rej_vs_eff("ROC_Prong3_5", "ROC_curves/", do_train=True)

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
        plt.savefig("ROC_curves/P3_Stability_Test.png")