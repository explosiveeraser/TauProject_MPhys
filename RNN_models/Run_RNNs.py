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


DataP1 = RNN_Data(1, True, "prong1_data", print_hists=True,
                  BacktreeFile=["0_background_wPU_tree_1-Prong", "1_background_wPU_tree_1-Prong", "2_background_wPU_tree_1-Prong", "3_background_wPU_tree_1-Prong", "4_background_wPU_tree_1-Prong"]
                , BackTreeName=["0_background_wPU_tree", "1_background_wPU_tree", "2_background_wPU_tree", "3_background_wPU_tree", "4_background_wPU_tree"]
                  , SignaltreeFile=["signal_wPU_tree_1-Prong"], SignalTreeName=["signal_wPU_tree"], BackendPartOfTree="", SignalendPartOfTree="")

#print_hist = True

do_RNN = True

#if print_hist:

#    DataP1.plot_hists()


if do_RNN:
    Prong1Model = Tau_Model(1, [DataP1.input_track[:,0:6,:], DataP1.input_tower[:,0:10,:], DataP1.input_jet[:, 1:12]], DataP1.sig_pt, DataP1.bck_pt, DataP1.jet_pt, DataP1.Ytrain, DataP1.new_weights, DataP1.cross_section, DataP1.mu)
    plot_2_histogram("Train_weights-Eval_weights", Prong1Model.w_train, Prong1Model.eval_w, np.ones_like(Prong1Model.w_train), np.ones_like(Prong1Model.eval_w), 70)

    training_sig_pt = Prong1Model.train_jet_pt[Prong1Model.train_sigbck_index == "s"]
    print("{}--------------".format(len(training_sig_pt)))
    training_bck_pt = Prong1Model.train_jet_pt[Prong1Model.train_sigbck_index == "b"]
    train_s_w = Prong1Model.w_train[Prong1Model.train_sigbck_index == "s"]
    train_b_w = Prong1Model.w_train[Prong1Model.train_sigbck_index == "b"]
    plot_2_histogram("Training_Jet_PT", training_sig_pt, training_bck_pt, train_s_w, train_b_w, 50, hist_min=20., hist_max=3300.0)

    eval_sig_pt = Prong1Model.eval_jet_pt[Prong1Model.eval_sigbck_index == "s"]
    eval_bck_pt = Prong1Model.eval_jet_pt[Prong1Model.eval_sigbck_index == "b"]
    eval_s_w = Prong1Model.eval_w[Prong1Model.eval_sigbck_index == "s"]
    eval_b_w = Prong1Model.eval_w[Prong1Model.eval_sigbck_index == "b"]
    plot_2_histogram("Evaluation_Jet_PT", eval_sig_pt, eval_bck_pt, eval_s_w, eval_b_w, 50, hist_min=20., hist_max=3300.0)

    sig_pt = DataP1.sig_pt
    bck_pt = DataP1.bck_pt

    # s_w = DataP1.new_weights[-DataP1.length_sig:-1]
    # b_w = DataP1.new_weights[0:DataP1.length_bck]

    s_w = DataP1.cross_section[-DataP1.length_sig:-1]
    b_w = DataP1.cross_section[0:DataP1.length_bck]

    plot_2_histogram("dataproc_jet_pt", sig_pt, bck_pt, s_w, b_w, 50, hist_min=20., hist_max=3300.0)



    # sig_train_pt = Prong1Model.train_jet_pt[Prong1Model.train_sigbck_index == "s"]
    # sig_train_weight = Prong1Model.w_train[Prong1Model.train_sigbck_index == "s"]
    # sig_train_score = Prong1Model.RNNmodel.predict([Prong1Model.inputs[0][Prong1Model.train_sigbck_index == "s"], Prong1Model.inputs[1][Prong1Model.train_sigbck_index == "s"], Prong1Model.inputs[2][Prong1Model.train_sigbck_index == "s"]])
    # sig_test_pt = Prong1Model.eval_jet_pt[Prong1Model.eval_sigbck_index == "s"]
    # sig_test_weight = Prong1Model.eval_w[Prong1Model.eval_sigbck_index == "s"]
    # bkg_test_pt = Prong1Model.eval_jet_pt[Prong1Model.eval_sigbck_index == "b"]
    # bkg_test_weight = Prong1Model.eval_w[Prong1Model.eval_sigbck_index == "b"]
    #
    # print("sig_train_pt {}".format(sig_train_pt))
    # input("enter")
    # print("sig_train_weight {}".format(sig_train_weight))
    # input("enter")
    # print("sig_train_score {}".format(sig_train_score.flatten()))
    # input("enter")
    # print("sig_test_pt {}".format(sig_test_pt))
    # input("enter")
    # print("sig_test_weight {}".format(sig_test_weight))
    # input("enter")
    # print("bkg_test_pt {}".format(bkg_test_pt))
    # input("enter")
    # print("bkg_test_weight {}".format(bkg_test_weight))
    # input("enter")



    #print(DataP1.input_jet[0,1:12])

    bck_weight = DataP1.new_weights[0:DataP1.length_bck]
    sig_weight = DataP1.new_weights[-DataP1.length_sig:-1]

    plot_2_histogram("new_weights", sig_weight, bck_weight, np.ones_like(sig_weight), np.ones_like(bck_weight), 75)

    #Prong1Model.Model_Fit(256, 40, 0.3, model=Prong1Model.basic_model, inputs=[Prong1Model.inputs[2]])
    #Prong1Model.Model_Fit(256, 40, 0.3, model=Prong1Model.RNNmodel_woTower, inputs=[Prong1Model.inputs[0], Prong1Model.inputs[2]])
    Prong1Model.Model_Fit(256, 100, 0.1, model=Prong1Model.RNNmodel, inputs=Prong1Model.inputs)#, #model=Prong1Model.basic_model, inputs=Prong1Model.inputs[2][:, 0:11])

    #Prong1Model.evaluate_model([DataP1.all_track[:, 0:6, :], DataP1.all_tower[:,0:10,:], DataP1.all_jet], DataP1.all_label, Prong1Model.RNNmodel)

    #Prong1Model.evaluate_model([DataP1.all_track ,D0ataP1.all_jet], DataP1.all_label, Prong1Model.RNNmodel_woTower)

    #Prong1Model.evaluate_model([D0ataP1.all_jet], DataP1.all_label, Prong1Model.basic_model)

    #Prong1Model.plot_accuracy()

    train_real_y, train_pred_y = Prong1Model.get_train_scores(Prong1Model.RNNmodel, Prong1Model.inputs)
    train_weights = Prong1Model.w_train

    # Prong1Model.plot_feature_heatmap({"track_PT", "track_D0", "track_DZ", "track_deltaEta",
    #                 "track_deltaPhi", "tower_ET", "tower_deltaEta", "tower_deltaPhi", "jet_PT", "jet_PT_LC_scale", "jet_f_cent", "jet_iF_leadtrack", "jet_max_deltaR",
    #             "jet_Ftrack_Iso", "jet_ratio_ToEem_P", "jet_frac_trEM_pt", "jet_mass_track_EM_system"
    #             , "jet_mass_track_system"}, Prong1Model.RNNmodel)

    Prong1Model.evaluate_model(Prong1Model.eval_inputs, Prong1Model.eval_y, Prong1Model.eval_w, Prong1Model.RNNmodel)

    real_y, pred_y, jet_pt = Prong1Model.predict(Prong1Model.RNNmodel, Prong1Model.eval_inputs)
    #print(pred_y)

    back_real_y, back_pred_y, back_jet_pt = Prong1Model.predict_back(Prong1Model.RNNmodel, Prong1Model.eval_back_inputs)

    #weights = Prong1Model.get_score_weights()

    Prong1Plots = Plots( "Prong1Plots", real_y, pred_y, Prong1Model.eval_w, train_real_y, train_pred_y, train_weights, jet_pt)

    back_plots = Plots("P1_backplots", back_real_y, back_pred_y, Prong1Model.eval_back_w, train_real_y, train_pred_y, train_weights, back_jet_pt)

    Prong1Plots.plot_raw_score_vs_jetPT()
    Prong1Plots.histogram_RNN_score()

    back_plots.histogram_RNN_score()

    prong1_rejveff = Prong1Plots.plot_rej_vs_eff()

    #plot_graph( "ROC_Curve_for_1_prong_RNN", Prong1Plots.eff[Prong1Plots.rej <= 10000], Prong1Plots.rej[Prong1Plots.rej <= 10000], len(Prong1Plots.rej[Prong1Plots.rej <= 10000]))

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
    rnn_score = ScorePlot(test=True, log_y=True)
    rnn_score.plot(sig_train_score, bkg_train_score, sig_train_weight, bkg_train_weight,
                   sig_test_score, bkg_test_score, sig_test_weight, bkg_test_weight)
    plt.show()
    input("Enter..")

    #Test ATLAS plotting algorithm
    if pile_up:
        FEP_tight = FlattenerEfficiencyPlot(sig_train_pt, sig_train_score, sig_train_mu, sig_test_pt, sig_test_score, sig_test_mu, 60 / 100)
        FEP_fig = FEP_tight.plot()
        plt.show()
        EPs = []
        RPs = []
        plots = []
        plots_idx = 0
        for idx in tqdm.trange(0, len(Prong1Model.inputs[2][0, 1:12])):
            sig_test_xvar = Prong1Model.eval_inputs[2][Prong1Model.eval_sigbck_index == "s", idx]
            bkg_test_xvar = Prong1Model.eval_inputs[2][Prong1Model.eval_sigbck_index == "b", idx]
            EPs.append(EfficiencyPlot(60 / 100))
            plots.append(
                EPs[plots_idx].plot(sig_train_pt, sig_train_score, sig_train_mu, sig_test_pt, sig_test_score, sig_test_mu, "xvar_{}".format(idx),
                                    sig_test_xvar))
            RPs.append(RejectionPlot(60 / 100))
            plots.append(RPs[plots_idx].plot(sig_train_pt, sig_train_score, sig_train_mu, sig_test_pt, sig_test_weight, bkg_test_pt,
                                             bkg_test_weight, bkg_test_score, bkg_test_mu, bkg_test_xvar, "xvar_{}".format(idx)))
        plt.show()
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




"""
    input("Press enter to see Prong 3")

    DataP3 = RNN_Data(3, False, "prong3_data", BacktreeFile="background_tree_3-Prong", BackTreeName="background_tree", SignaltreeFile="signal_tree_3-Prong", SignalTreeName="signal_tree", BackendPartOfTree=";9", SignalendPartOfTree=";2")

    Prong3Model = Tau_Model(3, [DataP3.input_track, DataP3.input_tower, DataP3.input_jet], DataP3.Ytrain)

    # Prong3Model.Model_Fit(256, 40, 0.3, model=Prong3Model.basic_model, inputs=[Prong3Model.inputs[2]])
    # Prong3Model.Model_Fit(256, 40, 0.3, model=Prong3Model.RNNmodel_woTower, inputs=[Prong3Model.inputs[0], Prong3Model.inputs[2]])
    Prong3Model.Model_Fit(256, 15, 0.1)

    Prong3Model.evaluate_model([DataP3.all_track[:, 0:6, :], DataP3.all_tower[:, 0:10, :], DataP3.all_jet],
                               DataP3.all_label, Prong3Model.RNNmodel)
    # Prong3Model.evaluate_model([DataP3.all_track ,D0ataP3.all_jet], DataP3.all_label, Prong3Model.RNNmodel_woTower)
    # Prong3Model.evaluate_model([D0ataP3.all_jet], DataP3.all_label, Prong3Model.basic_model)

    Prong3Model.plot_accuracy()
    input("Press enter to exit")
"""
