# import keras_preprocessing.sequence
# import matplotlib.pyplot as plt
# import numpy as np
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

from RNN_model import Tau_Model
from RNN_Data import RNN_Data
import root_numpy as rn
#from rootpy.plotting import Hist


DataP1 = RNN_Data(1, True, "prong1_data", BacktreeFile="background_tree_1-Prong", BackTreeName="background_tree", SignaltreeFile="signal_tree_1-Prong", SignalTreeName="signal_tree", BackendPartOfTree=";9", SignalendPartOfTree=";2")

print_hist = False

do_RNN = True

if print_hist:
    hists = {}
    #print(DataP1.input_jet[DataP1.Ytrain == 1])

    legendJ = []
    legendTr = []
    legendTo = []

    jet_i = 0

    jet_canvas = ROOT.TCanvas("Jet_Inputs", "Jet_Inputs")
    jet_canvas.Divide(4, 3)
    jet_canvas.cd(0)
    jet_canvas.cd(1)

    for key in {"jet_PT", "jet_Eta", "jet_Phi", "jet_deltaEta", "jet_deltaPhi", "jet_deltaR",
                             "jet_charge", "jet_NCharged", "jet_NNeutral", "jet_f_cent", "jet_iF_leadtrack", "jet_max_deltaR",
                             "jet_Ftrack_Iso"}:
        legendJ.append(ROOT.TLegend(0.05, 0.85, 0.2, 0.95))
        for l in [0, 1]:
            arr_ = DataP1.input_jet[DataP1.Ytrain == l, jet_i]
            arr = arr_.flatten().flatten()
            arr = arr[np.abs(arr) > 0.000000000001]
            #print(arr)
            hists["{}_label{}".format(key, str(l))] = ROOT.TH1D("{}_{}".format(key, l), "{}_{}".format(key, l), 68, -5., 5.)
            rn.fill_hist(hists["{}_label{}".format(key, str(l))], arr)
            if l == 0:
                hists["{}_label{}".format(key, str(l))].Draw("HIST")
            else:
                hists["{}_label{}".format(key, str(l))].SetLineColor(ROOT.kRed)
                hists["{}_label{}".format(key, str(l))].Draw("HIST SAMES0")
            legendJ[jet_i].AddEntry(hists["{}_label{}".format(key, l)], "{} Label: {}".format(key, l), "L")
        legendJ[jet_i].Draw()
        jet_canvas.Update()
        jet_i += 1
        jet_canvas.cd(jet_i+1)

    track_i = 0

    track_canvas = ROOT.TCanvas("Track_Inputs", "Track_Inputs")
    track_canvas.Divide(3, 3)
    track_canvas.cd(0)
    track_canvas.cd(1)

    print(np.shape(DataP1.input_track[DataP1.Ytrain == 1, :, 1]))

    for key in {"track_P", "track_PT", "track_L", "track_D0", "track_DZ", "track_deltaEta",
                              "track_deltaPhi", "track_deltaR"}:
        legendTr.append(ROOT.TLegend(0.05, 0.85, 0.2, 0.95))
        for l in [0, 1]:
            arr = DataP1.input_track[DataP1.Ytrain == l, 0:6, track_i]
            arr = np.ravel(arr, 'F')
            arr = arr[np.abs(arr) > 0.000000000001]
            #print(arr)
            hists["{}_label{}".format(key, str(l))] = ROOT.TH1D("{}_{}".format(key, l), "{}_{}".format(key, l), 68, -5.,
                                                                5.)
            rn.fill_hist(hists["{}_label{}".format(key, str(l))], arr)
            if l == 0:
                hists["{}_label{}".format(key, str(l))].Draw("HIST")
            else:
                hists["{}_label{}".format(key, str(l))].SetLineColor(ROOT.kRed)
                hists["{}_label{}".format(key, str(l))].Draw("HIST SAMES0")
            legendTr[track_i].AddEntry(hists["{}_label{}".format(key, l)], "{} Label: {}".format(key, l), "L")
        legendTr[track_i].Draw()
        track_canvas.Update()
        track_i += 1
        track_canvas.cd(track_i+1)

    tower_i = 0

    tower_canvas = ROOT.TCanvas("Tower_Inputs", "Tower_Inputs")
    tower_canvas.Divide(4, 4)
    tower_canvas.cd(0)
    tower_canvas.cd(1)

    for key in {"tower_E", "tower_ET", "tower_Eta", "tower_Phi", "tower_Edges0", "tower_Edges1", "tower_Edges2",
                              "tower_Edges3", "tower_Eem", "tower_Ehad", "tower_T", "tower_deltaEta", "tower_deltaPhi", "tower_deltaR"}:
        legendTo.append(ROOT.TLegend(0.05, 0.85, 0.2, 0.95))
        for l in [0, 1]:
            arr = DataP1.input_tower[DataP1.Ytrain == l, 0:10, tower_i]
            arr = np.ravel(arr, 'F')
            arr = arr[np.abs(arr) > 0.000000000001]
            hists["{}_label{}".format(key, str(l))] = ROOT.TH1D("{}_{}".format(key, l), "{}_{}".format(key, l), 68, -5.,
                                                                5.)
            rn.fill_hist(hists["{}_label{}".format(key, str(l))], arr)
            if l == 0:
                hists["{}_label{}".format(key, str(l))].Draw("HIST")
            else:
                hists["{}_label{}".format(key, str(l))].SetLineColor(ROOT.kRed)
                hists["{}_label{}".format(key, str(l))].Draw("HIST SAMES0")
            legendTo[tower_i].AddEntry(hists["{}_label{}".format(key, l)], "{} Label: {}".format(key, l), "L")
        legendTo[tower_i].Draw()
        tower_canvas.Update()
        tower_i += 1
        tower_canvas.cd(tower_i+1)

    jet_canvas.Print("Jet_Inputs.pdf")
    track_canvas.Print("Track_Inputs.pdf")
    tower_canvas.Print("Tower_Inputs.pdf")

    input("Enter to continue")

if do_RNN:
    Prong1Model = Tau_Model(1, [DataP1.input_track[:,0:6,:], DataP1.input_tower[:,0:10,:], DataP1.input_jet], DataP1.Ytrain)

    #Prong1Model.Model_Fit(256, 40, 0.3, model=Prong1Model.basic_model, inputs=[Prong1Model.inputs[2]])
    #Prong1Model.Model_Fit(256, 40, 0.3, model=Prong1Model.RNNmodel_woTower, inputs=[Prong1Model.inputs[0], Prong1Model.inputs[2]])
    Prong1Model.Model_Fit(256, 15, 0.2)

    Prong1Model.evaluate_model([DataP1.all_track[:,0:6,:], DataP1.all_tower[:,0:10,:], DataP1.all_jet], DataP1.all_label, Prong1Model.RNNmodel)
    #Prong1Model.evaluate_model([DataP1.all_track ,D0ataP1.all_jet], DataP1.all_label, Prong1Model.RNNmodel_woTower)
    #Prong1Model.evaluate_model([D0ataP1.all_jet], DataP1.all_label, Prong1Model.basic_model)

    Prong1Model.plot_accuracy()
    input("Press enter to see Prong 3")

    DataP3 = RNN_Data(3, True, "prong3_data", BacktreeFile="background_tree_3-Prong", BackTreeName="background_tree", SignaltreeFile="signal_tree_3-Prong", SignalTreeName="signal_tree", BackendPartOfTree=";9", SignalendPartOfTree=";2")

    Prong3Model = Tau_Model(3, [DataP3.input_track, DataP3.input_tower, DataP3.input_jet], DataP3.Ytrain)

    # Prong3Model.Model_Fit(256, 40, 0.3, model=Prong3Model.basic_model, inputs=[Prong3Model.inputs[2]])
    # Prong3Model.Model_Fit(256, 40, 0.3, model=Prong3Model.RNNmodel_woTower, inputs=[Prong3Model.inputs[0], Prong3Model.inputs[2]])
    Prong3Model.Model_Fit(256, 15, 0.2)

    Prong3Model.evaluate_model([DataP3.all_track[:, 0:6, :], DataP3.all_tower[:, 0:10, :], DataP3.all_jet],
                               DataP3.all_label, Prong3Model.RNNmodel)
    # Prong3Model.evaluate_model([DataP3.all_track ,D0ataP3.all_jet], DataP3.all_label, Prong3Model.RNNmodel_woTower)
    # Prong3Model.evaluate_model([D0ataP3.all_jet], DataP3.all_label, Prong3Model.basic_model)

    Prong3Model.plot_accuracy()
    input("Press enter to exit")
