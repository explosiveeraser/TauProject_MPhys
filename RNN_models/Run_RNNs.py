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

from RNN_model import Tau_Model
from RNN_Data import RNN_Data
import root_numpy as rn
#from rootpy.plotting import Hist
from Plots import Plots


DataP1 = RNN_Data(1, False, "prong1_data", print_hists=True, BacktreeFile="background_tree_1-Prong", BackTreeName="background_tree", SignaltreeFile="signal_tree_1-Prong", SignalTreeName="signal_tree", BackendPartOfTree="", SignalendPartOfTree="")

#print_hist = True

do_RNN = True

#if print_hist:

#    DataP1.plot_hists()


if do_RNN:
    Prong1Model = Tau_Model(1, [DataP1.input_track[:,0:6,:], DataP1.input_tower[:,0:10,:], DataP1.input_jet[:, 1:12]], DataP1.sig_pt, DataP1.bck_pt, DataP1.jet_pt, DataP1.Ytrain, DataP1.new_weights)

    #Prong1Model.Model_Fit(256, 40, 0.3, model=Prong1Model.basic_model, inputs=[Prong1Model.inputs[2]])
    #Prong1Model.Model_Fit(256, 40, 0.3, model=Prong1Model.RNNmodel_woTower, inputs=[Prong1Model.inputs[0], Prong1Model.inputs[2]])
    Prong1Model.Model_Fit(256, 1000, 0.1)

    #Prong1Model.evaluate_model([DataP1.all_track[:, 0:6, :], DataP1.all_tower[:,0:10,:], DataP1.all_jet], DataP1.all_label, Prong1Model.RNNmodel)

    #Prong1Model.evaluate_model([DataP1.all_track ,D0ataP1.all_jet], DataP1.all_label, Prong1Model.RNNmodel_woTower)

    #Prong1Model.evaluate_model([D0ataP1.all_jet], DataP1.all_label, Prong1Model.basic_model)

    #Prong1Model.plot_accuracy()

    train_real_y, train_pred_y = Prong1Model.get_train_scores(Prong1Model.RNNmodel)
    train_weights = Prong1Model.get_train_score_weights()

    # Prong1Model.plot_feature_heatmap({"track_PT", "track_D0", "track_DZ", "track_deltaEta",
    #                 "track_deltaPhi", "tower_ET", "tower_deltaEta", "tower_deltaPhi", "jet_PT", "jet_PT_LC_scale", "jet_f_cent", "jet_iF_leadtrack", "jet_max_deltaR",
    #             "jet_Ftrack_Iso", "jet_ratio_ToEem_P", "jet_frac_trEM_pt", "jet_mass_track_EM_system"
    #             , "jet_mass_track_system"}, Prong1Model.RNNmodel)

    Prong1Model.evaluate_model(Prong1Model.eval_inputs, Prong1Model.eval_y, Prong1Model.eval_w, Prong1Model.RNNmodel)

    real_y, pred_y, jet_pt = Prong1Model.predict(Prong1Model.RNNmodel)
    #print(pred_y)

    back_real_y, back_pred_y, back_jet_pt = Prong1Model.predict_back(Prong1Model.RNNmodel)

    #weights = Prong1Model.get_score_weights()

    Prong1Plots = Plots( "Prong1Plots", real_y, pred_y, Prong1Model.eval_w, train_real_y, train_pred_y, train_weights, jet_pt)

    back_plots = Plots("P1_backplots", back_real_y, back_pred_y, Prong1Model.eval_back_w, train_real_y, train_pred_y, train_weights, back_jet_pt)

    Prong1Plots.plot_raw_score_vs_jetPT()
    Prong1Plots.histogram_RNN_score()

    back_plots.histogram_RNN_score()

    prong1_rejveff = Prong1Plots.plot_rej_vs_eff()
    plt.draw()
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
