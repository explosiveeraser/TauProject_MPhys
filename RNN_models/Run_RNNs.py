# import keras_preprocessing.sequence
# import matplotlib.pyplot as plt
# import numpy as np
# from ROOT import TMVA, TFile, TTree, TCut
# import ROOT
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
from RNN_model import Tau_Model
from RNN_Data import RNN_Data

DataP1 = RNN_Data(1, True, "prong1_data", BacktreeFile="background_tree_1-Prong", BackTreeName="background_tree", SignaltreeFile="signal_tree_1-Prong", SignalTreeName="signal_tree", BackendPartOfTree=";9", SignalendPartOfTree=";2")

Prong1Model = Tau_Model(1, [DataP1.input_track[:,0:6,:], DataP1.input_tower[:,0:10,:], DataP1.input_jet], DataP1.Ytrain)

Prong1Model.Model_Fit(256, 100, 0.3)

Prong1Model.plot_accuracy()
input("Press enter to see Prong 3")

DataP3 = RNN_Data(3, False, "prong3_data", BacktreeFile="background_tree_3-Prong", BackTreeName="background_tree", SignaltreeFile="signal_tree_3-Prong", SignalTreeName="signal_tree", BackendPartOfTree=";9", SignalendPartOfTree=";2")

Prong3Model = Tau_Model(3, [DataP3.input_track, DataP3.input_tower, DataP3.input_jet], DataP3.Ytrain)

Prong3Model.Model_Fit(32, 10, 0.3)

Prong3Model.plot_accuracy()
input("Press enter to exit")



