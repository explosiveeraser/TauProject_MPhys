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


Prong1 = Tau_Model(1, "background_tree_1-Prong", "background_tree", "signal_tree_1-Prong", "signal_tree", BackendPartOfTree=";9", SignalendPartOfTree=";2")

Prong1.Model_Fit(32, 10, 0.3)

Prong1.plot_accuracy()
input("Press enter to see Prong 3")


Prong3 = Tau_Model(3, "background_tree_3-Prong", "background_tree", "signal_tree_3-Prong", "signal_tree", BackendPartOfTree=";9", SignalendPartOfTree=";2")

Prong3.Model_Fit(32, 10, 0.3)

Prong3.plot_accuracy()
input("Press enter to exit")



