import math

import keras_preprocessing.sequence
import matplotlib.pyplot as plt
import numpy as np
from ROOT import TMVA, TFile, TTree, TCut
import ROOT
from subprocess import call
from os.path import isfile
import pandas as pd
import random


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
from keras.layers.core import Masking
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm, trange


class Tau_Model():

    def __init__(self, Prongs, inputs, y):
        self.prong = Prongs
        self.output = TFile.Open("Prong-{}_RNN_Model.root".format(str(Prongs)), "RECREATE")
        self.inputs = inputs
        self.y = y
        self.RNN_Model()

    def RNN_Model(self):
        backwards = False
        unroll = False

        # HL Layers
        HL_input = Input(shape=(13,))
        HLdense1 = Dense(128, activation='relu', kernel_initializer = 'RandomUniform',
                bias_initializer = 'zeros')(HL_input)
        HLdense2 = Dense(128, activation='relu', kernel_initializer = 'RandomUniform',
                bias_initializer = 'zeros')(HLdense1)
        HLdense3 = Dense(16, activation='relu', kernel_initializer = 'RandomUniform',
                bias_initializer = 'zeros')(HLdense2)
        # Track Layers
        Track_input1 = Input(shape=(None, 8))
        maskedTrack = Masking()(Track_input1)
        # Track_input2 = Input(shape=(10,))
        trackDense1 = Dense(32, activation='relu', input_shape=(None, None, 8), kernel_initializer = 'RandomUniform',
                bias_initializer = 'zeros')
        trackDense2 = Dense(32, activation='relu', input_shape=(None, None, 32), kernel_initializer = 'RandomUniform',
                bias_initializer = 'zeros')
        trackSD1 = TimeDistributed(trackDense1)(maskedTrack)
        trackSD2 = TimeDistributed(trackDense2)(trackSD1)
        # mergeTrack = Concatenate()([trackSD1, trackSD2])
        # flatten = TimeDistributed(Flatten())(trackSD2)
        trackLSTM1 = LSTM(32, activation="tanh", go_backwards=backwards, unroll=unroll, input_shape=(None, 6, 32), return_sequences=True, kernel_initializer = 'RandomUniform',
                bias_initializer = 'zeros')(trackSD2)
        trackLSTM2 = LSTM(32, activation="tanh", go_backwards=backwards, unroll=unroll, input_shape=(None, 6, 32), return_sequences=False, kernel_initializer = 'RandomUniform',
                bias_initializer = 'zeros')(trackLSTM1)
        # Tower Layers
        Tower_input1 = Input(shape=(None, 14))
        maskedTower = Masking()(Tower_input1)
        # Tower_input2 = Input(shape=(14,))
        towerDense1 = Dense(32, activation='relu', input_shape=(None, None, 14), kernel_initializer = 'RandomUniform',
                bias_initializer = 'zeros')
        towerDense2 = Dense(32, activation='relu', input_shape=(None, None, 14), kernel_initializer = 'RandomUniform',
                bias_initializer = 'zeros')
        towerSD1 = TimeDistributed(towerDense1)(maskedTower)
        towerSD2 = TimeDistributed(towerDense2)(towerSD1)
        # towerFlatten = TimeDistributed(Flatten())(towerSD2)
        towerLSTM1 = LSTM(24, activation="tanh", go_backwards=backwards, unroll=unroll, input_shape=(None, 10, 14), return_sequences=True, kernel_initializer = 'RandomUniform',
                bias_initializer = 'zeros')(towerSD2)
        towerLSTM2 = LSTM(24, activation="tanh", go_backwards=backwards, unroll=unroll, input_shape=(None, 10, 14), return_sequences=False, kernel_initializer = 'RandomUniform',
                bias_initializer = 'zeros')(towerLSTM1)
        # Layers Merged
        mergedLayer = Concatenate()([trackLSTM2, towerLSTM2, HLdense3])
        fullDense1 = Dense(64, activation='relu', kernel_initializer = 'RandomUniform',
                bias_initializer = 'zeros')(mergedLayer)
        fullDense2 = Dense(32, activation='relu', kernel_initializer = 'RandomUniform',
                bias_initializer = 'zeros')(fullDense1)
        Output = Dense(1, activation='sigmoid', kernel_initializer = 'RandomUniform',
                bias_initializer = 'zeros')(fullDense2)
        self.RNNmodel = Model(inputs=[Track_input1, Tower_input1, HL_input], outputs=Output)
        self.RNNmodel.summary()
        plot_model(self.RNNmodel, to_file="RNNModel.png", show_shapes=True, show_layer_names=True)
        self.RNNmodel.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9, nesterov=True), loss="binary_crossentropy",
                         metrics=['accuracy','binary_crossentropy', 'TruePositives', 'FalsePositives'])

    def Model_Fit(self, batch_size, epochs, validation_split):
        self.history = self.RNNmodel.fit(self.inputs, self.y, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=validation_split)
        self.RNNmodel.save("RNN_Model_Prong-{}.h5".format(str(self.prong)))
        print(self.history.history.keys())

    def plot_accuracy(self):
        print("TP: {} |TN {}\n----------\n FP: {} | FN: {}".format(self.history.history['true_positives'][9], self.history.history['true_negatives'][9], self.history.history['false_positives'][9], self.history.history['false_negatives'][9]))
        figure, axis = plt.subplots(int(math.sqrt(len(self.history.history.keys())))+1, int(math.sqrt(len(self.history.history.keys())))+1)
        ax1 = 0
        ax2 = 0
        print(int(math.sqrt(len(self.history.history.keys()))))
        for key in self.history.history.keys():
            if key not in {'true_positives', 'true_negatives', 'false_positives', 'false_negatives'}:
                axis[ax1, ax2].plot(self.history.history[key])
                axis[ax1, ax2].plot(self.history.history[key])
                axis[ax1, ax2].set_title('model {}'.format(key))
                axis[ax1, ax2].set_ylabel(key)
                axis[ax1, ax2].set_xlabel('epoch')
                axis[ax1, ax2].legend(['train', 'test'], loc='upper left')
                if ax1 < int(math.sqrt(len(self.history.history.keys()))):
                    ax1 +=1
                else:
                    ax2 += 1
                    ax1 = 0
        plt.show()