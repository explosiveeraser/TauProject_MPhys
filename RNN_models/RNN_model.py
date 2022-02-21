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
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm, trange
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc



class Tau_Model():

    pt_bins = np.array([
        10., 25.178, 31.697, 39.905, 50.237, 63.245, 79.621, 100.000,
        130.000, 200.000, 316.978, 502.377, 796.214, 1261.914, 2000.000,
        1000000.000
    ])

    mu_bins = np.array([
        -0.5, 10.5, 19.5, 23.5, 27.5, 31.5, 35.5, 39.5, 49.5, 61.5
    ])

    def __init__(self, Prongs, inputs, sig_pt, bck_pt, jet_pt, y):
        self.prong = Prongs
        self.output = TFile.Open("Prong-{}_RNN_Model.root".format(str(Prongs)), "RECREATE")
        self.track_data = inputs[0]
        self.tower_data = inputs[1]
        self.jet_data = inputs[2]
        self.y_data = y
        self.sig_pt = sig_pt
        self.bck_pt = bck_pt
        self.jet_pt = jet_pt
        start_sig_index = -len(self.sig_pt)
        end_bck_index = len(self.bck_pt)
        self.index_of_sig_bck = np.array(['a']*(len(bck_pt)+len(sig_pt)))
        self.index_of_sig_bck[0:end_bck_index] = 'b'
        self.index_of_sig_bck[start_sig_index:-1] = 's'
        seed = np.random.randint(1, 9999)
        rng = np.random.default_rng(seed)
        np.random.seed(seed)
        shuffled_indices = rng.permutation(len(self.y_data))
        self.track_data = self.track_data[shuffled_indices]
        self.tower_data = self.tower_data[shuffled_indices]
        self.jet_data = self.jet_data[shuffled_indices]
        self.y_data = self.y_data[shuffled_indices]
        self.index_of_sig_bck = self.index_of_sig_bck[shuffled_indices]
        self.jet_pt = np.append(self.bck_pt, self.sig_pt)[shuffled_indices]

        self.inputs = [self.track_data[int(len(self.jet_data)/2+1):-1], self.tower_data[int(len(self.jet_data)/2+1):-1],
                       self.jet_data[int(len(self.jet_data)/2+1):-1]]
        self.y = self.y_data[int(len(self.jet_data)/2+1):-1]
        self.train_sigbck_index = self.index_of_sig_bck[int(len(self.jet_data)/2+1):-1]
        self.train_jet_pt = self.jet_pt[int(len(self.jet_data)/2+1):-1]

        self.eval_inputs = [self.track_data[0:int(len(self.jet_data)/2+1)], self.tower_data[0:int(len(self.jet_data)/2+1)],
                            self.jet_data[0:int(len(self.jet_data)/2+1)]]
        self.eval_y = self.y_data[0:int(len(self.jet_data)/2+1)]
        self.eval_sigbck_index = self.index_of_sig_bck[0:int(len(self.jet_data)/2+1)]
        self.eval_jet_pt = self.jet_pt[0:int(len(self.jet_data)/2+1)]
        self.RNN_Model()
        #self.basic_Model()
        #self.RNN_ModelwoTowers()

    def basic_Model(self):
        HL_input = Input(shape=(10,))
        HLdense1 = Dense(128, activation='relu', kernel_initializer='RandomUniform',
                         bias_initializer='zeros')(HL_input)
        HLdense3 = Dense(16, activation='relu', kernel_initializer='RandomUniform',
                         bias_initializer='zeros')(HLdense1)
        Output = Dense(1, activation='sigmoid', kernel_initializer='RandomUniform',
                       bias_initializer='zeros')(HLdense3)
        self.basic_model = Model(inputs=[HL_input], outputs=Output)
        self.basic_model.summary()
        plot_model(self.basic_model, to_file="basic_Model.png", show_shapes=True, show_layer_names=True)
        self.basic_model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9, nesterov=True), loss="binary_crossentropy",
                         metrics=['accuracy','binary_crossentropy', 'TruePositives', 'FalsePositives'])

    def RNN_ModelwoTowers(self):
        backwards = False
        unroll = False

        # HL Layers
        HL_input = Input(shape=(10,))
        HLdense1 = Dense(128, activation='relu', kernel_initializer = 'RandomUniform',
                bias_initializer = 'zeros')(HL_input)
        HLdense2 = Dense(128, activation='relu', kernel_initializer = 'RandomUniform',
                bias_initializer = 'zeros')(HLdense1)
        HLdense3 = Dense(16, activation='relu', kernel_initializer = 'RandomUniform',
                bias_initializer = 'zeros')(HLdense2)
        # Track Layers
        Track_input1 = Input(shape=(None, 5))
        maskedTrack = Masking()(Track_input1)
        # Track_input2 = Input(shape=(10,))
        trackDense1 = Dense(32, activation='relu', input_shape=(None, None, 5), kernel_initializer = 'RandomUniform',
                bias_initializer = 'zeros')
        trackDense2 = Dense(32, activation='relu', input_shape=(None, None, 32), kernel_initializer = 'RandomUniform',
                bias_initializer = 'zeros')
        trackSD1 = TimeDistributed(trackDense1)(maskedTrack)
        trackSD2 = TimeDistributed(trackDense2)(trackSD1)
        # mergeTrack = Concatenate()([trackSD1, trackSD2])
        # flatten = TimeDistributed(Flatten())(trackSD2)
        trackLSTM1 = LSTM(32, activation="tanh", go_backwards=backwards, unroll=unroll, input_shape=(None, 6, 32), return_sequences=True, kernel_initializer = 'RandomUniform',
                bias_initializer = 'zeros')(trackSD2)
        trackLSTM2 = LSTM(32, activation="tanh", go_backwards=backwards, unroll=unroll, input_shape=(None, 6, 32),  kernel_initializer = 'RandomUniform',
                bias_initializer = 'zeros')(trackLSTM1)
        # Layers Merged
        mergedLayer = Concatenate()([trackLSTM2, HLdense3])
        fullDense1 = Dense(64, activation='relu', kernel_initializer = 'RandomUniform',
                bias_initializer = 'zeros')(mergedLayer)
        fullDense2 = Dense(32, activation='relu', kernel_initializer = 'RandomUniform',
                bias_initializer = 'zeros')(fullDense1)
        Output = Dense(1, activation='sigmoid', kernel_initializer = 'RandomUniform',
                bias_initializer = 'zeros')(fullDense2)
        self.RNNmodel_woTower = Model(inputs=[Track_input1, HL_input], outputs=Output)
        self.RNNmodel_woTower.summary()
        plot_model(self.RNNmodel_woTower, to_file="RNNModel.png", show_shapes=True, show_layer_names=True)
        #SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
        self.RNNmodel_woTower.compile(optimizer='adam', loss="binary_crossentropy",
                         metrics=['accuracy','binary_crossentropy', 'TruePositives', 'FalsePositives'])


    def RNN_Model(self):
        backwards = False
        unroll = False

        # HL Layers
        HL_input = Input(shape=(10,))
        HLdense1 = Dense(128, activation='relu', kernel_initializer = 'RandomUniform',
                bias_initializer = 'zeros')(HL_input)
        HLdense2 = Dense(128, activation='relu', kernel_initializer = 'RandomUniform',
                bias_initializer = 'zeros')(HLdense1)
        HLdense3 = Dense(16, activation='relu', kernel_initializer = 'RandomUniform',
                bias_initializer = 'zeros')(HLdense2)
        # Track Layers
        Track_input1 = Input(shape=(None, 5))
        maskedTrack = Masking()(Track_input1)
        # Track_input2 = Input(shape=(10,))
        trackDense1 = Dense(32, activation='relu', input_shape=(None, None, 5), kernel_initializer = 'RandomUniform',
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
        Tower_input1 = Input(shape=(None, 3))
        maskedTower = Masking()(Tower_input1)
        # Tower_input2 = Input(shape=(14,))
        towerDense1 = Dense(32, activation='relu', input_shape=(None, None, 3), kernel_initializer = 'RandomUniform',
                bias_initializer = 'zeros')
        towerDense2 = Dense(32, activation='relu', input_shape=(None, None, 32), kernel_initializer = 'RandomUniform',
                bias_initializer = 'zeros')
        towerSD1 = TimeDistributed(towerDense1)(maskedTower)
        towerSD2 = TimeDistributed(towerDense2)(towerSD1)
        # towerFlatten = TimeDistributed(Flatten())(towerSD2)
        towerLSTM1 = LSTM(24, activation="tanh", go_backwards=backwards, unroll=unroll, input_shape=(None, 10, 32), return_sequences=True, kernel_initializer = 'RandomUniform',
                bias_initializer = 'zeros')(towerSD2)
        towerLSTM2 = LSTM(24, activation="tanh", go_backwards=backwards, unroll=unroll, input_shape=(None, 10, 24), return_sequences=False, kernel_initializer = 'RandomUniform',
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
        #SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
        self.RNNmodel.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy",
                         metrics=['accuracy','binary_crossentropy', 'TruePositives', 'FalsePositives', "FalseNegatives", "TrueNegatives"])

    def Model_Fit(self, batch_size, epochs, validation_split, model="RNNmodel", inputs="RNNmodel"):
        if type(model) == str:
            self.history = self.RNNmodel.fit(self.inputs, self.y, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=validation_split)
            self.RNNmodel.save("RNN_Model_Prong-{}.h5".format(str(self.prong)))
            print(self.history.history.keys())
        else:
            self.history = model.fit(inputs, self.y, batch_size=batch_size, epochs=epochs, verbose=1,
                                             validation_split=validation_split)
            model.save("RNN_Model_Prong-{}.h5".format(str(self.prong)))
            print(self.history.history.keys())

    def evaluate_model(self, inputs, outputs, model, batch_size=256):
        results = model.evaluate(inputs, outputs, batch_size=batch_size)
        print("test loss, test acc:", results)
        print("TrueTau/FakeTau", results[3]/(results[3]+results[4]))
        print("Taus Not IDed : ", results[5])
        print("Total Taus: ",results[3]+results[4]+results[5])

    def pt_reweight(self, sig_pt, bkg_pt):
        # Binning
        bin_edges = np.percentile(bkg_pt, np.linspace(0.0, 100.0, 50))

        bin_edges[0] = 20.0  # 20 GeV lower limit
        bin_edges[-1] = 1000.0  # 10000 GeV upper limit
        print(bin_edges)
        # Reweighting coefficient
        sig_hist, _ = np.histogram(sig_pt, bins=bin_edges, density=True)
        bkg_hist, _ = np.histogram(bkg_pt, bins=bin_edges, density=True)

        coeff = sig_hist / bkg_hist
        print(coeff)
        # Apply reweighting
        sig_weight = np.ones_like(sig_pt)
        bkg_weight = coeff[np.digitize(bkg_pt, bin_edges) - 1].astype(np.float32)

        return sig_weight, bkg_weight

    def get_train_scores(self, model):
        train_y = model.predict(self.inputs)
        return self.y, train_y

    def get_train_score_weights(self):
        sig = self.train_jet_pt[self.train_sigbck_index == 's']
        bck = self.train_jet_pt[self.train_sigbck_index == 'b']
        sig_w, bck_w = self.pt_reweight(sig, bck)
        new_arr = np.zeros(len(self.y))
        new_arr[self.train_sigbck_index == 's'] = sig_w
        new_arr[self.train_sigbck_index == 'b'] = bck_w
        return new_arr

    def predict(self, model):
        self.predictions = model.predict(self.eval_inputs)
        jet_pt = self.eval_inputs[2][:, 0]
        return self.eval_y, self.predictions, jet_pt

    def get_score_weights(self):
        sig = self.eval_jet_pt[self.eval_sigbck_index == 's']
        bck = self.eval_jet_pt[self.eval_sigbck_index == 'b']
        sig_w, bck_w = self.pt_reweight(sig, bck)
        new_arr = np.zeros(len(self.predictions))
        new_arr[self.eval_sigbck_index == 's'] = sig_w
        new_arr[self.eval_sigbck_index == 'b'] = bck_w
        return new_arr

    def plot_accuracy(self):
        #print("TP: {} |TN {}\n----------\n FP: {} | FN: {}".format(self.history.history['true_positives'][9], self.history.history['true_negatives'][9], self.history.history['false_positives'][9]))
        #figure, axis = plt.subplots(int(math.sqrt(len(self.history.history.keys())))+1, int(math.sqrt(len(self.history.history.keys())))+1)
        figure, axis = plt.subplots( 2, 2)
        ax1 = 0
        ax2 = 0
        print(int(math.sqrt(len(self.history.history.keys()))))
        for key in ["val_binary_crossentropy", 'val_accuracy', 'val_true_positives', 'val_false_positives']:
            if key not in {'true_positives', 'true_negatives', 'false_positives', 'false_negatives'}:
                axis[ax1, ax2].plot(self.history.history[key])
                axis[ax1, ax2].plot(self.history.history[key])
                axis[ax1, ax2].set_title('model {}'.format(key))
                axis[ax1, ax2].set_ylabel(key)
                axis[ax1, ax2].set_xlabel('epoch')
                axis[ax1, ax2].legend(['train', 'test'], loc='upper left')
                if ax1 < 1:
                    ax1 += 1
                else:
                    ax2 += 1
                    ax1 = 0
        plt.show()

    def plot_feature_heatmap(self, features, model):
        plt.figure(figsize=(40,5))
        plt.imshow(model.coefs_[0], interpolation='none', cmap='viridis')
        plt.yticks(len(features), features)
        plt.xlabel()