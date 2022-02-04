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
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm, trange


class Tau_Model():

    def __init__(self, Prongs, BacktreeFile, BackTreeName,  SignaltreeFile, SignalTreeName, BackendPartOfTree="", SignalendPartOfTree=""):
        self.prong = Prongs
        self.output = TFile.Open("Prong-{}_RNN_Model.root".format(str(Prongs)), "RECREATE")
        back_tree_file = TFile.Open("../NewTTrees/{}.root".format(BacktreeFile))
        sig_tree_file = TFile.Open("../NewTTrees/{}.root".format(SignaltreeFile))

        self.BackgroundTree = back_tree_file.Get("{}{}".format(BackTreeName, BackendPartOfTree))
        self.SignalTree = sig_tree_file.Get("{}{}".format(SignalTreeName, SignalendPartOfTree))

        SignalNumEntries = self.SignalTree.GetEntries()
        BackgroundNumEntries = self.BackgroundTree.GetEntries()
        # call function above for sig and back data
        sig_jet, sig_tr, sig_to, sig_label = self.read_tree(self.SignalTree)

        back_jet, back_tr, back_to, back_label = self.read_tree(self.BackgroundTree)
        # print(tf.convert_to_tensor(back_tr))
        rng = np.random.default_rng(123)
        np.random.seed(123)
        temp_jet = np.append(back_jet, sig_jet, axis=0)
        temp_track = np.append(back_tr, sig_tr, axis=0)
        temp_tower = np.append(back_to, sig_to, axis=0)
        temp_labels = np.append(back_label, sig_label)

        self.input_jet = rng.permutation(temp_jet, axis=0)
        self.input_track = rng.permutation(temp_track, axis=0)
        self.input_tower = rng.permutation(temp_tower, axis=0)
        self.Ytrain = rng.permutation(temp_labels, axis=0)
        self.RNN_Model()

    def RNN_Model(self):
        # HL Layers
        HL_input = Input(shape=(13,))
        HLdense1 = Dense(128, activation='relu')(HL_input)
        HLdense2 = Dense(128, activation='relu')(HLdense1)
        HLdense3 = Dense(16, activation='relu')(HLdense2)
        # Track Layers
        Track_input1 = Input(shape=(6, 8))
        TrackNorm = TimeDistributed(Normalization())(Track_input1)
        # Track_input2 = Input(shape=(10,))
        trackDense1 = Dense(32, activation='relu', input_shape=(None, 6, 8))
        trackDense2 = Dense(32, activation='relu', input_shape=(None, 6, 32))
        trackSD1 = TimeDistributed(trackDense1)(TrackNorm)
        trackSD2 = TimeDistributed(trackDense2)(trackSD1)
        # mergeTrack = Concatenate()([trackSD1, trackSD2])
        # flatten = TimeDistributed(Flatten())(trackSD2)
        trackLSTM1 = LSTM(32, input_shape=(None, 6, 32), return_sequences=True)(trackSD2)
        trackLSTM2 = LSTM(32, input_shape=(None, 6, 32), return_sequences=False)(trackLSTM1)
        # Tower Layers
        Tower_input1 = Input(shape=(10, 14))
        TowerNorm = TimeDistributed(Normalization())(Tower_input1)
        # Tower_input2 = Input(shape=(14,))
        towerDense1 = Dense(32, activation='relu', input_shape=(None, 10, 14))
        towerDense2 = Dense(32, activation='relu', input_shape=(None, 10, 14))
        towerSD1 = TimeDistributed(towerDense1)(TowerNorm)
        towerSD2 = TimeDistributed(towerDense2)(towerSD1)
        # towerFlatten = TimeDistributed(Flatten())(towerSD2)
        towerLSTM1 = LSTM(24, return_sequences=True)(towerSD2)
        towerLSTM2 = LSTM(24, return_sequences=False)(towerLSTM1)
        # Layers Merged
        mergedLayer = Concatenate()([trackLSTM2, towerLSTM2, HLdense3])
        fullDense1 = Dense(64, activation='relu')(mergedLayer)
        fullDense2 = Dense(32, activation='relu')(fullDense1)
        Output = Dense(1, activation='sigmoid')(fullDense2)
        self.RNNmodel = Model(inputs=[Track_input1, Tower_input1, HL_input], outputs=Output)
        self.RNNmodel.summary()
        plot_model(self.RNNmodel, to_file="RNNModel.png", show_shapes=True, show_layer_names=True)
        self.RNNmodel.compile(optimizer="SGD", loss="binary_crossentropy",
                         metrics=['accuracy', 'BinaryAccuracy', 'BinaryCrossentropy', 'AUC', 'Recall', 'Precision', 'TruePositives', 'TrueNegatives', 'FalsePositives', 'FalseNegatives'])

    def Process_Data(self, Data):
        scaler = MinMaxScaler()
        scaler.fit(Data)
        data_Scaler = scaler.transform(Data)
        return data_Scaler

    def Apply_Logarithm(self, Data, ApplyTo):
        new_array = []
        for entry in Data:
            temp = []
            for idx in range(0, len(entry)):
                feature = entry[idx]
                if ApplyTo[idx]:
                    feature = tf.math.log(feature + 1)
                temp.append(feature)
            array = np.asarray(temp).astype('float32')
            new_array.append(array)
        return new_array

    # loop to read tree and begin processing input (sorted tracks and towers)
    def read_tree(self, Tree):
        jet_index = 0
        track_index = 0
        tower_index = 0
        jet_array = []
        track_array = []
        tower_array = []
        label_array = []
        for entry in tqdm(Tree):
            jet_array.append([entry.jet_PT, entry.jet_Eta, entry.jet_Phi, entry.jet_deltaEta, entry.jet_deltaPhi,
                              entry.jet_charge, entry.jet_NCharged, entry.jet_NNeutral, entry.jet_deltaR,
                              entry.jet_f_cent,
                              entry.jet_iF_leadtrack, entry.jet_max_deltaR, entry.jet_Ftrack_Iso])
            if entry.jet_TruthTau == 1:
                label_array.append(1)
            elif entry.jet_TruthTau == 0:
                label_array.append(0)
            track_index = 0
            tower_index = 0
            nTrack = int(entry.nTrack)
            nTower = int(entry.nTower)
            # TRACK_ARRAY: ['[index]',[P], [PT], [L], [D0], [DZ], [e], [e], [deltaEta], [deltaPhi], [deltaR]]
            inside_tracks = []
            inside_towers = []
            # TOWER_ARRAY: ['[index]',[E], [ET], [Eta], [Phi], [Edges0], [Edges1], [Edges2], [Edges3], [Eem], [Ehad], [T],
            # [deltaEta], [deltaPhi], [deltaR]]
            tr_pt = []
            to_et = []
            for t in range(0, nTrack):
                tr_pt.append(entry.track_PT[t])
            index_sorted_tracks = sorted(range(len(tr_pt)), key=lambda k: tr_pt[k], reverse=True)
            for t in range(0, nTower):
                to_et.append(entry.tower_ET[t])
            index_sorted_towers = sorted(range(len(to_et)), key=lambda k: to_et[k], reverse=True)
            for idx in index_sorted_tracks[0:5]:
                track = np.asarray([entry.track_P[idx], entry.track_PT[idx], entry.track_L[idx], entry.track_D0[idx],
                                    entry.track_DZ[idx], entry.track_deltaEta[idx], entry.track_deltaPhi[idx],
                                    entry.track_deltaR[idx]]).astype('float32')
                inside_tracks.append(track)
                track_index += 1
            for jdx in index_sorted_towers[0:9]:
                tower = np.asarray([entry.tower_E[jdx], entry.tower_ET[jdx], entry.tower_Eta[jdx], entry.tower_Phi[jdx],
                                    entry.tower_Edges0[jdx], entry.tower_Edges1[jdx],
                                    entry.tower_Edges2[jdx], entry.tower_Edges3[jdx], entry.tower_Eem[jdx],
                                    entry.tower_Ehad[jdx], entry.tower_T[jdx], entry.tower_deltaEta[jdx],
                                    entry.tower_deltaPhi[jdx], entry.tower_deltaR[jdx]]).astype('float32')
                inside_towers.append(tower)
                tower_index += 1
            track_array.append(
                self.Apply_Logarithm(self.Process_Data(inside_tracks), [True, True, True, True, True, False, False, False]))
            # print(Apply_Logarithm(Process_Data(inside_tracks), [True, True, True, True, True, False, False, False]))
            tower_array.append(self.Apply_Logarithm(self.Process_Data(inside_towers),
                                               [True, True, False, False, False, False, False, False, True, True, True,
                                                False, False, False]))
            jet_index += 1
            #if jet_index == 3000:
             #   break
        track_array = pad_sequences(track_array, dtype='float32', maxlen=6, padding='post')
        tower_array = pad_sequences(tower_array, dtype='float32', maxlen=10, padding='post')
        jet_array = np.array(self.Apply_Logarithm(self.Process_Data(jet_array),
                                             [False, False, False, False, False, False, False, False, False, True, True,
                                              False, False]))
        track_array = np.array(track_array)
        tower_array = np.array(tower_array)
        label_array = np.array(label_array)
        return jet_array, track_array, tower_array, label_array

    def Model_Fit(self, batch_size, epochs, validation_split):
        self.history = self.RNNmodel.fit([self.input_track, self.input_tower, self.input_jet], self.Ytrain, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=validation_split)
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