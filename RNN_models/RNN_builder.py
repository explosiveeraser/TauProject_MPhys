import keras_preprocessing.sequence
import numpy as np
from ROOT import TMVA, TFile, TTree, TCut
import ROOT
from subprocess import call
from os.path import isfile
import pandas as pd

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

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.timeseries import timeseries_dataset_from_array
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
from tqdm import tqdm, trange

output = TFile.Open("RNN_1-Prong.root", "RECREATE")


Signal_File = TFile.Open("../NewTTrees/signal_tree_1-Prong.root")
Background_File = TFile.Open("../NewTTrees/background_tree_1-Prong.root")

Signal_Tree = Signal_File.Get('signal_tree;2')
Background_Tree = Background_File.Get('background_tree;9')

back_numEntries = Background_Tree.GetEntries()
back_jet_array = np.array([0, 0., 0., 0., 0., 0., 0., 0, 0, 0, 0., 0., 0., 0.]*back_numEntries)
back_track_array = np.array([[], [], [], [], [], [], [], [], []]*back_numEntries)
back_tower_array = np.array([[], [], [], [], [], [], [], [], [], [], [], [], [], [], []]*back_numEntries)

sig_numEntries = Signal_Tree.GetEntries()
sig_jet_array = []
sig_track_array = []
sig_tower_array = []

# NUMPY INPUTS:
# () - Single Value
# [] - Array of Values
# JET_ARRAY: ['(index)',(PT),(Eta),(Phi),(deltaEta),(deltaPhi),(charge),(NCharged),(NNeutral),(deltaR),(f_cent),
# (iF_leadtrack),(max_deltaR),(Ftrack_Iso)]
# TRACK_ARRAY: ['[index]',[P], [PT], [L], [D0], [DZ], [e], [e], [deltaEta], [deltaPhi], [deltaR]]
# TOWER_ARRAY: ['[index]',[E], [ET], [Eta], [Phi], [Edges0], [Edges1], [Edges2], [Edges3], [Eem], [Ehad], [T],
# [deltaEta], [deltaPhi], [deltaR]]

jet_index = 0
track_index = 0
tower_index = 0

#loop to read tree and begin processing input (sorted tracks and towers)
def read_tree(Tree):
    jet_index = 0
    track_index = 0
    tower_index = 0
    jet_array = []
    track_array = []
    tower_array = []
    for entry in tqdm(Tree):
        jet_array.append([entry.jet_PT, entry.jet_Eta, entry.jet_Phi, entry.jet_deltaEta, entry.jet_deltaPhi,
                              entry.jet_charge, entry.jet_NCharged, entry.jet_NNeutral, entry.jet_deltaR, entry.jet_f_cent,
                              entry.jet_iF_leadtrack, entry.jet_max_deltaR, entry.jet_Ftrack_Iso])
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
            track = np.asarray([entry.track_P[idx], entry.track_PT[idx], entry.track_L[idx], entry.track_D0[idx], entry.track_DZ[idx], entry.track_deltaEta[idx], entry.track_deltaPhi[idx], entry.track_deltaR[idx]]).astype('float32')
            inside_tracks.append(track)
            track_index += 1
        for jdx in index_sorted_towers[0:9]:
            tower = np.asarray([entry.tower_E[jdx], entry.tower_ET[jdx], entry.tower_Eta[jdx], entry.tower_Phi[jdx], entry.tower_Edges0[jdx], entry.tower_Edges1[jdx],
                     entry.tower_Edges2[jdx], entry.tower_Edges3[jdx], entry.tower_Eem[jdx], entry.tower_Ehad[jdx], entry.tower_T[jdx], entry.tower_deltaEta[jdx],
                     entry.tower_deltaPhi[jdx], entry.tower_deltaR[jdx]]).astype('float32')
            inside_towers.append(tower)
            tower_index += 1
        track_array.append(inside_tracks)
        tower_array.append(inside_towers)
        jet_index += 1
        if jet_index == 4000:
            break
    track_array = pad_sequences(track_array, dtype='float32', maxlen=6, padding='post')
    tower_array = pad_sequences(tower_array, dtype='float32', maxlen=10, padding='post')
    jet_array = np.array(jet_array)
    track_array = np.array(track_array)
    tower_array = np.array(tower_array)
    return jet_array, track_array, tower_array

#call function above for sig and back data
sig_jet, sig_tr, sig_to = read_tree(Signal_Tree)

print(sig_tr[3].shape)
print(sig_tr[3])

print(tf.convert_to_tensor(sig_tr))


back_jet, back_tr, back_to = read_tree(Background_Tree)

print(back_tr[3].shape)
print(back_tr[3])

print(tf.convert_to_tensor(back_tr))

#Data preprocessing: Normalizing the data and inserting timesteps for tracks and towers





#HL Layers
HL_input = Input(shape=(13,))
HLdense1 = Dense(128, activation='relu')(HL_input)
HLdense2 = Dense(128, activation='relu')(HLdense1)
HLdense3 = Dense(16, activation='relu')(HLdense2)

#Track Layers
Track_input1 = Input(shape=(6, 8))
#Track_input2 = Input(shape=(10,))

trackDense1 = Dense(32, activation='relu', input_shape=(None, 6, 8))
trackDense2 = Dense(32, activation='relu', input_shape=(None, 6, 8))
trackSD1 = TimeDistributed(trackDense1)(Track_input1)
trackSD2 = TimeDistributed(trackDense2)(trackSD1)

#mergeTrack = Concatenate()([trackSD1, trackSD2])
#flatten = TimeDistributed(Flatten())(trackSD2)

trackLSTM1 = LSTM(32, input_shape=(None, 6, 8), return_sequences=True)(trackSD2)
trackLSTM2 = LSTM(32, input_shape=(None, 6, 8), return_sequences=False)(trackLSTM1)

#Tower Layers
Tower_input1 = Input(shape=(10, 14))
#Tower_input2 = Input(shape=(14,))

towerDense1 = Dense(32, activation='relu', input_shape=(None, 10, 14))
towerDense2 = Dense(32, activation='relu', input_shape=(None, 10, 14))
towerSD1 = TimeDistributed(towerDense1)(Tower_input1)
towerSD2 = TimeDistributed(towerDense2)(towerSD1)

#towerFlatten = TimeDistributed(Flatten())(towerSD2)

towerLSTM1 = LSTM(24, return_sequences=True)(towerSD2)
towerLSTM2 = LSTM(24, return_sequences=False)(towerLSTM1)

#Layers Merged
mergedLayer = Concatenate()([trackLSTM2, towerLSTM2, HLdense3])
fullDense1 = Dense(64, activation='relu')(mergedLayer)
fullDense2 = Dense(32, activation='relu')(fullDense1)
last = Flatten()(fullDense2)
Output = Dense(1, activation='sigmoid')(last)

RNNmodel = Model(inputs=[Track_input1, Tower_input1, HL_input], outputs=Output)
RNNmodel.save('tauRNN_1-prong.h5')
RNNmodel.summary()

RNNmodel.compile(optimizer="SGD", loss="binary_crossentropy", metrics=[tf.keras.metrics.BinaryCrossentropy()])
RNNmodel.evaluate([back_tr, back_to, back_jet], [sig_tr, sig_to, sig_jet], batch_size=32, epochs=10, verbose=1, validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)])
