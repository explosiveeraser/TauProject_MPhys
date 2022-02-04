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


def Process_Data(Data):
    scaler = MinMaxScaler()
    scaler.fit(Data)
    data_Scaler = scaler.transform(Data)
    return data_Scaler

def Apply_Logarithm(Data, ApplyTo):
    new_array = []
    for entry in Data:
        temp = []
        for idx in range(0, len(entry)):
            feature = entry[idx]
            if ApplyTo[idx]:
                feature = tf.math.log(feature+1)
            temp.append(feature)
        array = np.asarray(temp).astype('float32')
        new_array.append(array)
    return new_array


#loop to read tree and begin processing input (sorted tracks and towers)
def read_tree(Tree):
    jet_index = 0
    track_index = 0
    tower_index = 0
    jet_array = []
    track_array = []
    tower_array = []
    label_array = []
    for entry in tqdm(Tree):
        jet_array.append([entry.jet_PT, entry.jet_Eta, entry.jet_Phi, entry.jet_deltaEta, entry.jet_deltaPhi,
                              entry.jet_charge, entry.jet_NCharged, entry.jet_NNeutral, entry.jet_deltaR, entry.jet_f_cent,
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
            track = np.asarray([entry.track_P[idx], entry.track_PT[idx], entry.track_L[idx], entry.track_D0[idx], entry.track_DZ[idx], entry.track_deltaEta[idx], entry.track_deltaPhi[idx], entry.track_deltaR[idx]]).astype('float32')
            inside_tracks.append(track)
            track_index += 1
        for jdx in index_sorted_towers[0:9]:
            tower = np.asarray([entry.tower_E[jdx], entry.tower_ET[jdx], entry.tower_Eta[jdx], entry.tower_Phi[jdx], entry.tower_Edges0[jdx], entry.tower_Edges1[jdx],
                     entry.tower_Edges2[jdx], entry.tower_Edges3[jdx], entry.tower_Eem[jdx], entry.tower_Ehad[jdx], entry.tower_T[jdx], entry.tower_deltaEta[jdx],
                     entry.tower_deltaPhi[jdx], entry.tower_deltaR[jdx]]).astype('float32')
            inside_towers.append(tower)
            tower_index += 1
        track_array.append(Apply_Logarithm(Process_Data(inside_tracks), [True, True, True, True, True, False, False, False]))
        #print(Apply_Logarithm(Process_Data(inside_tracks), [True, True, True, True, True, False, False, False]))
        tower_array.append(Apply_Logarithm(Process_Data(inside_towers), [True, True, False, False, False, False, False, False, True, True, True, False, False, False]))
        jet_index += 1
        if jet_index == 6000:
            break
    track_array = pad_sequences(track_array, dtype='float32', maxlen=6, padding='post')
    tower_array = pad_sequences(tower_array, dtype='float32', maxlen=10, padding='post')
    jet_array = np.array(Apply_Logarithm(Process_Data(jet_array), [False, False, False, False, False, False, False, False, False, True, True, False, False]))
    print(Apply_Logarithm(Process_Data(jet_array), [False, False, False, False, False, False, False, False, False, True, True, False, False]))
    track_array = np.array(track_array)
    tower_array = np.array(tower_array)
    return jet_array, track_array, tower_array, label_array

#call function above for sig and back data
sig_jet, sig_tr, sig_to, sig_label = read_tree(Signal_Tree)

#Split train and test
signal_trainJet = sig_jet[0:3999]
signal_testJet = sig_jet[4000:6000]
signal_trainTr = sig_tr[0:3999]
signal_testTr = sig_tr[4000:6000]
signal_trainTo = sig_to[0:3999]
signal_testTo = sig_to[4000:6000]

sig_trainY = sig_label[0:3999]
sig_testY = sig_label[4000:6000]

#print(tf.convert_to_tensor(sig_tr))

back_jet, back_tr, back_to, back_label = read_tree(Background_Tree)

#Split train and test
back_trainJet = back_jet[0:3999]
back_testJet = back_jet[4000:6000]
back_trainTr = back_tr[0:3999]
back_testTr = back_tr[4000:6000]
back_trainTo = back_to[0:3999]
back_testTo = back_to[4000:6000]

back_trainY = back_label[0:3999]
back_testY = back_label[4000:6000]

#print(tf.convert_to_tensor(back_tr))

input_jet = np.append(back_trainJet, signal_trainJet, axis=0)
input_track = np.append(back_trainTr, signal_trainTr, axis=0)
input_tower = np.append(back_trainTo, signal_trainTo, axis=0)

test_jet = np.append(back_testJet, signal_testJet, axis=0)
test_track = np.append(back_testTr, signal_testTr, axis=0)
test_tower = np.append(back_testTo, signal_testTo, axis=0)

Ytrain = np.append(back_trainY, sig_trainY)
Ytest = np.append(back_testY, sig_testY)



print(input_jet.shape)
print(input_tower.shape)
print(Ytrain.shape)

#Data preprocessing: Normalizing the data and inserting timesteps for tracks and towers





#HL Layers
HL_input = Input(shape=(13,))
HLdense1 = Dense(128, activation='relu')(HL_input)
HLdense2 = Dense(128, activation='relu')(HLdense1)
HLdense3 = Dense(16, activation='relu')(HLdense2)

#Track Layers
Track_input1 = Input(shape=(6, 8))
TrackNorm = TimeDistributed(Normalization())(Track_input1)
#Track_input2 = Input(shape=(10,))

trackDense1 = Dense(32, activation='relu', input_shape=(None, 6, 8))
trackDense2 = Dense(32, activation='relu', input_shape=(None, 6, 32))
trackSD1 = TimeDistributed(trackDense1)(TrackNorm)
trackSD2 = TimeDistributed(trackDense2)(trackSD1)

#mergeTrack = Concatenate()([trackSD1, trackSD2])
#flatten = TimeDistributed(Flatten())(trackSD2)

trackLSTM1 = LSTM(32, input_shape=(None, 6, 32), return_sequences=True)(trackSD2)
trackLSTM2 = LSTM(32, input_shape=(None, 6, 32), return_sequences=False)(trackLSTM1)

#Tower Layers
Tower_input1 = Input(shape=(10, 14))
TowerNorm = TimeDistributed(Normalization())(Tower_input1)

#Tower_input2 = Input(shape=(14,))

towerDense1 = Dense(32, activation='relu', input_shape=(None, 10, 14))
towerDense2 = Dense(32, activation='relu', input_shape=(None, 10, 14))
towerSD1 = TimeDistributed(towerDense1)(TowerNorm)
towerSD2 = TimeDistributed(towerDense2)(towerSD1)

#towerFlatten = TimeDistributed(Flatten())(towerSD2)

towerLSTM1 = LSTM(24, return_sequences=True)(towerSD2)
towerLSTM2 = LSTM(24, return_sequences=False)(towerLSTM1)

#Layers Merged
mergedLayer = Concatenate()([trackLSTM2, towerLSTM2, HLdense3])
fullDense1 = Dense(64, activation='relu')(mergedLayer)
fullDense2 = Dense(32, activation='relu')(fullDense1)
Output = Dense(1, activation='sigmoid')(fullDense2)

RNNmodel = Model(inputs=[Track_input1, Tower_input1, HL_input], outputs=Output)
RNNmodel.save('tauRNN_1-prong.h5')
RNNmodel.summary()

plot_model(RNNmodel, to_file="RNNModel.png", show_shapes=True, show_layer_names=True)

RNNmodel.compile(optimizer="SGD", loss="binary_crossentropy", metrics=[tf.keras.metrics.BinaryCrossentropy(), tf.keras.metrics.BinaryAccuracy()])
RNNmodel.fit([input_track, input_tower, input_jet], Ytrain, batch_size=32, epochs=10, verbose=1, validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)])

scores = RNNmodel.evaluate([test_track, test_tower, test_jet], Ytest, batch_size=32, verbose=1)
print("%s: %.2f%%" % (RNNmodel.metrics_names[2], scores[2]*100))

