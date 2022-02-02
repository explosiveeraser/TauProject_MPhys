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

# back_jet_df = pd.DataFrame({
#     "jet_PT" : np.array([0]*back_numEntries),
#     "jet_Eta" : np.array([0.]*back_numEntries),
#     "jet_Phi" : np.array([0.]*back_numEntries),
#     "jet_deltaEta" : np.array([0.]*back_numEntries),
#     "jet_deltaPhi" : np.array([0.]*back_numEntries),
#     "jet_charge" : np.array([0.]*back_numEntries),
#     "jet_NCharged" : np.array([0.]*back_numEntries),
#     "jet_NNeutral" : np.array([0.]*back_numEntries),
#     "jet_deltaR" : np.array([0.]*back_numEntries),
#     "jet_f_cent" : np.array([0.]*back_numEntries),
#     "jet_iF_leadtrack" : np.array([0.]*back_numEntries),
#     "jet_max_deltaR" : np.array([0.]*back_numEntries),
#     "jet_Ftrack_Iso" : np.array([0.]*back_numEntries)
# })
#
# back_track_df = pd.DataFrame({
#     "jet_Index" : np.array([0]*back_numEntries),
#     "nTrack" : np.array(([0]*back_numEntries)),
#
# })

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
for entry in tqdm(Signal_Tree):
    sig_jet_array.append([np.log(entry.jet_PT), entry.jet_Eta, entry.jet_Phi, entry.jet_deltaEta, entry.jet_deltaPhi,
                          entry.jet_charge, entry.jet_NCharged, entry.jet_NNeutral, entry.jet_deltaR, np.log(entry.jet_f_cent),
                          np.log(entry.jet_iF_leadtrack), entry.jet_max_deltaR, entry.jet_Ftrack_Iso])
    track_index = 0
    tower_index = 0
    nTrack = int(entry.nTrack)
    nTower = int(entry.nTower)
    # TRACK_ARRAY: ['[index]',[P], [PT], [L], [D0], [DZ], [e], [e], [deltaEta], [deltaPhi], [deltaR]]
    tr_p = []
    tr_pt = []
    tr_l = []
    tr_d0 = []
    tr_dz = []
    tr_delEta = []
    tr_delPhi = []
    tr_delR = []
    # TOWER_ARRAY: ['[index]',[E], [ET], [Eta], [Phi], [Edges0], [Edges1], [Edges2], [Edges3], [Eem], [Ehad], [T],
    # [deltaEta], [deltaPhi], [deltaR]]
    to_e = []
    to_et = []
    to_eta = []
    to_phi = []
    to_ed0 = []
    to_ed1 = []
    to_ed2 = []
    to_ed3 = []
    to_eem = []
    to_ehad = []
    to_t = []
    to_deleta = []
    to_delphi = []
    to_delr = []
    for t in range(0, nTrack):
        tr_pt.append(entry.track_PT[t])
    index_sorted_tracks = sorted(range(len(tr_pt)), key=lambda k: tr_pt[k], reverse=True)
    for t in range(0, nTower):
        to_et.append(entry.tower_ET[t])
    index_sorted_towers = sorted(range(len(to_et)), key=lambda k: to_et[k], reverse=True)
    for idx in index_sorted_tracks:
        tr_p.append(np.log(entry.track_P[idx]))
        tr_pt.append(np.log(entry.track_PT[idx]))
        tr_l.append(np.log(entry.track_L[idx]))
        tr_d0.append(np.log(entry.track_D0[idx]))
        tr_dz.append(np.log(entry.track_DZ[idx]))
        tr_delEta.append(entry.track_deltaEta[idx])
        tr_delPhi.append(entry.track_deltaPhi[idx])
        tr_delR.append(entry.track_deltaR[idx])
        track_index += 1
    for jdx in index_sorted_towers:
        to_e.append(np.log(entry.tower_E[jdx]))
        to_et.append(np.log(entry.tower_ET[jdx]))
        to_eta.append(entry.tower_Eta[jdx])
        to_phi.append(entry.tower_Phi[jdx])
        to_ed0.append(entry.tower_Edges0[jdx])
        to_ed1.append(entry.tower_Edges1[jdx])
        to_ed2.append(entry.tower_Edges2[jdx])
        to_ed3.append(entry.tower_Edges3[jdx])
        to_eem.append(np.log(entry.tower_Eem[jdx]))
        to_ehad.append(np.log(entry.tower_Ehad[jdx]))
        to_t.append(np.log(entry.tower_T[jdx]))
        to_deleta.append(entry.tower_deltaEta[jdx])
        to_delphi.append(entry.tower_deltaPhi[jdx])
        to_delr.append(entry.tower_deltaR[jdx])
        tower_index += 1
    sig_track_array.append([tr_p, tr_pt, tr_l, tr_d0, tr_dz, tr_delEta, tr_delPhi, tr_delR])
    sig_tower_array.append([to_e, to_et, to_eta, to_phi, to_ed0, to_ed1, to_ed2, to_ed3, to_eem, to_ehad, to_t, to_deleta, to_delphi, to_delr])
    jet_index += 1

input_sigjet = np.array(sig_jet_array)
input_sigtrack = np.array(sig_track_array)
input_sigtower = np.array(sig_tower_array)

input_backjet = np.array(back_jet_array)
input_backtrack = np.array(back_track_array)
input_backtower = np.array(back_tower_array)



#Data preprocessing: Normalizing the data and inserting timesteps for tracks and towers
#
#TO FILL
#

#HL Layers
HL_input = Input(shape=(13,))
HLdense1 = Dense(128, activation='relu')(HL_input)
HLdense2 = Dense(128, activation='relu')(HLdense1)
HLdense3 = Dense(16, activation='relu')(HLdense2)

#Track Layers
Track_input1 = Input(shape=(None, 10))
#Track_input2 = Input(shape=(10,))

trackDense1 = Dense(32, activation='relu', input_shape=(None, None, 10))
trackDense2 = Dense(32, activation='relu', input_shape=(None, None, 10))
trackSD1 = TimeDistributed(trackDense1)(Track_input1)
trackSD2 = TimeDistributed(trackDense2)(trackSD1)

#mergeTrack = Concatenate()([trackSD1, trackSD2])
flatten = TimeDistributed(Flatten())(trackSD2)

trackLSTM1 = LSTM(32, return_sequences=True)(flatten)
trackLSTM2 = LSTM(32, return_sequences=False)(trackLSTM1)

#Tower Layers
Tower_input1 = Input(shape=(None,14))
#Tower_input2 = Input(shape=(14,))

towerDense1 = Dense(32, activation='relu', input_shape=(None, None, 14))
towerDense2 = Dense(32, activation='relu', input_shape=(None, None, 14))
towerSD1 = TimeDistributed(towerDense1)(Tower_input1)
towerSD2 = TimeDistributed(towerDense2)(towerSD1)

towerFlatten = TimeDistributed(Flatten())(towerSD2)

towerLSTM1 = LSTM(24, return_sequences=True)(towerFlatten)
towerLSTM2 = LSTM(24, return_sequences=False)(towerLSTM1)

#Layers Merged
mergedLayer = Concatenate()([trackLSTM2, towerLSTM2, HLdense3])
fullDense1 = Dense(64, activation='relu')(mergedLayer)
fullDense2 = Dense(32, activation='relu')(fullDense1)
Output = Dense(1, activation='sigmoid')(fullDense2)

RNNmodel = Model(inputs=[Track_input1, Tower_input1, HL_input], outputs=Output)
RNNmodel.save('tauRNN_1-prong.h5')
RNNmodel.summary()


