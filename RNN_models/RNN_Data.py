from functools import partial
import math
import pickle
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
import root_numpy as rn
from root_numpy import tree2array




class RNN_Data():

    def __init__(self, Prongs, load_pickled_data, pickle_file, BacktreeFile="", BackTreeName="",  SignaltreeFile="", SignalTreeName="", BackendPartOfTree="", SignalendPartOfTree=""):
        self.prong = Prongs
        if not load_pickled_data:
            self.preprocessing = {}
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

            self.input_jet = temp_jet
            self.input_track = temp_track
            self.input_tower = temp_tower
            self.Ytrain = temp_labels


            file = open(pickle_file, "wb")
            pickle.dump([self.input_track, self.input_tower, self.input_jet, self.Ytrain], file)
            file.close()

            self.input_jet = rng.permutation(temp_jet, axis=0)
            self.input_track = rng.permutation(temp_track, axis=0)
            self.input_tower = rng.permutation(temp_tower, axis=0)
            self.Ytrain = rng.permutation(temp_labels, axis=0)

        elif load_pickled_data:
            #print(rn.root2array("../NewTTrees/{}.root".format(BacktreeFile), treename="{}{}".format(BackTreeName, BackendPartOfTree), branches=["track_PT"]))
            file = open(pickle_file, "rb")
            data = pickle.load(file)
            file.close()
            temp_jet = data[2]
            temp_track = data[0]
            temp_tower = data[1]
            temp_labels = data[3]
            rng = np.random.default_rng(123)
            np.random.seed(123)
            self.input_jet = rng.permutation(np.append(temp_jet[0:20000], temp_jet[-17212:-1], axis=0), axis=0)
            self.input_track = rng.permutation(np.append(temp_track[0:20000], temp_track[-17212:-1], axis=0), axis=0)
            self.input_tower = rng.permutation(np.append(temp_tower[0:20000], temp_tower[-17212:-1], axis=0), axis=0)
            self.Ytrain = rng.permutation(np.append(temp_labels[0:20000], temp_labels[-17212:-1], axis=0), axis=0)
            self.Ytrain = self.Ytrain.astype(np.int8)


    # TRACK_ARRAY: ['[index]',[P], [PT], [L], [D0], [DZ], [e], [e], [deltaEta], [deltaPhi], [deltaR]]
    # TOWER_ARRAY: ['[index]',[E], [ET], [Eta], [Phi], [Edges0], [Edges1], [Edges2], [Edges3], [Eem], [Ehad], [T],
    # [deltaEta], [deltaPhi], [deltaR]]

    #LOGs and ABS

    def log_epsilon(self, arr, epsilon=None):
        new_arr = arr
        if epsilon:
            new_arr = np.add(new_arr, epsilon)
        new_arr = np.log10(new_arr)
        return new_arr

    def abs_log_epsilon(self, arr, epsilon=None):
        new_arr = np.abs(arr)
        if epsilon:
            new_arr = np.add(new_arr, epsilon)
        new_arr = np.log10(new_arr)
        return new_arr

    def abs_var(self, arr):
        new_arr = np.abs(arr)
        return new_arr

    #SCALING
    def scale(self, arr, mean=True, std=True, per_obj=True):
        offset = np.zeros(arr.shape[1], dtype=np.float32)
        scale = np.ones(arr.shape[1], dtype=np.float32)

        if mean:
            if per_obj:
                np.nanmean(arr, out=offset, axis=0)
            else:
                offset[:] = np.nanmean(arr)
        if std:
            if per_obj:
                np.nanstd(arr, out=scale, axis=0)
            else:
                scale[:] = np.nanstd(arr)

        return offset, scale

    def scale_flat(self, arr, mean=True, std=True):
        offset = np.float32(0)
        scale = np.float32(1)

        if mean:
            offset = np.mean(arr)
        if std:
            scale = np.std(arr)

        return offset, scale

    def robust_scale(self, arr, median=True, interquartile=True,
                     low_perc=25.0, high_perc=75.0):
        offset = np.zeros(arr.shape[1], dtype=np.float32)
        scale = np.ones(arr.shape[1], dtype=np.float32)

        if median:
            np.nanmedian(arr, out=offset, axis=0)
        if interquartile:
            assert high_perc > low_perc
            perc = np.nanpercentile(arr, [high_perc, low_perc], axis=0)
            np.subtract.reduce(perc, out=scale)

        return offset, scale

    def max_scale(self, arr):
        offset = np.zeros(arr.shape[1], dtype=np.float32)
        scale = np.nanmax(arr, axis=0)

        return offset, scale

    def min_max_scale(self, arr, per_obj=True):
        if per_obj:
            offset = np.nanmin(arr, axis=0)
            scale = np.nanmax(arr, axis=0) - offset
        else:
            offset = np.nanmin(arr)
            scale = np.nanmax(arr) - offset
            offset = np.full(arr.shape[1], fill_value=offset, dtype=np.float32)
            scale = np.full(arr.shape[1], fill_value=scale, dtype=np.float32)

        return offset, scale

    def constant_scale(self, arr, offset=0.0, scale=1.0):
        offset = np.full(arr.shape[1], fill_value=offset, dtype=np.float32)
        scale = np.full(arr.shape[1], fill_value=scale, dtype=np.float32)

        return offset, scale

    def preprocess(self, feature, arr, func):
        new_arr = arr
        if func is not None:
            offset, scale = func(arr)
            new_arr -= offset
            new_arr /= scale
        else:
            num = arr.shape[1]
            offset = np.zeros((num,), dtype=np.float32)
            scale = np.ones((num,), dtype=np.float32)
        self.preprocessing[feature] = (offset, scale)
        return new_arr

#["track_P", "track_PT", "track_Eta", "track_Phi", "track_L", "track_D0", "track_DZ", "track_deltaEta",
 #                         "track_deltaPhi", "track_deltaR"]:

    def sort_inputs(self, jet_dict, track_dict, tower_dict):
        jet_input = []
        track_input = []
        tower_input = []
        for idx in trange(0, len(jet_dict["jet_PT"])):
            temp = [jet_dict["jet_PT"][idx], jet_dict["jet_Eta"][idx], jet_dict["jet_Phi"][idx], jet_dict["jet_deltaEta"][idx], jet_dict["jet_deltaPhi"][idx],
                    jet_dict["jet_deltaR"][idx], jet_dict["jet_charge"][idx], jet_dict["jet_NCharged"][idx], jet_dict["jet_NNeutral"][idx], jet_dict["jet_f_cent"][idx],
                    jet_dict["jet_iF_leadtrack"][idx], jet_dict["jet_max_deltaR"][idx], jet_dict["jet_Ftrack_Iso"][idx]]
            jet_input.append(np.asarray(temp).astype(np.float32))
        for idx in trange(0, len(track_dict["track_PT"])):
            temp1 = []
            index_track_sorted = sorted(range(len(track_dict["track_PT"][idx])),
                                        key=lambda k: track_dict["track_PT"][idx][k], reverse=True)
            for jdx in index_track_sorted:
                temp2 = [track_dict["track_P"][idx][jdx], track_dict["track_PT"][idx][jdx], track_dict["track_L"][idx][jdx], track_dict["track_D0"][idx][jdx], track_dict["track_DZ"][idx][jdx],
                         track_dict["track_deltaEta"][idx][jdx], track_dict["track_deltaPhi"][idx][jdx], track_dict["track_deltaR"][idx][jdx]]
                if np.isnan(temp2[0]) or np.isnan(temp2[1]) or np.isnan(temp2[2]) or np.isnan(temp2[3]) or np.isnan(temp2[4]) or np.isnan(temp2[5]) or np.isnan(temp2[6]) or np.isnan(temp2[7]):
                    temp2 = np.zeros((len(temp2)), dtype=np.float32)
                temp1.append(np.asarray(temp2).astype(np.float32))
            track_input.append(np.array(temp1))
        for idx in trange(0, len(tower_dict["tower_ET"])):
            temp1 = []
            index_tower_sorted = sorted(range(len(tower_dict["tower_ET"][idx])), key=lambda k: tower_dict["tower_ET"][idx][k],
                                        reverse=True)
            for jdx in index_tower_sorted:
                temp2 = [tower_dict["tower_E"][idx][jdx], tower_dict["tower_ET"][idx][jdx], tower_dict["tower_Eta"][idx][jdx], tower_dict["tower_Phi"][idx][jdx],
                         tower_dict["tower_Edges0"][idx][jdx], tower_dict["tower_Edges1"][idx][jdx], tower_dict["tower_Edges2"][idx][jdx], tower_dict["tower_Edges3"][idx][jdx]
                         , tower_dict["tower_Eem"][idx][jdx], tower_dict["tower_Ehad"][idx][jdx], tower_dict["tower_T"][idx][jdx], tower_dict["tower_deltaEta"][idx][jdx],
                         tower_dict["tower_deltaPhi"][idx][jdx], tower_dict["tower_deltaR"][idx][jdx]]
                if np.isnan(temp2[0]) or np.isnan(temp2[1]) or np.isnan(temp2[2]) or np.isnan(temp2[3]) or np.isnan(
                        temp2[4]) or np.isnan(temp2[5]) or np.isnan(temp2[6]) or np.isnan(temp2[7]) or np.isnan(
                        temp2[8]) or np.isnan(temp2[9]) or np.isnan(temp2[10]) or np.isnan(temp2[11]) or np.isnan(
                        temp2[12]) or np.isnan(temp2[13]):
                    temp2 = np.zeros((len(temp2)), dtype=np.float32)
                temp1.append(np.asarray(temp2).astype(np.float32))
            tower_input.append(np.array((temp1)))
        jet_input = np.array(jet_input)
        track_input = np.array(track_input)
        tower_input = np.array(tower_input)
        return jet_input, track_input, tower_input

    def read_tree(self, tree):
        jet = {}
        track = {}
        tower = {}
        for jet_var in [ "jet_TruthTau", "jet_PT", "jet_Eta", "jet_Phi", "jet_deltaEta", "jet_deltaPhi", "jet_deltaR",
                         "jet_charge", "jet_NCharged", "jet_NNeutral", "jet_f_cent", "jet_iF_leadtrack", "jet_max_deltaR",
                         "jet_Ftrack_Iso", "nTrack", "nTower"]:
            if jet_var != "jet_TruthTau":
                feature = tree2array(tree, branches=[jet_var]).astype(np.float32)
                jet[jet_var] = tree2array(tree,  branches=[jet_var]).astype(np.float32)
                if jet_var is "nTrack":
                    max_nTrack = int(np.amax(jet["nTrack"]))
                if jet_var is "nTower":
                    max_nTower = int(np.max(jet["nTower"]))
                if jet_var in ["jet_PT", "jet_Eta", "jet_Phi", "jet_charge", "jet_f_cent", "jet_iF_leadtrack"]:
                    if jet_var in ["jet_PT", "jet_f_cent", "jet_iF_leadtrack"]:
                        if jet_var in ["jet_PT"]:
                            jet[jet_var] = self.log_epsilon(jet[jet_var])
                        elif jet_var in ["jet_f_cent", "jet_iF_leadtrack"]:
                            for idx in range(0, len(jet[jet_var])):
                                if jet[jet_var][idx] < 0:
                                    jet[jet_var][idx] = np.inf
                            jet[jet_var] = self.log_epsilon(jet[jet_var])
                jet[jet_var] = self.preprocess(jet_var, feature, self.scale_flat)
            elif jet_var == "jet_TruthTau":
                yLabel = tree2array(tree, branches=["jet_TruthTau"]).astype(np.int)
        max_nTower = 100
        max_nTrack = 100
        for track_var in ["track_P", "track_PT", "track_Eta", "track_Phi", "track_L", "track_D0", "track_DZ", "track_deltaEta",
                          "track_deltaPhi", "track_deltaR"]:
            temp_arr = tree2array(tree, branches=[track_var])
            new_arr = np.empty((len(temp_arr), max_nTrack+1), dtype=np.float32)
            new_arr[:][:] = np.NaN
            for idx in range(0, len(temp_arr)):
                jdx = 0
                for val in temp_arr[idx]:
                    for v in val:
                        new_arr[idx][jdx] = v
                        jdx += 1
            track[track_var] = new_arr
            if track_var == "track_PT":
                track_PT = track["track_PT"]
            if track_var in ["track_P", "track_PT", "track_L", "track_D0", "track_DZ"]:
                if track_var in ["track_P", "track_PT", "track_L"]:
                    track[track_var] = self.log_epsilon(track[track_var])
                    track[track_var] = self.preprocess(track_var, track[track_var], partial(self.scale, per_obj=False))
                if track_var in ["track_D0", "track_DZ"]:
                    track[track_var] = self.abs_log_epsilon(track[track_var], epsilon=0.000001)
                    track[track_var] = self.preprocess(track_var, track[track_var], partial(self.scale, per_obj=False))
            if track_var in ["track_deltaEta", "track_deltaPhi", "track_deltaR"]:
                track[track_var] = self.preprocess(track_var, track[track_var], partial(self.constant_scale, scale=0.6))
   #     print(sorted(range(len(track["track_PT"][3])), key=lambda k: track["track_PT"][3][k], reverse=True))
  #      print(sorted(range(len(track_PT[3])), key=lambda k: track_PT[3][k], reverse=True))
        for tower_var in ["tower_E", "tower_ET", "tower_Eta", "tower_Phi", "tower_Edges0", "tower_Edges1", "tower_Edges2",
                          "tower_Edges3", "tower_Eem", "tower_Ehad", "tower_T", "tower_deltaEta", "tower_deltaPhi", "tower_deltaR"]:
            temp_arr = tree2array(tree, branches=[tower_var])
            new_arr = np.empty((len(temp_arr), max_nTower), dtype=np.float32)
            new_arr[:][:] = np.NaN
            for idx in range(0, len(temp_arr)):
                jdx = 0
                for val in temp_arr[idx]:
                    for v in val:
                        new_arr[idx][jdx] = v
                        jdx += 1
            tower[tower_var] = new_arr
            if tower_var == "tower_ET":
                tower_ET = tower["tower_ET"]
            if tower_var in ["tower_E", "tower_ET", "tower_Eem", "tower_Ehad", "tower_T"]:
                if tower_var in ["tower_E", "tower_ET"]:
                    tower[tower_var] = self.log_epsilon(tower[tower_var])
                elif tower_var in ["tower_Eem", "tower_Ehad", "tower_T"]:
                    tower[tower_var] = self.log_epsilon(tower[tower_var], epsilon=0.000001)
                tower[tower_var] = self.preprocess(tower_var, tower[tower_var], partial(self.scale, per_obj=False))
            if tower_var in ["tower_Eta", "tower_Phi"]:
                tower[tower_var] = self.abs_var(tower[tower_var])
                tower[tower_var] = self.preprocess(tower_var, tower[tower_var], partial(self.scale, per_obj=False))
            if tower_var in ["tower_deltaEta", "tower_deltaPhi", "tower_deltaR"]:
                tower[tower_var] = self.abs_var(tower[tower_var])
                tower[tower_var] = self.preprocess(tower_var, tower[tower_var], partial(self.constant_scale, scale=0.6))
            if tower_var in ["Edges0", "Edges1", "Edges2", "Edges3"]:
                tower[tower_var] = self.abs_var(tower[tower_var])
                tower[tower_var] = self.preprocess(tower_var, tower[tower_var], partial(self.min_max_scale, per_obj=False))
       # print(sorted(range(len(tower["tower_ET"][3])), key=lambda k: tower["tower_ET"][3][k], reverse=True))
      #  print(sorted(range(len(tower_ET[3])), key=lambda k: tower_ET[3][k], reverse=True))
        jet_input, track_input, tower_input = self.sort_inputs(jet, track, tower)
        return jet_input, track_input, tower_input, yLabel


























