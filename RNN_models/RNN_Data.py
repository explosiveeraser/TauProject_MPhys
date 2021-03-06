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
import os

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

    jet_keys = {"jet_PT", "jet_PT_LC_scale", "jet_f_cent", "jet_iF_leadtrack", "jet_max_deltaR",
                "jet_Ftrack_Iso", "jet_ratio_ToEem_P", "jet_frac_trEM_pt", "jet_mass_track_EM_system"
                , "jet_mass_track_system"}

    track_keys = {"track_PT", "track_D0", "track_DZ", "track_deltaEta",
                    "track_deltaPhi"}

    tower_keys = {"tower_ET", "tower_deltaEta", "tower_deltaPhi"}

    def __init__(self, Prongs, load_pickled_data, pickle_file, print_hists=True, BacktreeFile="", BackTreeName="",  SignaltreeFile="", SignalTreeName="", BackendPartOfTree="", SignalendPartOfTree=""):
        self.prong = Prongs
        self.dicts_file = "input_dicts"
        self.hists_before_trans = {}
        self.hists_after_trans = {}
        if not load_pickled_data:
            self.preprocessing = {}
            back_tree_file = TFile.Open("../NewTTrees/{}.root".format(BacktreeFile))
            sig_tree_file = TFile.Open("../NewTTrees/{}.root".format(SignaltreeFile))

            self.BackgroundTree = back_tree_file.Get("{}{}".format(BackTreeName, BackendPartOfTree))
            self.SignalTree = sig_tree_file.Get("{}{}".format(SignalTreeName, SignalendPartOfTree))
            SignalNumEntries = self.SignalTree.GetEntries()
            BackgroundNumEntries = self.BackgroundTree.GetEntries()
            # call function above for sig and back data
            #sig_jet, sig_tr, sig_to, sig_label = self.read_tree(self.SignalTree)
            #back_jet, back_tr, back_to, back_label = self.read_tree(self.BackgroundTree)

            jet_arr, track_arr, tower_arr, label = self.read_tree(self.BackgroundTree, self.SignalTree)
            self.input_jet = jet_arr
            self.input_track = track_arr
            self.input_tower = tower_arr
            self.Ytrain = label
            file = open(pickle_file, "wb")
            pickle.dump([self.input_track, self.input_tower, self.input_jet, self.Ytrain, [self.sig_pt, self.bck_pt]], file)
            file.close()
            self.bck_pt = self.bck_pt[0:50000]
            self.jet_pt = np.append(self.jet_pt[0:50000], self.jet_pt[-17212:-1], axis=0)
            self.input_jet = np.append(jet_arr[0:50000], jet_arr[-17212:-1], axis=0)
            self.input_track = np.append(track_arr[0:50000], track_arr[-17212:-1], axis=0)
            self.input_tower = np.append(tower_arr[0:50000], tower_arr[-17212:-1], axis=0)
            self.Ytrain = np.append(label[0:50000], label[-17212:-1], axis=0)
        elif load_pickled_data:
            #print(rn.root2array("../NewTTrees/{}.root".format(BacktreeFile), treename="{}{}".format(BackTreeName, BackendPartOfTree), branches=["track_PT"]))
            file = open(pickle_file, "rb")
            data = pickle.load(file)
            file.close()
            file = open("{}_untransformed_data".format(self.prong), "rb")
            [track_untrans, tower_untrans, jet_untrans, yval] = pickle.load(file)
            file.close()
            self.jet_pt = jet_untrans["jet_PT"]
            if print_hists:
                file = open("{}_transformed_data".format(self.prong), "rb")
                [track_trans, tower_trans, jet_trans, yval] = pickle.load(file)
                file.close()
                self.fill_untrans_hists(jet_untrans, track_untrans, tower_untrans, yval)
                self.fill_trans_hists(jet_trans, track_trans, tower_trans, yval)
                self.plot_hists()
            temp_jet = data[2]
            temp_track = data[0]
            temp_tower = data[1]
            temp_labels = data[3]
            self.sig_pt = data[4][0]
            self.bck_pt = data[4][1]
            seed = np.random.randint(1, 9999)
            rng = np.random.default_rng(seed)
            np.random.seed(seed)
            self.all_jet = temp_jet
            self.all_track = temp_track
            self.all_tower = temp_tower
            self.all_label = temp_labels
            self.bck_pt = self.bck_pt[0:50000]
            self.jet_pt = np.append(self.jet_pt[0:50000], self.jet_pt[-17212:-1], axis=0)
            self.input_jet = np.append(temp_jet[0:50000], temp_jet[-17212:-1], axis=0)
            self.input_track = np.append(temp_track[0:50000], temp_track[-17212:-1], axis=0)
            self.input_tower = np.append(temp_tower[0:50000], temp_tower[-17212:-1], axis=0)
            self.Ytrain = np.append(temp_labels[0:50000], temp_labels[-17212:-1], axis=0)


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
            offset = np.nanmean(arr)
        if std:
            scale = np.nanstd(arr)
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


    def sort_inputs(self, jet_dict, track_dict, tower_dict):
        jet_input = []
        track_input = []
        tower_input = []
        for idx in trange(0, len(jet_dict["jet_PT"])):
            temp = [jet_dict["jet_PT"][idx], jet_dict["jet_PT_LC_scale"][idx], jet_dict["jet_f_cent"][idx],
                    jet_dict["jet_iF_leadtrack"][idx], jet_dict["jet_max_deltaR"][idx], jet_dict["jet_Ftrack_Iso"][idx],
                    jet_dict["jet_ratio_ToEem_P"][idx], jet_dict["jet_frac_trEM_pt"][idx], jet_dict["jet_mass_track_EM_system"][idx],
                    jet_dict["jet_mass_track_system"][idx]]
            jet_input.append(np.asarray(temp).astype(np.float32))
        for idx in trange(0, len(track_dict["track_PT"])):
            temp1 = []
            index_track_sorted = sorted(range(len(track_dict["track_PT"][idx])),
                                        key=lambda k: track_dict["track_PT"][idx][k], reverse=True)
            for jdx in index_track_sorted:
                temp2 = [track_dict["track_PT"][idx][jdx], track_dict["track_D0"][idx][jdx],
                         track_dict["track_DZ"][idx][jdx], track_dict["track_deltaEta"][idx][jdx],
                         track_dict["track_deltaPhi"][idx][jdx]]
                if np.isnan(temp2[0]) or np.isnan(temp2[1]) or np.isnan(temp2[2]) or np.isnan(temp2[3]) or np.isnan(temp2[4]):
                    temp2 = np.zeros((len(temp2)), dtype=np.float32)
                temp1.append(np.asarray(temp2).astype(np.float32))
            track_input.append(np.array(temp1))
        for idx in trange(0, len(tower_dict["tower_ET"])):
            temp1 = []
            index_tower_sorted = sorted(range(len(tower_dict["tower_ET"][idx])), key=lambda k: tower_dict["tower_ET"][idx][k],
                                        reverse=True)
            for jdx in index_tower_sorted:
                temp2 = [tower_dict["tower_ET"][idx][jdx], tower_dict["tower_deltaEta"][idx][jdx],
                         tower_dict["tower_deltaPhi"][idx][jdx]]
                if np.isnan(temp2[0]) or np.isnan(temp2[1]) or np.isnan(temp2[2]):
                    temp2 = np.zeros((len(temp2)), dtype=np.float32)
                temp1.append(np.asarray(temp2).astype(np.float32))
            tower_input.append(np.array((temp1)))
        jet_input = np.array(jet_input)
        track_input = np.array(track_input)
        tower_input = np.array(tower_input)
        return jet_input, track_input, tower_input

    # untrans_jet_PT_bins = np.array([10., 25.178, 31.697, 39.905, 50.237, 63.245, 79.621, 100.000,
    #     130.000, 200.000, 316.978, 502.377, 796.214, 1261.914, 2000.000,
    #     1000000.000])
    #
    # untrans_jet_PT_LCscale_bins = np.array([10., 25.178, 31.697, 39.905, 50.237, 63.245, 79.621, 100.000,
    #                130.000, 200.000, 316.978, 502.377, 796.214, 1261.914, 2000.000,
    #                1000000.000])
    #
    # trans_jet_PT_LCscale_bins = np.array([10., 25.178, 31.697, 39.905, 50.237, 63.245, 79.621, 100.000,
    #                                         130.000, 200.000, 316.978, 502.377, 796.214, 1261.914, 2000.000,
    #                                         1000000.000]) / 250.
    #
    # untrans_f_cent_bins = np.log10(np.arange(50) * 1000)


    def fill_untrans_hists(self, jet, track, tower, label):
        if not os.path.exists("{}_untransformed_data".format(self.prong)):
            file = open("{}_untransformed_data".format(self.prong), "wb")
            pickle.dump([track, tower, jet, label], file)
            file.close()
        for key in tqdm(RNN_Data.jet_keys):
            min_val = np.min(jet[key].flatten().flatten())
            max_val = np.max(jet[key].flatten().flatten())
            for l in [0, 1]:
                arr_ = jet[key][label == l]
                arr = arr_.flatten().flatten()
                arr = arr[np.abs(arr) != 0.]
                # print(arr)
                self.hists_before_trans["{}_label{}".format(key, str(l))] = ROOT.TH1D("{}_{}".format(key, l), "{}_{}".format(key, l), 35,
                                                                    min_val, max_val)
                rn.fill_hist(self.hists_before_trans["{}_label{}".format(key, str(l))], arr)
        for key in tqdm(RNN_Data.track_keys):
            a = track[key].flatten().flatten()
            min_val = np.min(a[~np.isnan(a)])
            max_val = np.max(a[~np.isnan(a)])
            for l in [0, 1]:
                arr_ = track[key][label == l]
                print(arr_)
                arr = arr_.flatten().flatten().flatten()
                arr = arr[np.abs(arr) != 0.]
                arr = arr[~np.isnan(arr)]
                self.hists_before_trans["{}_label{}".format(key, str(l))] = ROOT.TH1D("{}_{}".format(key, l), "{}_{}".format(key, l),
                                                                    35, min_val,
                                                                    max_val)
                rn.fill_hist(self.hists_before_trans["{}_label{}".format(key, str(l))], arr)
        for key in tqdm(RNN_Data.tower_keys):
            a = tower[key].flatten().flatten()
            min_val = np.min(a[~np.isnan(a)])
            max_val = np.max(a[~np.isnan(a)])
            for l in [0, 1]:
                arr_ = tower[key][label == l]
                arr = arr_.flatten().flatten().flatten()
                arr = arr[np.abs(arr) != 0.]
                arr = arr[~np.isnan(arr)]
                self.hists_before_trans["{}_label{}".format(key, str(l))] = ROOT.TH1D("{}_{}".format(key, l), "{}_{}".format(key, l),
                                                                    35, min_val,
                                                                    max_val)
                rn.fill_hist(self.hists_before_trans["{}_label{}".format(key, str(l))], arr)


    def fill_trans_hists(self, jet, track, tower, label):
        if not os.path.exists("{}_transformed_data".format(self.prong)):
            file = open("{}_transformed_data".format(self.prong), "wb")
            pickle.dump([track, tower, jet, label], file)
            file.close()
        for key in tqdm(RNN_Data.jet_keys):
            min_val = np.min(jet[key].flatten().flatten())
            max_val = np.max(jet[key].flatten().flatten())
            for l in [0, 1]:
                arr_ = jet[key][label == l]
                arr = arr_.flatten().flatten()
                arr = arr[np.abs(arr) != 0.]
                # print(arr)
                self.hists_after_trans["{}_label{}".format(key, str(l))] = ROOT.TH1D("{}_{}".format(key, l), "{}_{}".format(key, l), 35,
                                                                    min_val, max_val)
                rn.fill_hist(self.hists_after_trans["{}_label{}".format(key, str(l))], arr)
        for key in tqdm(RNN_Data.track_keys):
            a = track[key].flatten().flatten()
            min_val = np.min(a[~np.isnan(a)])
            max_val = np.max(a[~np.isnan(a)])
            for l in [0, 1]:
                arr_ = track[key][label == l]
                arr = arr_.flatten().flatten().flatten()
                arr = arr[np.abs(arr) != 0.]
                arr = arr[~np.isnan(arr)]
                self.hists_after_trans["{}_label{}".format(key, str(l))] = ROOT.TH1D("{}_{}".format(key, l), "{}_{}".format(key, l),
                                                                    35, min_val,
                                                                    max_val)
                rn.fill_hist(self.hists_after_trans["{}_label{}".format(key, str(l))], arr)
        for key in tqdm(RNN_Data.tower_keys):
            a = tower[key].flatten().flatten()
            min_val = np.min(a[~np.isnan(a)])
            max_val = np.max(a[~np.isnan(a)])
            for l in [0, 1]:
                arr_ = tower[key][label == l]
                arr = arr_.flatten().flatten().flatten()
                arr = arr[np.abs(arr) != 0.]
                arr = arr[~np.isnan(arr)]
                self.hists_after_trans["{}_label{}".format(key, str(l))] = ROOT.TH1D("{}_{}".format(key, l), "{}_{}".format(key, l),
                                                                    35, min_val,
                                                                    max_val)
                rn.fill_hist(self.hists_after_trans["{}_label{}".format(key, str(l))], arr)

    def plot_hists(self):
        jet_i = 0

        legendJ = []
        legendTr = []
        legendTo = []
        jet_canvases = []


        for key in tqdm(RNN_Data.jet_keys):
            c_i = 0
            jet_canvas = ROOT.TCanvas("Jet_Inputs", "Jet_Inputs")
            jet_canvas.Divide(2, 1)
            jet_canvas.cd(c_i)
            c_i += 1
            jet_canvas.cd(c_i)
            for t in ["not_transformed", "transformed"]:
                legendJ.append(ROOT.TLegend(0.05, 0.85, 0.2, 0.95))
                if t == "not_transformed":
                    for l in [0, 1]:
                        if l == 0:
                          #  print(self.hists_before_trans)
                            integral = self.hists_before_trans["{}_label{}".format(key, str(l))].Integral()
                            if integral != 0.:
                                self.hists_before_trans["{}_label{}".format(key, str(l))].Scale(1 / integral)
                            self.hists_before_trans["{}_label{}".format(key, str(l))].Draw("HIST")
                        else:
                            integral = self.hists_before_trans["{}_label{}".format(key, str(l))].Integral()
                            if integral != 0.:
                                self.hists_before_trans["{}_label{}".format(key, str(l))].Scale(1 / integral)
                            self.hists_before_trans["{}_label{}".format(key, str(l))].SetLineColor(ROOT.kRed)
                            self.hists_before_trans["{}_label{}".format(key, str(l))].Draw("HIST SAMES0")
                        legendJ[jet_i].AddEntry(self.hists_before_trans["{}_label{}".format(key, l)], "{} Label: {}".format(key, l), "L")
                if t == "transformed":
                    for l in [0, 1]:
                        if l == 0:
                            integral = self.hists_after_trans["{}_label{}".format(key, str(l))].Integral()
                            if integral != 0.:
                                self.hists_after_trans["{}_label{}".format(key, str(l))].Scale(1 / integral)
                            self.hists_after_trans["{}_label{}".format(key, str(l))].Draw("HIST")
                        else:
                            integral = self.hists_after_trans["{}_label{}".format(key, str(l))].Integral()
                            if integral != 0.:
                                self.hists_after_trans["{}_label{}".format(key, str(l))].Scale(1 / integral)
                            self.hists_after_trans["{}_label{}".format(key, str(l))].SetLineColor(ROOT.kRed)
                            self.hists_after_trans["{}_label{}".format(key, str(l))].Draw("HIST SAMES0")
                        legendJ[jet_i].AddEntry(self.hists_after_trans["{}_label{}".format(key, l)], "{} Label: {}".format(key, l), "L")
                legendJ[jet_i].Draw()
                jet_canvas.Update()
                jet_i += 1
                jet_canvas.cd(c_i + 1)
            jet_canvas.Print("Jet_{}.pdf".format(key))
            jet_canvases.append([key, jet_canvas])
        track_i = 0

        track_canvases = []

        for key in tqdm(RNN_Data.track_keys):
            track_canvas = ROOT.TCanvas("Track_Inputs", "Track_Inputs")
            track_canvas.Divide(2, 1)
            c_i = 0
            track_canvas.cd(c_i)
            c_i += 1
            track_canvas.cd(c_i)
            for t in ["not_transformed", "transformed"]:
                legendTr.append(ROOT.TLegend(0.05, 0.85, 0.2, 0.95))
                if t == "not_transformed":
                    for l in [0, 1]:
                        if l == 0:
                            integral = self.hists_before_trans["{}_label{}".format(key, str(l))].Integral()
                            if integral != 0.:
                                self.hists_before_trans["{}_label{}".format(key, str(l))].Scale(1 / integral)
                            self.hists_before_trans["{}_label{}".format(key, str(l))].Draw("HIST")
                        else:
                            integral = self.hists_before_trans["{}_label{}".format(key, str(l))].Integral()
                            if integral != 0.:
                                self.hists_before_trans["{}_label{}".format(key, str(l))].Scale(1 / integral)
                            self.hists_before_trans["{}_label{}".format(key, str(l))].SetLineColor(ROOT.kRed)
                            self.hists_before_trans["{}_label{}".format(key, str(l))].Draw("HIST SAMES0")
                        legendTr[track_i].AddEntry(self.hists_before_trans["{}_label{}".format(key, l)], "{} Label: {}".format(key, l), "L")
                if t == "transformed":
                    for l in [0, 1]:
                        if l == 0:
                            integral = self.hists_after_trans["{}_label{}".format(key, str(l))].Integral()
                            if integral != 0.:
                                self.hists_after_trans["{}_label{}".format(key, str(l))].Scale(1 / integral)
                            self.hists_after_trans["{}_label{}".format(key, str(l))].Draw("HIST")
                        else:
                            integral = self.hists_after_trans["{}_label{}".format(key, str(l))].Integral()
                            if integral != 0.:
                                self.hists_after_trans["{}_label{}".format(key, str(l))].Scale(1 / integral)
                            self.hists_after_trans["{}_label{}".format(key, str(l))].SetLineColor(ROOT.kRed)
                            self.hists_after_trans["{}_label{}".format(key, str(l))].Draw("HIST SAMES0")
                        legendTr[track_i].AddEntry(self.hists_after_trans["{}_label{}".format(key, l)], "{} Label: {}".format(key, l), "L")
                legendTr[track_i].Draw()
                track_canvas.Update()
                track_i += 1
                track_canvas.cd(c_i + 1)
            track_canvas.Print("Track_{}.pdf".format(key))
            track_canvases.append([key, track_canvas])

        tower_i = 0
        tower_canvases = []

        for key in tqdm(RNN_Data.tower_keys):
            tower_canvas = ROOT.TCanvas("Tower_Inputs", "Tower_Inputs")
            tower_canvas.Divide(2, 1)
            c_i = 0
            tower_canvas.cd(c_i)
            c_i += 1
            tower_canvas.cd(c_i)
            for t in ["not_transformed", "transformed"]:
                legendTo.append(ROOT.TLegend(0.05, 0.85, 0.2, 0.95))
                if t == "not_transformed":
                    for l in [0, 1]:
                        if l == 0:
                            integral = self.hists_before_trans["{}_label{}".format(key, str(l))].Integral()
                            if integral != 0.:
                                self.hists_before_trans["{}_label{}".format(key, str(l))].Scale(1 / integral)
                            self.hists_before_trans["{}_label{}".format(key, str(l))].Draw("HIST")
                        else:
                            integral = self.hists_before_trans["{}_label{}".format(key, str(l))].Integral()
                            if integral != 0.:
                                self.hists_before_trans["{}_label{}".format(key, str(l))].Scale(1 / integral)
                            self.hists_before_trans["{}_label{}".format(key, str(l))].SetLineColor(ROOT.kRed)
                            self.hists_before_trans["{}_label{}".format(key, str(l))].Draw("HIST SAMES0")
                        legendTo[tower_i].AddEntry(self.hists_before_trans["{}_label{}".format(key, l)], "{} Label: {}".format(key, l), "L")
                if t == "transformed":
                    for l in [0, 1]:
                        if l == 0:
                            integral = self.hists_after_trans["{}_label{}".format(key, str(l))].Integral()
                            if integral != 0.:
                                self.hists_after_trans["{}_label{}".format(key, str(l))].Scale(1 / integral)
                            self.hists_after_trans["{}_label{}".format(key, str(l))].Draw("HIST")
                        else:
                            integral = self.hists_after_trans["{}_label{}".format(key, str(l))].Integral()
                            if integral != 0.:
                                self.hists_after_trans["{}_label{}".format(key, str(l))].Scale(1 / integral)
                            self.hists_after_trans["{}_label{}".format(key, str(l))].SetLineColor(ROOT.kRed)
                            self.hists_after_trans["{}_label{}".format(key, str(l))].Draw("HIST SAMES0")
                        legendTo[tower_i].AddEntry(self.hists_after_trans["{}_label{}".format(key, l)], "{} Label: {}".format(key, l), "L")
                legendTo[tower_i].Draw()
                tower_canvas.Update()
                tower_i += 1
                tower_canvas.cd(c_i + 1)
            tower_canvas.Print("Tower_{}.pdf".format(key))
            tower_canvases.append([key, tower_canvas])

        input("Enter to continue")
        return True



    def read_tree(self, backtree, sigtree):
        jet = {}
        track = {}
        tower = {}
        jet_untrans = {}
        track_untrans = {}
        tower_untrans = {}
        for jet_var in tqdm([ "jet_TruthTau", "jet_PT", "jet_PT_LC_scale", "jet_f_cent", "jet_iF_leadtrack", "jet_max_deltaR",
                         "jet_Ftrack_Iso", "jet_ratio_ToEem_P", "jet_frac_trEM_pt", "jet_mass_track_EM_system",
                              "jet_mass_track_system", "nTrack", "nTower"]):
            if jet_var != "jet_TruthTau":
                backjet = tree2array(backtree, branches=[jet_var]).astype(np.float32)
                sigjet = tree2array(sigtree, branches=[jet_var]).astype(np.float32)
                jet[jet_var] = np.append(backjet, sigjet, axis=0).astype(np.float32)
                jet_untrans[jet_var] = np.append(backjet, sigjet, axis=0).astype(np.float32)
                if jet_var == "nTrack":
                    max_nTrack = int(np.amax(jet["nTrack"]))
                if jet_var == "nTower":
                    max_nTower = int(np.max(jet["nTower"]))
                #if jet_var in ["jet_deltaEta", "jet_deltaPhi"]:
                 #   jet[jet_var] = self.abs_var(jet[jet_var])
                if jet_var in ["jet_PT", "jet_f_cent", "jet_iF_leadtrack"]:
                    if jet_var in ["jet_PT", "jet_f_cent", "jet_iF_leadtrack"]:
                        if jet_var in ["jet_PT"]:
                            self.sig_pt = sigjet
                            self.bck_pt = backjet
                            jet[jet_var] = self.log_epsilon(jet[jet_var])
                        elif jet_var in ["jet_f_cent", "jet_iF_leadtrack"]:
                            for idx in range(0, len(jet[jet_var])):
                                if jet[jet_var][idx] < 0.:
                                    jet[jet_var][idx] = random.uniform(0.5, 1.)*(5869*2)**(random.uniform(1., 2.))
                            jet[jet_var] = self.log_epsilon(jet[jet_var], epsilon=1e-6)
                if jet_var in ["jet_PT", "jet_PT_LC_scale", "jet_f_cent", "jet_iF_leadtrack", "jet_max_deltaR",
                "jet_Ftrack_Iso", "jet_ratio_ToEem_P", "jet_frac_trEM_pt", "jet_mass_track_EM_system"
                , "jet_mass_track_system"]:
                    jet[jet_var] = self.preprocess(jet_var, jet[jet_var], self.scale_flat)
            elif jet_var == "jet_TruthTau":
                backlabel = tree2array(backtree, branches=["jet_TruthTau"]).astype(np.float32)
                siglabel = tree2array(sigtree, branches=["jet_TruthTau"]).astype(np.float32)
                yLabel = np.append(backlabel, siglabel, axis=0)
        max_nTower = 100
        max_nTrack = 100
        for track_var in tqdm(["track_PT", "track_D0", "track_DZ", "track_deltaEta",
                          "track_deltaPhi"]):
            backtrack = tree2array(backtree, branches=[track_var])
            sigtrack = tree2array(sigtree, branches=[track_var])
            temp_arr = np.append(backtrack, sigtrack, axis=0)
            new_arr = np.empty((len(temp_arr), max_nTrack+1), dtype=np.float32)
            new_arr[:][:] = np.NaN
            for idx in range(0, len(temp_arr)):
                jdx = 0
                for val in temp_arr[idx]:
                    for v in val:
                        new_arr[idx][jdx] = v
                        jdx += 1
            track[track_var] = new_arr
            track_untrans[track_var] = new_arr
            if track_var == "track_PT":
                track_PT = track["track_PT"]
            if track_var in ["track_PT", "track_D0", "track_DZ"]:
                if track_var in ["track_PT"]:
                    track[track_var] = self.log_epsilon(track[track_var])
                    track[track_var] = self.preprocess(track_var, track[track_var], partial(self.scale, per_obj=False))
                if track_var in ["track_D0", "track_DZ"]:
                    track[track_var] = self.abs_log_epsilon(track[track_var], epsilon=0.000001)
                    track[track_var] = self.preprocess(track_var, track[track_var], partial(self.scale, per_obj=False))
            if track_var in ["track_deltaEta", "track_deltaPhi"]:
                track[track_var] = self.abs_var(track[track_var])
                track_untrans[track_var] = self.abs_var(track_untrans[track_var])
                track[track_var] = self.preprocess(track_var, track[track_var], partial(self.constant_scale, scale=0.6))
   #     print(sorted(range(len(track["track_PT"][3])), key=lambda k: track["track_PT"][3][k], reverse=True))
  #      print(sorted(range(len(track_PT[3])), key=lambda k: track_PT[3][k], reverse=True))
        for tower_var in tqdm(["tower_ET", "tower_deltaEta", "tower_deltaPhi"]):
            backtower = tree2array(backtree, branches=[tower_var])
            sigtower = tree2array(sigtree, branches=[tower_var])
            temp_arr = np.append(backtower, sigtower, axis=0)
            new_arr = np.empty((len(temp_arr), max_nTower), dtype=np.float32)
            new_arr[:][:] = np.NaN
            for idx in range(0, len(temp_arr)):
                jdx = 0
                for val in temp_arr[idx]:
                    for v in val:
                        new_arr[idx][jdx] = v
                        jdx += 1
            tower[tower_var] = new_arr
            tower_untrans[tower_var] = new_arr
            if tower_var == "tower_ET":
                tower_ET = tower["tower_ET"]
            if tower_var in ["tower_ET"]:
                if tower_var in ["tower_ET"]:
                    tower[tower_var] = self.log_epsilon(tower[tower_var])
                tower[tower_var] = self.preprocess(tower_var, tower[tower_var], partial(self.scale, per_obj=False))
            if tower_var in ["tower_deltaEta", "tower_deltaPhi", "tower_deltaR"]:
                tower[tower_var] = self.abs_var(tower[tower_var])
                tower_untrans[tower_var] = self.abs_var(tower_untrans[tower_var])
                tower[tower_var] = self.preprocess(tower_var, tower[tower_var], partial(self.constant_scale, scale=0.6))
            # if tower_var in ["Edges0", "Edges1", "Edges2", "Edges3"]:
            #     tower[tower_var] = self.abs_var(tower[tower_var]+np.amax(tower[tower_var]))
            #     tower[tower_var] = self.preprocess(tower_var, tower[tower_var], partial(self.min_max_scale, per_obj=False))
       # print(sorted(range(len(tower["tower_ET"][3])), key=lambda k: tower["tower_ET"][3][k], reverse=True))
      #  print(sorted(range(len(tower_ET[3])), key=lambda k: tower_ET[3][k], reverse=True))
        file = open(self.dicts_file, "wb")
        pickle.dump([jet, track, tower, yLabel], file)
        file.close()
        self.jet_pt = jet_untrans['jet_PT']
        self.fill_untrans_hists(jet_untrans, track_untrans, tower_untrans, yLabel)
        self.fill_trans_hists(jet, track, tower, yLabel)
        jet_input, track_input, tower_input = self.sort_inputs(jet, track, tower)
        return jet_input, track_input, tower_input, yLabel


























