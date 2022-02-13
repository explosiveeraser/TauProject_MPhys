import math
from collections import namedtuple

import numpy as np
import h5py
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic


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
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import roc_curve
from flattener import Flattener
from scipy.stats import binned_statistic

from sklearn.metrics import auc


from scipy.stats import binned_statistic_2d




class Plots():

    pt_bins = np.array([
        10., 25.178, 31.697, 39.905, 50.237, 63.245, 79.621, 100.000,
        130.000, 200.000, 316.978, 502.377, 796.214, 1261.914, 2000.000,
        1000000.000
    ])

    mu_bins = np.array([
        -0.5, 10.5, 19.5, 23.5, 27.5, 31.5, 35.5, 39.5, 49.5, 61.5
    ])


    def __init__(self, real_y, pred_y, weights, train_y, train_pred_y, train_weights, legend=True, ylim=(1, 1e7)):
        self.real_y = real_y
        self.pred_y = pred_y
        self.ylim = ylim
        self.weights = weights
        self.train_y = train_y
        self.train_pred_y = train_pred_y
        self.train_weights = train_weights
        self.legend = legend

    def plot_efficiencies(self, sig_train, sig_eval, eff):
        flat = Flattener(Plots.pt_bins, eff)
        flat.fit(sig_train, self.train_pred_y)

        pass_thr = flat.passes_thr(sig_eval, self.pred_y)

        statistic, _, _ = binned_statistic(sig_eval, pass_thr, statistic=lambda arr: np.count_nonzero(arr) / float(len(arr)),
                                           bins=flat.x_bins)
        #plot
        fig, ax = plt.subplots()
        xx =


    # def plotScore(self, kwargs):
    #     kwargs.setdefault("bins", 50)
    #     kwargs.setdefault("range", (0, 1))
    #     kwargs.setdefault("density", True)
    #     kwargs.setdefault("histtype", "step")
    #
    #     self.histopt = kwargs
    #
    #     fig, ax = plt.subplots()
    #     ax.hist(self.tr)

    def roc(self, y_true, y, **kwargs):
        fpr, tpr, thr = roc_curve(y_true, y, **kwargs)
        nonzero = fpr != 0
        eff, rej = tpr[nonzero], 1.0 / fpr[nonzero]

        return eff, rej

    def roc_ratio(self, y_true, y1, y2, **kwargs):
        eff1, rej1 = self.roc(y_true, y1, **kwargs)
        eff2, rej2 = self.roc(y_true, y2, **kwargs)

        roc1 = interp1d(eff1, rej1, copy=False)
        roc2 = interp1d(eff2, rej2, copy=False)

        lower_bound = max(eff1.min(), eff2.min())
        upper_bound = min(eff1.max(), eff2.max())

        eff = np.linspace(lower_bound, upper_bound, 100)
        ratio = roc1(eff) / roc2(eff)

        return eff, ratio

    def plot_rej_vs_eff(self):
        eff, rej = self.roc(self.real_y, self.pred_y, sample_weight=self.weights)
        fig, ax = plt.subplots()
        ax.plot(eff, rej, color='g', label='Delphes Tau RNN')
        ax.set_ylim(self.ylim)
        ax.set_xlim((0., 1.))
        ax.set_yscale("log")
        ax.set_xlabel("Signal efficiency", x=1, ha="right")
        ax.set_ylabel("Background rejection", y=1, ha="right")
        if self.legend:
            ax.legend()
        return fig


