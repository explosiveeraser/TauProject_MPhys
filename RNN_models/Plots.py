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


    def __init__(self, name, real_y, pred_y, weights, train_y, train_pred_y, train_weights, jet_pt, legend=True, ylim=(1, 1e7)):
        self.real_y = real_y
        self.pred_y = pred_y
        self.ylim = ylim
        self.weights = weights
        self.train_y = train_y
        self.train_pred_y = train_pred_y
        self.train_weights = train_weights
        self.jet_pt = jet_pt
        self.legend = legend
        self.name = name
        eff, rej = self.roc(self.real_y, self.pred_y, sample_weight=self.weights)
        # eff, rej = self.roc(self.real_y, self.pred_y)
        self.eff = eff
        self.rej = rej

    def plot_raw_score_vs_jetPT(self):
        fig, ax = plt.subplots(2)
        y = []
        pt = []
        for i in trange(0, len(self.pred_y)):
            y.append(self.pred_y[i,0])
            pt.append(self.jet_pt[i])
        ax[0].hist2d(y, pt, bins=100)
        ax[1].hist2d(y, pt, bins=100, weights=self.weights)
        plt.show()


    def histogram_RNN_score(self):
        canvas = ROOT.TCanvas("RNN Scores")
        canvas.Divide(2,1)
        true_hist = ROOT.TH1D("Raw_RNN_score_trueTaus", "Raw_RNN_score_trueTaus", 50, 0., 1.)
        fake_hist = ROOT.TH1D("Raw_RNN_score_fakeTaus", "Raw_RNN_score_fakeTaus", 50, 0., 1.)
        true_hist_reweight = ROOT.TH1D("Reweight_RNN_score_trueTaus", "Reweight_RNN_score_trueTaus", 50, 0., 1.)
        fake_hist_reweight = ROOT.TH1D("Reweight_RNN_score_fakeTaus", "Reweight_RNN_score_fakeTaus", 50, 0., 1.)

    #    new_eff, _ = self.roc(self.real_y, self.pred_y)

   #     eff = new_eff[:][0]

  #      true_hist_test = ROOT.TH1D("TEST_rew_RNN_score_trueTaus", "TEST_rew_RNN_score_trueTaus", 50, 0., 1.)
   #     fake_hist_test = ROOT.TH1D("TEST_rew_RNN_score_fakeTaus", "TEST_rew_RNN_score_fakeTaus", 50, 0., 1.)

        test = 0

        for i in trange(0, len(self.pred_y)):
            if self.real_y[i] == 0.:
                fake_hist.Fill(self.pred_y[i])
                fake_hist_reweight.Fill(self.pred_y[i], self.weights[i])
           #     fake_hist_test.Fill(eff[i][0])
            elif self.real_y[i] > 0.5:
                test += 1
                true_hist.Fill(self.pred_y[i])
                true_hist_reweight.Fill(self.pred_y[i], self.weights[i])
          #      true_hist_test.Fill(eff[i][0])

        print(test)

        fake_hist_integral = fake_hist.Integral()
        true_hist_integral = true_hist.Integral()
        fake_hist.Scale(1/fake_hist_integral)
        true_hist.Scale(1/true_hist_integral)

        fake_hist_reweight_integral = fake_hist_reweight.Integral()
        true_hist_reweight_integral = true_hist_reweight.Integral()
        fake_hist_reweight.Scale(1/fake_hist_reweight_integral)
        true_hist_reweight.Scale(1/true_hist_reweight_integral)

        # fake_hist_test_integral = fake_hist_test.Integral()
        # true_hist_test_integral = true_hist_test.Integral()
        # fake_hist_test.Scale(1/fake_hist_test_integral)
        # true_hist_test.Scale(1/true_hist_test_integral)

        canvas.cd(1)
        fake_hist.Draw('HIST')
        true_hist.SetLineColor(ROOT.kRed)
        true_hist.Draw('HIST SAMES0')
        canvas.cd(2)
        fake_hist_reweight.Draw('HIST')
        true_hist_reweight.SetLineColor(ROOT.kRed)
        true_hist_reweight.Draw('HIST SAMES0')
        # canvas.cd(3)
        # fake_hist_test.Draw('HIST')
        # true_hist_test.SetLineColor(ROOT.kRed)
        # true_hist_test.Draw('HIST SAMES0')
        canvas.Update()
        canvas.Print(self.name + "_histogram_RNN_score.pdf")


    # def plot_efficiencies(self, sig_train, sig_eval, eff):
    #     flat = Flattener(Plots.pt_bins, eff)
    #     flat.fit(sig_train, self.train_pred_y)
    #
    #     pass_thr = flat.passes_thr(sig_eval, self.pred_y)
    #
    #     statistic, _, _ = binned_statistic(sig_eval, pass_thr, statistic=lambda arr: np.count_nonzero(arr) / float(len(arr)),
    #                                        bins=flat.x_bins)
    #     #plot
    #     fig, ax = plt.subplots()
    #     xx =


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
        #**kwargs
        fpr, tpr, thr = roc_curve(y_true, y, **kwargs)
        #fpr, tpr, thr = roc_curve(y_true, y)
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

    def plot_rej_vs_eff(self, name, save_dir):
        eff, rej = self.roc(self.real_y, self.pred_y, sample_weight=self.weights)
        #eff, rej = self.roc(self.real_y, self.pred_y)
        self.eff = eff
        self.rej = rej
        fig, ax = plt.subplots()
        ax.plot(eff, rej, color='g', label='Delphes Tau RNN')
        plt.imread("note_roc_curve.png")
        #ax.set_ylim(self.ylim)
        ax.set_xlim((0., 1.))
        ax.set_ylim((1., 1e4))
        ax.set_yscale("log")
        ax.set_xlabel("Signal efficiency", x=1, ha="right")
        ax.set_ylabel("Background rejection", y=1, ha="right")
        if self.legend:
            ax.legend()
        plt.savefig("{}{}.png".format(save_dir, name))
        return fig

    #Copied from Original RNN Project
    def plot_flatefficiency(self):
        # Flatten on training sample
        sig_train = sh.sig_train.get_variables("TauJets/pt", "TauJets/mu",
                                               self.pred_y)
        flat = Flattener(pt_bins, mu_bins, self.eff)
        flat.fit(sig_train["TauJets/pt"], sig_train["TauJets/mu"],
                 sig_train[self.score])

        # Efficiency on testing sample
        sig_test = sh.sig_test.get_variables("TauJets/pt", "TauJets/mu",
                                             self.score)
        pass_thr = flat.passes_thr(sig_test["TauJets/pt"],
                                   sig_test["TauJets/mu"],
                                   sig_test[self.score])

        statistic, _, _, _ = binned_statistic_2d(
            sig_test["TauJets/pt"], sig_test["TauJets/mu"], pass_thr,
            statistic=lambda arr: np.count_nonzero(arr) / float(len(arr)),
            bins=[flat.x_bins, flat.y_bins])

        # Plot
        fig, ax = plt.subplots()

        xx, yy = np.meshgrid(flat.x_bins, flat.y_bins - 0.5)
        cm = ax.pcolormesh(xx / 1000.0, yy, statistic.T)

        ax.set_xscale("log")
        ax.set_xlim(20, 2000)
        ax.set_xlabel(r"Reco. tau $p_\mathrm{T}$ / GeV",
                      ha="right", x=1)
        ax.set_ylim(0, 60)
        ax.set_ylabel(r"$\mu$", ha="right", y=1)

        cb = fig.colorbar(cm)
        cb.set_label("Signal efficiency", ha="right", y=1)

        label = r"$\epsilon_\mathrm{sig}$ = " + "{:.0f} %".format(
            100 * self.eff)
        ax.text(0.93, 0.85, label, ha="right", va="bottom", fontsize=7,
                transform=ax.transAxes)

        return fig



