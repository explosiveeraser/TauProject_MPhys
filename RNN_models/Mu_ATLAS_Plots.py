
import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
from Mu_ATLAS_Utils import colors, colorseq, roc, roc_ratio, \
    binned_efficiency_ci
from flattener import Flattener
from scipy.stats import binned_statistic, binned_statistic_2d

def mpl_setup(scale=0.49, aspect_ratio=8.0 / 6.0,
              pad_left=0.16, pad_bottom=0.18,
              pad_right=0.95, pad_top=0.95):
    mpl.rcParams["font.sans-serif"] = ["Liberation Sans", "helvetica",
                                       "Helvetica", "Arial"]
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.size"] = 8
    mpl.rcParams["mathtext.default"] = "regular"

    # LaTeX \the\textwidth
    text_width_pt = 451.58598
    inches_per_pt = 1.0 / 72.27
    fig_width = text_width_pt * inches_per_pt * scale
    fig_height = fig_width / aspect_ratio

    mpl.rcParams["figure.figsize"] = [fig_width, fig_height]

    mpl.rcParams["figure.subplot.left"] = pad_left
    mpl.rcParams["figure.subplot.bottom"] = pad_bottom
    mpl.rcParams["figure.subplot.top"] = pad_top
    mpl.rcParams["figure.subplot.right"] = pad_right

    mpl.rcParams["axes.xmargin"] = 0.0
    mpl.rcParams["axes.ymargin"] = 0.0

    mpl.rcParams["axes.labelsize"] = 10
    mpl.rcParams["axes.linewidth"] = 0.6

    mpl.rcParams["xtick.major.size"] = 6.0
    mpl.rcParams["xtick.major.width"] = 0.6
    mpl.rcParams["xtick.minor.size"] = 3.0
    mpl.rcParams["xtick.minor.width"] = 0.6
    mpl.rcParams["xtick.minor.visible"] = True
    mpl.rcParams["xtick.top"] = True
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["xtick.labelsize"] = 8

    mpl.rcParams["ytick.major.size"] = 6.0
    mpl.rcParams["ytick.major.width"] = 0.6
    mpl.rcParams["ytick.minor.size"] = 3.0
    mpl.rcParams["ytick.minor.width"] = 0.6
    mpl.rcParams["ytick.minor.visible"] = True
    mpl.rcParams["ytick.right"] = True
    mpl.rcParams["ytick.direction"] = "in"
    mpl.rcParams["ytick.labelsize"] = 8

    mpl.rcParams["legend.frameon"] = False

    mpl.rcParams["lines.linewidth"] = 1.1
    mpl.rcParams["lines.markersize"] = 3.0

pt_bins = np.array([
    20., 25.178, 31.697, 39.905, 50.237, 63.245, 79.621, 100.000,
    130.000, 200.000, 316.978, 502.377, 796.214, 1261.914, 2000.000,
    1000000.000
])

#mu_bins = np.array([
 #   0, 10, 12, 14, 16, 18, 20, 22, 24, 50
  #  ]) * 2

mu_bins = np.array([
    -0.5, 10.5, 19.5, 23.5, 27.5, 31.5, 35.5, 39.5, 49.5, 101.5
])

class Plot(object):
    def __init__(self):
        pass

class ScorePlot(Plot):
    def __init__(self, train=False, test=True, log_y=False, **kwargs):
        super(ScorePlot, self).__init__()

        self.train = train
        self.test = test
        self.log_y = log_y

        kwargs.setdefault("bins", 50)
        kwargs.setdefault("range", (0, 1))
        kwargs.setdefault("density", True)
        kwargs.setdefault("histtype", "step")

        self.histopt = kwargs

    def plot_flattened_sig_data(self, sig_train, bkg_train, sig_train_weight, bkg_train_weight,
             sig_test, bkg_test, sig_test_weight, bkg_test_weight, name, save_dir):

        #plot
        fig, ax = plt.subplots()
        print("sig train shape : {}".format(np.shape(sig_train)))
        print("sig train weight shape : {}".format(np.shape(sig_train_weight)))
        print("bkg train shape : {}".format(np.shape(bkg_train)))
        print("bkg train weight shape : {}".format(np.shape(bkg_train_weight)))

        print("sig test shape : {}".format(np.shape(sig_test)))
        print("sig test weight shape : {}".format(np.shape(sig_test_weight)))
        print("bkg test shape : {}".format(np.shape(bkg_test)))
        print("bkg test weight shape : {}".format(np.shape(bkg_test_weight)))

        if self.train:
            sig_train_percent = np.linspace(0, 100, 26)
            print(sig_train_percent)
            sig_train_binedges = np.percentile(sig_train, sig_train_percent)
            #sig_train_binedges[-1] = 1.001
            bck_trainbins, _ = np.histogram(bkg_train, bins=sig_train_binedges, weights=bkg_train_weight)
            sig_trainbins, _ = np.histogram(sig_train, bins=sig_train_binedges, weights=sig_train_weight)
            sig_train_percent /= 100.0
            x = (sig_train_percent[1:] + sig_train_percent[:-1]) / 2.0
            print(x)
            print(sig_train_binedges)
            print(sig_trainbins)
            print(bck_trainbins)
            ax.hist(x, bins=sig_train_percent, weights=bck_trainbins,
                    color=colors["violet"], label="Bkg. train", histtype="step", density=True)
            ax.hist(x, bins=sig_train_percent, weights=sig_trainbins,
                    color=colors["green"], label="Sig. train", histtype="step", density=True)

        if self.test:
            sig_test_percent = np.linspace(0.0, 100.0, 26)
            sig_test_binedges = np.percentile(sig_test, sig_test_percent)
            #sig_test_binedges[-1] = 1.001
            bck_testbins, _ = np.histogram(bkg_test, bins=sig_test_binedges, weights=bkg_test_weight)
            sig_testbins, _ = np.histogram(sig_test, bins=sig_test_binedges, weights=sig_test_weight)
            sig_test_percent /= 100.0
            x = (sig_test_percent[1:] + sig_test_percent[:-1]) / 2.0
            ax.hist(x,  bins=sig_test_percent, weights=bck_testbins,
                    color=colors["blue"], label="Bkg. train", histtype="step", density=True)
            ax.hist(x, bins=sig_test_percent, weights=sig_testbins,
                    color=colors["red"], label="Sig. test", histtype="step", density=True)

        ax.legend()
        if self.log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Signal probability", x=1, ha="right")
        ax.set_ylabel("Norm. number of entries", y=1, ha="right")
        ax.set_ylim((10e-5, 10e0))
        # Set y-range limits
        ax.autoscale()
        y_lo, y_hi = ax.get_ylim()

        if self.log_y:
            y_hi *= 1.4
            y_lo /= 1.4
        else:
            diff = y_hi - y_lo
            y_hi += 0.05 * diff
        y_lo = 10e-5
        ax.set_ylim(y_lo, y_hi)

        plt.savefig("{}flattened_{}_score_plot.png".format(save_dir, name))
        return fig

    def plot(self, sig_train, bkg_train, sig_train_weight, bkg_train_weight,
             sig_test, bkg_test, sig_test_weight, bkg_test_weight, name, save_dir):

        # Plot
        fig, ax = plt.subplots()
        if self.train:
            ax.hist(sig_train, weights=sig_train_weight,
                    color=colors["green"], label="Sig. train", **self.histopt)
            ax.hist(bkg_train, weights=bkg_train_weight,
                    color=colors["violet"], label="Bkg. train", **self.histopt)

        if self.test:
            ax.hist(sig_test, weights=sig_test_weight,
                    color=colors["red"], label="Sig. test", **self.histopt)
            ax.hist(bkg_test, weights=bkg_test_weight,
                    color=colors["blue"], label="Bkg. test", **self.histopt)

        if self.log_y:
            ax.set_yscale("log")

        ax.legend()
        ax.set_xlabel("Signal probability", x=1, ha="right")
        ax.set_ylabel("Norm. number of entries", y=1, ha="right")
        ax.set_ylim((10e-3, 10e2))
        # Set y-range limits
        ax.autoscale()
        y_lo, y_hi = ax.get_ylim()

        if self.log_y:
            y_hi *= 1.4
            y_lo /= 1.4
        else:
            diff = y_hi - y_lo
            y_hi += 0.05 * diff
        y_lo = 10e-3
        ax.set_ylim(y_lo, y_hi)

        plt.savefig("{}{}_score_plot.png".format(save_dir, name))
        return fig


# class ROC(Plot):
#     def __init__(self, scores, legend=True, ylim=(1, 1e4)):
#         super(ROC, self).__init__()
#
#         if not isinstance(scores, list):
#             self.scores = [scores]
#         else:
#             self.scores = scores
#
#         self.legend = legend
#         self.ylim = ylim
#
#
#     def plot(self, sig_test, bkg_test, sig_test_weight, bkg_test_weight):
#         y_true = np.concatenate([np.ones_like(sig_test_weight),
#                                  np.zeros_like(bkg_test_weight)])
#         weights = np.concatenate([sig_test_weight, bkg_test_weight])
#
#         rocs = []
#         for s in self.scores:
#             y = np.concatenate([sig_test[s], bkg_test[s]])
#             eff, rej = roc(y_true, y, sample_weight=weights)
#             rocs.append((eff, rej))
#
#         # Plot
#         fig, ax = plt.subplots()
#
#         for s, (eff, rej), c in zip(self.scores, rocs, colorseq):
#             label = s.split("/")[-1]
#             ax.plot(eff, rej, color=c, label=label)
#
#         ax.set_ylim(self.ylim)
#         ax.set_yscale("log")
#         ax.set_xlabel("Signal efficiency", x=1, ha="right")
#         ax.set_ylabel("Background rejection", y=1, ha="right")
#
#         if self.legend:
#             ax.legend()
#
#         return fig


# class ROCRatio(Plot):
#     def __init__(self, ratios, legend=True, ylim=(0.9, 2.5)):
#         super(ROCRatio, self).__init__()
#
#         if not isinstance(ratios, list):
#             self.ratios = [ratios]
#         else:
#             self.ratios = ratios
#
#         scores = []
#         for num, denom in ratios:
#             scores.append(num)
#             scores.append(denom)
#
#         self.scores = scores
#         self.legend = legend
#         self.ylim = ylim
#
#
#     def plot(self, sh):
#         sig_test = sh.sig_test.get_variables("TauJets/pt", *self.scores)
#         bkg_test = sh.bkg_test.get_variables("TauJets/pt", *self.scores)
#         sig_test_weight, bkg_test_weight = pt_reweight(
#             sig_test["TauJets/pt"], bkg_test["TauJets/pt"])
#
#         y_true = np.concatenate([np.ones_like(sig_test_weight),
#                                  np.zeros_like(bkg_test_weight)])
#         weights = np.concatenate([sig_test_weight, bkg_test_weight])
#
#         ratios = []
#         for num, denom in self.ratios:
#             y1 = np.concatenate([sig_test[num], bkg_test[num]])
#             y2 = np.concatenate([sig_test[denom], bkg_test[denom]])
#             eff, ratio = roc_ratio(y_true, y1, y2, sample_weight=weights)
#             ratios.append((eff, ratio))
#
#         # Plot
#         fig, ax = plt.subplots()
#
#         for (num, denom), (eff, ratio), c in zip(self.ratios, ratios, colorseq):
#             num = num.split("/")[-1]
#             denom = denom.split("/")[-1]
#             label = "{} / {}".format(num, denom)
#             ax.plot(eff, ratio, color=c, label=label)
#
#         ax.set_ylim(self.ylim)
#         ax.set_xlabel("Signal efficiency", x=1, ha="right")
#         ax.set_ylabel("Rejection ratio", y=1, ha="right")
#
#         if self.legend:
#             ax.legend()
#
#         return fig


class FlattenerCutmapPlot(Plot):
    def __init__(self, score, sig_pt, sig_mu, eff):
        super(FlattenerCutmapPlot, self).__init__()
        self.sig_pt = sig_pt
        self.sig_mu = sig_mu
        self.score = score
        self.eff = eff


    def plot(self):
        # Flatten on training sample
        flat = Flattener(pt_bins, mu_bins, self.eff)
        flat.fit(self.sig_pt, self.sig_mu, self.score)

        fig, ax = plt.subplots()

        #plot
        xx, yy = np.meshgrid(flat.x_bins, flat.y_bins - 0.5)
        cm = ax.pcolormesh(xx / 1000.0, yy, flat.cutmap.T)

        ax.set_xscale("log")
        ax.set_xlim(20, 2000)
        ax.set_xlabel(r"Reco. tau $p_\mathrm{T}$ / GeV",
                      ha="right", x=1)
        ax.set_ylim(0, 60)
        ax.set_ylabel(r"$\mu$", ha="right", y=1)

        cb = fig.colorbar(cm)
        cb.set_label("Score threshold", ha="right", y=1)

        label = r"$\epsilon_\mathrm{sig}$ = " + "{:.0f} %".format(100 * self.eff)
        ax.text(0.93, 0.85, label, ha="right", va="bottom", fontsize=7,
                transform=ax.transAxes)

        return fig


class FlattenerEfficiencyPlot(Plot):
    def __init__(self, sig_train_pt, sig_train_score, sig_train_mu, sig_test_pt, sig_test_score, sig_test_mu,
                 eff):
        super(FlattenerEfficiencyPlot, self).__init__()

        self.sig_train_pt = sig_train_pt
        self.sig_train_score = sig_train_score
        self.sig_train_mu = sig_train_mu
        self.sig_test_pt = sig_test_pt
        self.sig_test_score = sig_test_score
        self.sig_test_mu = sig_test_mu
        self.eff = eff


    def plot(self):
        # Flatten on training sample
        flat = Flattener(pt_bins, mu_bins, self.eff)
        flat.fit(self.sig_train_pt, self.sig_train_mu, self.sig_train_score)

        # Efficiency on testing sample
        pass_thr = flat.passes_thr(self.sig_test_pt, self.sig_test_mu, self.sig_test_score)

        statistic, _, _, _ = binned_statistic_2d(
            self.sig_test_pt, self.sig_test_mu, pass_thr,
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


class EfficiencyPlot(Plot):
    def __init__(self, eff, colors, bins=10, scale=1.0, label=None,
                 ylim=None):
        super(EfficiencyPlot, self).__init__()

        self.label = label

        self.eff = eff
        self.colors = colors
        self.bins = bins / scale
        self.scale = scale
        self.ylim = ylim


    def plot(self, sig_train_pt, sig_train_score, sig_train_mu, sig_test_pt, sig_test_score,
             sig_test_mu, name, xvar_name, xvar, xvar_weight, save_dir):

        flat = []
        pass_thr = []
        efficiencies = []

        # Determine flattening on training sample for all scores
        for idx in range(0, len(self.eff)):
            flat.append(Flattener(pt_bins, mu_bins, self.eff[idx]))
            flat[idx].fit(sig_train_pt, sig_train_mu, sig_train_score)

        # Check which events pass the working point for each score
            pass_thr.append(flat[idx].passes_thr(sig_test_pt, sig_test_mu, sig_test_score))

            efficiencies.append(binned_efficiency_ci(xvar, pass_thr[idx],
                                            bins=self.bins, weight=xvar_weight, nbootstrap=200))

        print(efficiencies)

        # Plot
        fig, ax = plt.subplots()

        bin_center = self.scale * (self.bins[1:] + self.bins[:-1]) / 2.0
        bin_half_width = self.scale * (self.bins[1:] - self.bins[:-1]) / 2.0
        if self.label != None:
            for idx in range(0, len(self.eff)):
                for z, (eff, c, label) in enumerate(
                        zip(efficiencies[idx], colorseq, self.label)):
                    ci_lo, ci_hi = eff.ci
                    yerr = np.vstack([eff.median - ci_lo, ci_hi - eff.median])
                    ax.errorbar(bin_center, eff.median,
                                xerr=bin_half_width,
                                yerr=yerr,
                                fmt="o", ms=0.25, color=colors[self.colors[idx]], label=label, zorder=z)

        else:
            for idx in range(0, len(self.eff)):
                for z, (eff, c) in enumerate(
                        zip(efficiencies[idx], colorseq)):
                    ci_lo, ci_hi = eff.ci
                    yerr = np.vstack([eff.median - ci_lo, ci_hi - eff.median])

                    ax.errorbar(bin_center, eff.median,
                                xerr=bin_half_width,
                                yerr=yerr,
                                fmt="o", ms=0.25,  color=colors[self.colors[idx]], label="Working point: {}".format(self.eff[idx]), zorder=z)

        if self.ylim:
            ax.set_ylim(self.ylim)
        else:
            y_lo, y_hi = ax.get_ylim()
            d = 0.05 * (y_hi - y_lo)
            ax.set_ylim(y_lo - d, y_hi + d)

        ax.set_xlabel(xvar_name.split("/")[-1], x=1, ha="right")
        ax.set_ylabel("Efficiency {}".format(xvar_name), y=1, ha="right")
        ax.legend()

        plt.savefig("{}{}.png".format(save_dir, name))
        return fig


class RejectionPlot(Plot):
    def __init__(self,eff, colors, bins=10, scale=1.0, label=None,
                 ylim=None):
        super(RejectionPlot, self).__init__()

        self.label = label

        self.eff = eff
        self.colors = colors
        self.bins = bins / scale
        self.scale = scale
        self.ylim = ylim


    def plot(self, sig_train_pt, sig_train_score, sig_train_mu, sig_test_pt, sig_test_weight,
             bkg_test_pt, bkg_test_weight, bkg_test_score, bkg_test_mu, bkg_test_xvar, name, xvar_name, save_dir):

        flat = []
        pass_thr = []
        rejections = []

        for idx in range(0, len(self.eff)):
            flat.append(Flattener(pt_bins, mu_bins, self.eff[idx]))
            flat[idx].fit(sig_train_pt, sig_train_mu, sig_train_score)

            # Check which events pass the working point for each score
            pass_thr.append(flat[idx].passes_thr(bkg_test_pt, bkg_test_mu, bkg_test_score))

            rejections.append(binned_efficiency_ci(bkg_test_xvar, pass_thr[idx],
                                          weight=bkg_test_weight,
                                          bins=self.bins, return_inverse=True, nbootstrap=200))

        print(rejections)

        # Plot
        fig, ax = plt.subplots()

        bin_center = self.scale * (self.bins[1:] + self.bins[:-1]) / 2.0
        bin_half_width = self.scale * (self.bins[1:] - self.bins[:-1]) / 2.0



        if self.label != None:
            for idx in range(0, len(self.eff)):
                for z, (rej, c, label) in enumerate(
                        zip(rejections[idx], colorseq, self.label)):
                    ci_lo, ci_hi = rej.ci
                    yerr = np.vstack([rej.median - ci_lo, ci_hi - rej.median])
                    ax.errorbar(bin_center, rej.median,
                                xerr=bin_half_width,
                                yerr=yerr,
                                fmt="o",  ms=0.25, color=colors[self.colors[idx]], label=label, zorder=z)
        else:
            for idx in range(0, len(self.eff)):
                for z, (rej, c) in enumerate(
                        zip(rejections[idx], colorseq)):
                    ci_lo, ci_hi = rej.ci
                    yerr = np.vstack([rej.median - ci_lo, ci_hi - rej.median])

                    ax.errorbar(bin_center, rej.median,
                                xerr=bin_half_width,
                                yerr=yerr,
                               fmt="o", ms=0.25, color=colors[self.colors[idx]], label="Working point: {}".format(self.eff[idx]), zorder=z)

        if self.ylim:
            ax.set_ylim(self.ylim)
        else:
            y_lo, y_hi = ax.get_ylim()
            d = 0.05 * (y_hi - y_lo)
            ax.set_ylim(y_lo - d, y_hi + d)

        ax.set_xlabel(xvar_name.split("/")[-1], x=1, ha="right")
        ax.set_ylabel("Rejection {}".format(xvar_name), y=1, ha="right")
        ax.legend()
        plt.savefig("{}{}.png".format(save_dir, name))

        return fig
