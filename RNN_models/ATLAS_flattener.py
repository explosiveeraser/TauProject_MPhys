import numpy as np
from scipy.stats import binned_statistic_2d, binned_statistic


class Flattener:
    """
    Efficieny flattener.
    """
    def __init__(self, x_bins, eff):
        self.x_bins = x_bins
        #self.y_bins = y_bins
        self.eff = eff

        self.cutmap = None

    def fit(self, x, values):
        """
        Fits the flattener.

        Returns:
        --------
        passes_thr : (N,) array of bools
            Array indicating which of the inputs pass the working point.
        """
        values = values.flatten()
        print("shape of x: {}".format(np.shape(x)))
        print("shape of values: {}".format(np.shape(values)))
        statistic, _, binnumber = binned_statistic(
            x, values,
            statistic=lambda arr: np.percentile(arr, 100 * (1 - self.eff)),
            bins=self.x_bins
        )

        self.cutmap = statistic
        x_idx = binnumber[:] - 1

        print("cutmap: {}".format(self.cutmap))
        print("x_idx {}".format(x_idx))

        return values > self.cutmap[x_idx]

    def passes_thr(self, x, values):
        """
        Checks which entries pass the working point.

        Returns:
        --------
        passes_thr : (N,) array of bools
            Array indicating which of the inputs pass the working point.
        """
        if self.cutmap is None:
            return None

        _, _, binnumber = binned_statistic(
            x, values,
            statistic="count",
            bins=self.x_bins
        )

        x_idx= binnumber[:] - 1

        return values > self.cutmap[x_idx]
