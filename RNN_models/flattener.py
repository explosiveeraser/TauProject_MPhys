#altered from ATLAS tau ID code

import numpy as np
from scipy.stats import binned_statistic, binned_statistic_2d


class Flattener:
    """
    Efficieny flattener.
    """
    def __init__(self, x_bins, y_bins, eff):
        self.x_bins = x_bins
        self.y_bins = y_bins
        self.eff = eff

        self.cutmap = None

    def fit(self, x, y, values):
        """
        Fits the flattener.

        Returns:
        --------
        passes_thr : (N,) array of bools
            Array indicating which of the inputs pass the working point.
        """
       # x = np.asarray(x).astype(np.float32)
       # y = np.asarray(y).astype(np.float32)
        values = values[:,0]
        print("len x: {} | y: {} | values: {}".format(np.shape(x), np.shape(y), np.shape(values)))
        statistic, _, _, binnumber = binned_statistic_2d(
            x, y, values,
            statistic=lambda arr: np.percentile(arr, 100 * (1 - self.eff)),
            bins=[self.x_bins, self.y_bins], expand_binnumbers=True
        )

        #print(statistic)

        self.cutmap = statistic
        x_idx, y_idx = binnumber[0, :] - 1, binnumber[1, :] - 1



        return values > self.cutmap[x_idx, y_idx]


    def passes_thr(self, x, y, values):
        """
        Checks which entries pass the working point.

        Returns:
        --------
        passes_thr : (N,) array of bools
            Array indicating which of the inputs pass the working point.
        """
        values = values[:,0]
        if self.cutmap is None:
            return None

        _, _, _, binnumber = binned_statistic_2d(
            x, y, values,
            statistic="count",
            bins=[self.x_bins, self.y_bins],
            expand_binnumbers=True
        )

        x_idx, y_idx = binnumber[0, :] - 1, binnumber[1, :] - 1

        return values > self.cutmap[x_idx, y_idx]

