import logging
from typing import Union

import numpy as np
import scipy.spatial.distance as distance
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors


class LoessRegressor(BaseEstimator, RegressorMixin):
    """
    Lo(w)ess regressor.

    """
    def __init__(self, weighting_method: Union[str, None] = None, span: float = .1):
        super().__init__()

        self.weighting_method = weighting_method
        self.span = span
        self.xs = None
        self.ys = None
        self.logger = logging.getLogger(__name__)

    def fit(self, xs: np.array, ys: np.array):
        """
        Fit the regressor on the data
        :param xs: Input data for the model
        :param ys: Target output data for the model
        :return: The fitted instance
        """
        self.xs = xs
        self.ys = ys

        return self

    def _get_window_indices(self, x: Union[float, np.array]):
        """
        Find and return the indices of the input data points that are closest to
        the given point x. The size of the returned set of indices is determined by the span setting.
        :param x: data point to find closest points to
        :return: List of indices of the data points closest to the requested point
        """
        n_points = int(len(self.xs) * self.span)

        knn = NearestNeighbors(n_neighbors=n_points).fit(self.xs.reshape(-1, 1))

        return knn.kneighbors(x)[1][0]

    def _create_weights(self, x: Union[float, np.array], xs: np.array):
        """

        :param x:
        :param xs:
        :return:
        """
        if self.weighting_method == 'euclidean':
            weights = np.array([distance.euclidean(x, xsi) for xsi in xs])
        else:
            weights = np.ones(xs.shape)

        weights = weights/weights.max()

        return weights

    def predict(self, xs: np.array, with_indices: bool = False):
        """

        :param xs:
        :param with_indices:
        :return:
        """
        y_pred = np.array([])

        if with_indices:
            indices = np.array([])

        for x in xs:
            idx_window = self._get_window_indices(x.reshape(-1, 1))

            if with_indices:
                indices = np.append(indices, idx_window)

            X = self.xs[idx_window].reshape(-1, 1)
            y = self.ys[idx_window]

            model = LinearRegression().fit(X, y, sample_weight=self._create_weights(x, X))

            y_pred = np.append(y_pred, model.predict(x.reshape(-1, 1)))

        if with_indices:
            return y_pred, indices
        else:
            return y_pred
