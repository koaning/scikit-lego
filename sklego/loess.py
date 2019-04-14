import logging
from typing import Union

import numpy as np
import scipy.spatial.distance as distance
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_X_y
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.utils.validation import check_is_fitted


class LoessRegressor(BaseEstimator, RegressorMixin):
    """
    Lo(w)ess regressor.

    """

    def __init__(self, weighting_method: str = 'equal', span: float = .1):
        super().__init__()

        self._check_init_inputs(weighting_method, span)

        self.weighting_method = weighting_method
        self.span = span
        self.xs = None
        self.ys = None
        self.dim_ = None
        self.logger = logging.getLogger(__name__)

    def fit(self, X: np.array, y: np.array):
        """
        Fit the regressor on the data by storing the data sorted on the inputs.

        :param X: Input data for the model
        :param y: Target output data for the model
        :return: The fitted instance
        """

        self.xs, self.ys = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)
        self.dim_ = self.xs.shape[1]

        return self

    def predict(self, X: np.array, with_indices: bool = False) -> Union[np.array, np.array]:
        """

        :param X:
        :param with_indices:
        :return:
        """
        check_is_fitted(self, ['dim_'])
        X = check_array(X)

        y_pred = np.array([])

        if with_indices:
            indices = []

        for x in X:
            idx_window = self._get_window_indices(x.reshape(-1, 1))

            if with_indices:
                indices.append(idx_window)

            x = x.reshape(-1, 1)
            idx_window = self._get_window_indices(x)

            X = self.xs[idx_window].reshape(-1, 1)
            y = self.ys[idx_window]

            model = LinearRegression().fit(X, y, sample_weight=self._create_weights(x, X))

            y_pred = np.append(y_pred, model.predict(x.reshape(-1, 1)))

        if with_indices:
            return y_pred, indices
        else:
            return y_pred

    @staticmethod
    def _check_init_inputs(weighting_method, span):
        if not 0 < span <= 1:
            raise ValueError("Span should be larger than 0 and smaller or equal to 1")

        expected_weighting_methods = ['euclidean', 'equal']

        if weighting_method not in expected_weighting_methods:
            raise ValueError(f"Received unexpected weighting method. "
                             f"Choose one from: {expected_weighting_methods}. "
                             f"If no weighting method is provided, default 'equal' is used.")

    def _get_window_indices(self, x: Union[float, np.array]):
        """
        Find and return the indices of the input data points that are closest to
        the given point x. The size of the returned set of indices is determined
        by the span setting.
        :param x: data point to find closest points to
        :return: List of indices of the data points closest to the requested point
        """
        n_points = int(len(self.xs) * self.span)

        knn = NearestNeighbors(n_neighbors=n_points).fit(self.xs.reshape(-1, 1))

        return knn.kneighbors(x)[1][0]

    def _create_weights(self, x: Union[float, np.array], xs: np.array) -> np.array:
        """
        Create an array that serves as a weight mask for the regressor.
        :param x:
        :param xs:
        :return:
        """
        if self.weighting_method == 'euclidean':
            distances = np.array([distance.euclidean(x, xsi) for xsi in xs])
            distances = np.where(distances == 0, 0.1 * distances[distances != 0].min(), distances)
            weights = 1 / distances
        else:
            weights = np.ones(xs.shape)

        return weights.flatten()
