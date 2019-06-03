import logging
from typing import Union

import autograd.numpy as np
import numpy as np
import scipy.spatial.distance as distance
from autograd import grad
from autograd.test_util import check_grads
from scipy.spatial import KDTree
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_X_y
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.utils.validation import check_is_fitted


class DeadZoneRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, threshold=0.3, relative=False, effect="linear", n_iter=2000, stepsize=0.01, check_grad=False):
        self.threshold = threshold
        self.relative = relative
        self.effect = effect
        self.n_iter = n_iter
        self.stepsize = stepsize
        self.check_grad = check_grad
        self.allowed_effects = ("linear", "quadratic", "constant")
        self.loss_log_ = None
        self.wts_log_ = None
        self.deriv_log_ = None
        self.coefs_ = None

    def fit(self, X, y):
        X, y = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)
        if self.effect not in self.allowed_effects:
            raise ValueError(f"effect {self.effect} must be in {self.allowed_effects}")

        def deadzone(errors):
            if self.effect == "linear":
                return np.where(errors > self.threshold, errors, np.zeros(errors.shape))
            if self.effect == "quadratic":
                return np.where(errors > self.threshold, errors**2, np.zeros(errors.shape))

        def training_loss(weights):
            diff = np.abs(np.dot(X, weights) - y)
            if self.relative:
                diff = diff / y
            return np.mean(deadzone(diff))

        n, k = X.shape

        # Build a function that returns gradients of training loss using autograd.
        training_gradient_fun = grad(training_loss)

        # Check the gradients numerically, just to be safe.
        weights = np.random.normal(0, 1, k)
        if self.check_grad:
            check_grads(training_loss, modes=['rev'])(weights)

        # Optimize weights using gradient descent.
        self.loss_log_ = np.zeros(self.n_iter)
        self.wts_log_ = np.zeros((self.n_iter, k))
        self.deriv_log_ = np.zeros((self.n_iter, k))
        for i in range(self.n_iter):
            weights -= training_gradient_fun(weights) * self.stepsize
            self.wts_log_[i, :] = weights.ravel()
            self.loss_log_[i] = training_loss(weights)
            self.deriv_log_[i, :] = training_gradient_fun(weights).ravel()
        self.coefs_ = weights
        return self

    def predict(self, X):
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        check_is_fitted(self, ['coefs_'])
        return np.dot(X, self.coefs_)


class LoessRegressor(BaseEstimator, RegressorMixin):
    """
    Lo(w)ess regressor.

    See for example: https://www.itl.nist.gov/div898/handbook/pmd/section1/pmd144.htm

    """

    def __init__(self, weighting_method: Union[str, None] = 'unweighted', span: float = .1):

        self.available_weighting_methods = ['euclidean', 'unweighted']

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

        :param X: Array-like object of shape (n_samples, 1), input data for the model.
        :param y: Array-like object of shape (n_samples, 1), target output data for the model
        :return: The fitted instance
        """

        self.xs, self.ys = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)
        self.dim_ = self.xs.shape[1]

        return self

    def predict(self, X: np.array, with_indices: bool = False) -> Union[np.array, tuple]:
        """
        Predict targets using the fitted model
        :param X: Array-like object of shape (n_samples, 1), input data for the model.
        :param with_indices: If set to True, function also returns the indices of the sub-set in the
            train data used for fitting the local model.
        :return: Array-like object of shape (n_samples, 1), predicted target output. If with_indices
            is set to true, a tuple of two array-like objects (y_predictions, indices) is returned
            instead.
        """

        check_is_fitted(self, ['dim_'])
        X = check_array(X)
        y_pred = np.zeros(shape=(X.shape[0], 1))

        if with_indices:
            indices = [[] for _ in np.arange(X.shape[0])]

        for index, x in enumerate(X):
            idx_window = self._get_window_indices(x.reshape(-1, 1))
            if with_indices:
                indices[index] = idx_window

            X = self.xs[idx_window].reshape(-1, 1)
            y = self.ys[idx_window]

            weights = self._create_weights(x, X)
            model = LinearRegression().fit(X, y, sample_weight=weights)
            y_pred[index] = model.predict(x.reshape(-1, 1))

        if with_indices:
            return y_pred, indices
        else:
            return y_pred

    def _check_init_inputs(self, weighting_method: str, span: float) -> None:
        """
        Checks if the provided model parameters are valid.

        :param weighting_method: String id of the weighting method to be used. Should be 'euclidean'
            or 'equal'.
        :param span: Float in (0,1]. Sets the fraction of data to use for making local predictions.
            If 1, the full training data set is used.
        """

        if not 0 < span <= 1:
            raise ValueError("Span should be larger than 0 and smaller or equal to 1")

        if weighting_method not in self.available_weighting_methods:
            raise ValueError(f"Received unexpected weighting method. "
                             f"Choose one from: {self.available_weighting_methods}. "
                             f"If no weighting method is provided, default 'equal' is used.")

    def _get_window_indices(self, x: Union[float, np.array]) -> np.array:
        """
        Find and return the indices of the input data points that are closest to
        the given point x. The size of the returned set of indices is determined
        by the span setting.

        :param x: data point to find closest points to
        :return: List of indices of the data points closest to the requested point
        """

        n_points = int(len(self.xs) * self.span)
        lookup_closest = KDTree(self.xs)
        return lookup_closest.query(x=x, k=n_points)[1].flatten()

    def _create_weights(self, x: Union[float, np.array], xs: np.array) -> Union[np.array, None]:
        """
        Create an array that serves as a weight mask for the regressor.

        :param x: Data point of reference to calculate weights for
        :param xs: full data set to calculate the weights with respect to the data point of
            reference
        :return: Array of weight values using the specified method, None if no method is specified.
            When no method is specified, the regressor will resort to an approach using equal
            weights.
        """

        if self.weighting_method == 'euclidean':
            distances = np.array([distance.euclidean(x, xsi) for xsi in xs])
            return (1 - distances / distances.max()).ravel()

        return None
