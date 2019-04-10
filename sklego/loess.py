import logging
from math import floor

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


class LoessSmoother(TransformerMixin, BaseEstimator):
    def __init__(self,
                 model=None,
                 n_degree=2,
                 transformer_type=True,
                 window_method='fixed',
                 window_size=2,
                 step_size=.2,
                 fraction=.1):

        """
        Loess Regression. This class implements multiple methods of Loess
        regression for the single input, single output case. The first main
        choice to make is if the instance should behave as a transformer or
        as an estimator.

        In transformer mode, the instance will take each data_point and
        return it's y value changed according to the Loess regression. Hence,
        no interpolation is done in between missing datapoints.

        In estimator mode, the instance will create a x_focal_base to evalue
        the y values for. The created x_focal_base is an equally spaced
        distance vector between x_min and x_man with distance step_size
        between elements. Hence, interpolation will happen because of the
        fixed step sizes.

        The second main choice to make is the windowing method. This
        determines how the algorithm selects datapoints for each x_focal
        point to fit a model. Currently three options exist:

        - fixed: the window is created using a fixed distance from the
        x_focal point in both + and - direction. If this window doesn't yield
        sufficient datapoints, it falls back to the 'knn_symmetric' window
        method.
        - knn: The window is created by taking the n_closest datapoints,
        determined by the fraction parameter. It doesn't matter if the points
        are in the + or - direction of x_focal
        - knn_symmetric: The window is created by taking n_closest/2
        datapoints in the + and - direction from the x_focal point.

        See http://web.ipac.caltech.edu/staff/fmasci/home/astro_refs/LocalRegressionBook_1999.pdf
        for theoretical background and more information on
        Loess regression.

        TODO:
        - Clean knn_symmetric code
        - Add tests
        - Fix sklearn interface
        - Decide on how to use predict outside of original x_focal_base.
        - Add function typehints
        - Include weighted fit
        - Generalize to multi-input
        - Iso hand-tune window and step parameters, perform grid_search,
        store all results and sample from results. Use distribution to
        get 'line-prediction' and confidence bound around it.

        :param model: Sklearn estimator. Optional to include a sklearn
            pipeline or model to use
        :param n_degree: Integer, Select n-th order polynomial for fitting
        :param transformer_type: Bool, True for instantiating a Transformer,
            False for an estimator
        :param window_method: String, Select one of the available methods for
            windowing
        :param window_size: Integer, Determine the window size for the 'fixed'
            window method
        :param step_size: Float, Determine the step size for the x_focal_base
            if instantiated as an estimator
        :param fraction: Float, value between 0 and 1, determines the fraction
            of datapoints to use as window for the 'knn' and 'knn_symmetric'
            window method.
        """

        self.logger = logging.getLogger(__name__)

        self.model = self._init_model(n_degree, model)
        self.transformer_type = transformer_type
        self.window_method = self._init_window_method(window_method)

        self.window_size = window_size
        self.step_size = step_size
        self.fraction = fraction

        # initialize attributes
        self.x_focal_base = np.array([])
        self.y_focal_base = np.array([])
        self.indices = {}
        self.model_per_window = np.array([])

    def _init_model(self, n_degree, model):
        """
        Initialize the model that will be used for local model fitting. If n_degree is defined,
        a PolynomalFeatures with degree=n_degree with LinearRegression sklearn pipeline is used.

        The user can also define it's own sklearn pipeline and pass this as the model argument.
        :param n_degree: Integer, degree for the PolynomialFeatures preprocessor
        :param model: sklearn-like model or pipeline
        """

        # TODO: Use argparse mutually_exclusive_group iso if statement?
        if n_degree and model is None:
            self.logger.info(f"Creating linear model with polynomial features of degree {model}.\n"
                             f"NOTE: interaction terms are also included!")
            return Pipeline([('p_features', PolynomialFeatures(degree=n_degree,
                                                               include_bias=True,
                                                               interaction_only=False)),
                             ('linear', LinearRegression(fit_intercept=True,
                                                         normalize=False,
                                                         copy_X=True,
                                                         n_jobs=2))
                            ])

        elif not n_degree and model:
            self.logger.info(f"Using user defined model: {model}")
            return model
        else:
            message = f"Parameters n_degree and model are required mutually exclusive. Choose one."
            self.logger.error(message)
            # TODO: Use better Exception?
            raise ValueError(message)

    def _init_window_method(self, window_method):
        """
        Initialize windowing method. See class main docstring for more information.
        :param window_method: String, method identifier
        """

        available_window_methods = ['knn', 'knn_symmetric', 'fixed']
        if window_method in available_window_methods:
            return window_method
        else:
            message = (f"Unrecognized window method. Use one of the following:"
                       f"\n{available_window_methods}")
            self.logger.error(message)
            raise NotImplementedError(message)

    def fit(self, x, y):
        """
        Fit the Loess Regression to the data.
        :param x:
        :param y:
        """

        self._get_x_focal_base(x)
        self._get_window_indices(x)
        self._fit_model_per_window(x, y)

        return self

    def transform(self, x):
        """
        Return the transformed y-values of the data.
        :param x:
        """

        if self.transformer_type:
            return self.y_focal_base
        else:
            message = f"Instance not instantiated as transformer. Use predict method."
            self.logger.error(message)
            raise RuntimeError(message)

    def predict(self, x):
        """
        Predict for new input values. Not yet implemented
        :param x:
        """

        if not self.transformer_type:
            raise NotImplemented("Not yet implemented")
        else:
            message = f"Instance instantiated as transformer. Use transform method."
            self.logger.error(message)
            raise RuntimeError(message)

    def _get_x_focal_base(self, x):
        """
        Define the x_focal_base. Each x_focal point will be used as a point of reference to create
        a window on and fit a local model. In transformer mode, this is simply the actual data x
        values. In estimator mode, a base is created from the min and max of the actual data.
        :param x: np.ndarray like. Input data set
        :return:
        """

        if self.transformer_type:
            self.x_focal_base = np.sort(x)
            self.logger.info('Creating x basis as a transformer')

        else:
            self.logger.info('Creating x basis as an estimator')
            self.x_focal_base = np.array([np.min(x)])

            while self.x_focal_base.max() < np.array(x).max():
                self.x_focal_base = np.append(self.x_focal_base,
                                              self.x_focal_base[-1] + self.step_size)

    def _get_window_indices(self, x):
        """
        For each x_focal point in x_focal_base, create the window for model fitting.
        :param x: np.ndarray like. Input data set
        """

        # TODO: Create generators instead of self.indices list.

        n_points = len(x)*self.fraction

        for index, x_focal in enumerate(self.x_focal_base):
            self.logger.info(f"Creating data windows using method: {self.window_method}")

            if self.window_method == 'fixed':
                self._get_fixed_window_indices(index, x_focal, x, self.window_size)

            elif self.window_method == 'knn':
                self._get_knn_window_indices(index, x_focal, x, n_points)

            elif self.window_method == 'knn_symmetric':
                self._get_knn_symmetric_indices(index, x_focal, x, n_points)

    def _get_fixed_window_indices(self, index, x_focal, x, n_points):
        x_indices = np.argwhere(
            (x > (x_focal - self.window_size)) & (x < (x_focal + self.window_size)))

        # TODO: Improve interpolation
        if len(x_indices) > 2 * n_points:
            self.indices[index] = x_indices
        else:  # If the number of returned indices is too small, resort to knn_symmetric
            self._get_knn_symmetric_indices(index, x_focal, x, n_points)

    def _get_knn_window_indices(self, index, x_focal, x, n_points):
        x_focal = np.asarray([x_focal]).reshape(-1, 1)

        knn = NearestNeighbors(n_neighbors=int(n_points)).fit(x.reshape(-1, 1))
        self.indices[index] = knn.kneighbors(x_focal)[1][0]

    def _get_knn_symmetric_indices(self, index, x_focal, x, n_points):

        low_mask = x < x_focal
        high_mask = ~low_mask

        x_low = x[low_mask]
        x_high = x[high_mask]

        x_focal = np.asarray([x_focal]).reshape(-1, 1)

        x_low_indices = np.where(low_mask)[0]
        x_high_indices = np.where(high_mask)[0]

        if (x_low.size > floor(n_points / 2)) & (x_high.size > floor(n_points / 2)):
            knn_low = NearestNeighbors(n_neighbors=int(n_points / 2)).fit(x_low.reshape(-1, 1))
            knn_high = NearestNeighbors(n_neighbors=int(n_points / 2)).fit(x_high.reshape(-1, 1))
            indices_low_local = knn_low.kneighbors(x_focal)[1][0]
            indices_high_local = knn_high.kneighbors(x_focal)[1][0]

            indices_low = x_low_indices[indices_low_local]
            indices_high = x_high_indices[indices_high_local]
            self.indices[index] = np.concatenate([indices_low, indices_high])

        elif (x_low.size > floor(n_points / 2)) & (not x_high.size > floor(n_points / 2)):
            knn_low = NearestNeighbors(n_neighbors=int(n_points)).fit(x_low.reshape(-1, 1))
            indices_low_local = knn_low.kneighbors(x_focal)[1][0]
            self.indices[index] = x_low_indices[indices_low_local]

        else:
            knn_high = NearestNeighbors(n_neighbors=int(n_points)).fit(x_high.reshape(-1, 1))
            indices_high_local = knn_high.kneighbors(x_focal)[1][0]
            self.indices[index] = x_high_indices[indices_high_local]

    def _fit_model_per_window(self, x, y):
        """
        With the created x focal points and accompanying windows, fit per x focal point a model on
        the local window and predict an y focal using the fitted model.
        :param x:
        :param y:
        """
        for index in self.indices.keys():
            x_focal = np.asarray([self.x_focal_base[index]]).reshape(-1, 1)
            x_window = x[self.indices[index]].reshape(-1, 1)
            y_window = y[self.indices[index]].reshape(-1, 1)

            # TODO: implement iterative weighted fit
            model = self.model.fit(x_window, y_window)
            self.y_focal_base = np.append(self.y_focal_base, model.predict(x_focal))
            self.model_per_window = np.append(self.model_per_window, model)
