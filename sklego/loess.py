import logging
from math import floor

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


class LoessSmoother:
    def __init__(self,
                 model=None,
                 n_degree=2,
                 transform=True,
                 window_method='fixed',
                 window_size=2,
                 step_size=.2,
                 fraction=.1):
        """
        Loess Regression. This class implements multiple methods of Loess regression.
        Options:
        - Select if the instance should behave as a transformer or as a model
        - Select if the window should be fixed size or always include a specified number of nearest
          datapoints.

        TODO:
        - Add typehints
        - Improve interpolation for 'transform == False' estimator mode
        - Include weighted fit
        - Generalize to multi-input
        """

        self.logger = logging.getLogger(__name__)

        self.model = self._init_model(n_degree, model)
        self.transform = transform
        self.window_method = self._init_window_method(window_method)

        self.window_size = window_size
        self.step_size = step_size
        self.fraction = fraction

        # initialize fit attributes
        self.x_focal_base = np.ndarray([])
        self.y_focal_base = np.ndarray([])
        self.indices = {}
        self.model_per_window = np.ndarray([])

    def _init_model(self, n_degree, model):
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

        :param x:
        :param y:
        :return:
        """
        self._get_x_focal_base(x)
        self._get_window_indices(x)
        self._fit_model_per_window(x, y)

    def _get_x_focal_base(self, x):
        """

        :param x:
        :return:
        """
        if self.transform:
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

        :param x:
        :return:
        """
        # TODO: Create generators instead of self.indices list.
        x_length = len(x)
        n_points = x_length*self.fraction

        for index, x_focal in enumerate(self.x_focal_base):
            self.logger.info(f"Creating data windows using method: {self.window_method}")

            if self.window_method == 'fixed':
                self._get_fixed_window_indices(self, index, x_focal, x, n_points)

            elif self.window_method == 'knn':
                self._get_knn_window_indices(self, index, x_focal, x, n_points)

            elif self.window_method == 'knn_symmetric':
                self._get_knn_symmetric_indices(self, index, x_length, n_points)

    def _get_fixed_window_indices(self, index, x_focal, x, n_points):
        x_indices = np.argwhere(
            (x > (x_focal - self.window_size)) & (x < (x_focal + self.window_size)))

        # TODO: Improve interpolation
        if len(x_indices) > 2 * n_points:
            self.indices[index] = x_indices
        else:  # If the number of returned indices is too small, resort to nearest points
            self._get_knn_window_indices(self, index, x_focal, x, n_points)

    def _get_knn_window_indices(self, index, x_focal, x, n_points):
        x_focal = np.asarray([x_focal]).reshape(-1, 1)
        knn = NearestNeighbors(n_neighbors=int(n_points)).fit(x.reshape(-1, 1))
        self.indices[index] = knn.kneighbors(x_focal)[1][0]

    def _get_knn_symmetric_indices(self, index, x_length, n_points):
        if index < floor(n_points / 2):
            self.indices[index] = np.array(range(0, index + floor(n_points / 2)))

        elif (index >= floor(n_points / 2)) & (
                (index <= x_length - floor(n_points / 2))):
            self.indices[index] = np.array(range(index - floor(n_points / 2),
                                                 index + floor(n_points / 2))
                                           )
        else:
            self.indices[index] = np.array(range(x_length - floor(n_points / 2), x_length))

    def _fit_model_per_window(self, x, y):
        """

        :param x:
        :param y:
        :return:
        """
        for index in self.indices.keys():
            x_focal = np.asarray([self.x_focal_base[index]]).reshape(-1, 1)
            x_window = x[self.indices[index]].reshape(-1, 1)
            y_window = y[self.indices[index]].reshape(-1, 1)

            # TODO: implement iterative weighted fit
            model = self.model.fit(x_window, y_window)
            self.y_focal_base = np.append(self.y_focal_base, model.predict(x_focal))
            self.model_per_window = np.append(self.model_per_window, model)


def plot_windows(x, y, x_focal_base, y_focal_base, indices):
    """
    TODO: Use matplotlib.animation to create video

    :param x:
    :param y:
    :param x_focal_base:
    :param y_focal_base:
    :param indices:
    :return:
    """
    for index in indices.keys():
        ax = plt.axes(xlim=(np.min(x), np.max(x)), ylim=(np.min(y), np.max(y)))
        ax.scatter(x[indices[index]], y[indices[index]])
        ax.scatter(x_focal_base[index], y_focal_base[index], c='r')
        ax.set_title(f'Window for {x_focal_base[index]}')
        plt.pause(0.01)
        plt.show()


# HELPER FUNCTIONS TO GENERATE RANDOM DATA
def random_x(minimum_val, maximum_val, size):
    """
    Generate n random data-points of size between minimum_val and right bounds
    :param minimum_val: float, minimal value of the generated data
    :param maximum_val: float, maximum value of the generated data
    :param size: tuple or list, shape of the desired output
    :return: x, y
    """
    return (maximum_val - minimum_val) * np.random.random(size=size) + minimum_val


def generate_noisy_sine_data(noise_std):
    """
    Generate x with a gap and f(x) with added normal distributed noise with standard deviation
    noise_std, with:
    f(x) = 5*sin(x/3) + N(mu=0, sigma=n)

    :param noise_std: non-negative float, standard deviation of added noise.
    :return: xnp.Array, y = f(x) + N(0, std)
    """
    x1 = random_x(10, 30, 100)
    x2 = random_x(37, 60, 100)
    x = np.append(x1, x2)
    np.random.shuffle(x)

    return x, 5 * np.sin(x / 3) + np.random.normal(loc=0, scale=noise_std, size=x.shape)
