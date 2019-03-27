import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors


class LoessSmoother:
    def __init__(self,
                 model=LinearRegression(),
                 transform=True,
                 window_method='fixed',
                 fixed_window=2,
                 step_size=.2,
                 n_points=5):
        """
        Loess Regression. This class implements multiple methods of Loess regression.
        Options:
        - Select if the instance should behave as a transformer or as a model
        - Select if the window should be fixed size or always include a specified number of nearest
          datapoints.

        TODO:
        - Fix interpolation
        - iterative process of updating weights based on y-distance in fit method
        -
        :param model: Initialized sklearn object that has .fit() and .predict method
        :param transform: boolean, if Yes
        :param window_method:
        :param fixed_window:
        :param step_size:
        :param n_points:
        """
        self.model = model
        self.transform = transform
        self.window_method = window_method
        self.fixed_window = fixed_window
        self.step_size = step_size
        self.n_points = n_points

        # initialize empty objects
        self.x_focal_base = np.ndarray([])
        self.y_focal_base = np.ndarray([])
        self.indices = {}
        self.model_per_window = np.ndarray([])

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
            return np.sort(x)

        else:
            self.x_focal_base = np.array([np.min(x)])

            while self.x_focal_base.max() < np.array(x).max():
                self.x_focal_base = np.append(self.x_focal_base,
                                              self.x_focal_base[-1] + self.step_size)

    def _get_window_indices(self, x):
        """

        :param x:
        :return:
        """
        for index, x_focal in enumerate(self.x_focal_base):

            if self.window_method == 'fixed':
                x_indices = np.argwhere(
                    (x > (x_focal - self.fixed_window)) & (x < (x_focal + self.fixed_window)))

                if len(x_indices) > 2 * self.n_points:
                    self.indices[index] = x_indices
                else:  # If the number of returned indices is too small, resort to nearest points
                    self.indices[index] = x_indices

            if self.window_method == 'closest':
                x_focal = np.asarray([x_focal]).reshape(-1, 1)
                knn = NearestNeighbors(n_neighbors=self.n_points).fit(x.reshape(-1, 1))
                self.indices[index] = knn.kneighbors(x_focal)[1][0]

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

            # TODO: Development purposes: catch error during fit when x_window is empty
            try:
                # TODO: implement iterative weighted fit
                model = self.model.fit(x_window, y_window, sample_weight=None)
                np.append(self.y_focal_base, model.predict(x_focal))
                np.append(self.model_per_window, model)
            except ValueError:
                np.append(self.y_focal_base, np.asarray([np.mean(self.y_focal_base)]))
                np.append(self.model_per_window, None)


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
    Generate x with a gap and f(x) with added normal distributed noise with standard deviation noise_std, with:
    f(x) = 5*sin(x/3) + N(mu=0, sigma=n)

    :param noise_std: non-negative float, standard deviation of added noise.
    :return: xnp.Array, y = f(x) + N(0, std)
    """
    x1 = random_x(10, 30, 100)
    x2 = random_x(37, 60, 100)
    x = np.append(x1, x2)
    np.random.shuffle(x)

    return x, 5 * np.sin(x / 3) + np.random.normal(loc=0, scale=noise_std, size=x.shape)
