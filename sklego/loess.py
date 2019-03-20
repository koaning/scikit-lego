import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression


class LoessSmoother:
    def __init__(self, model='linear', window=20, step_size=0.2):
        self.window = window
        self.model = LinearRegression()
        self.n_points = 2
        self.step_size = step_size

    def generate_reg_indexes(self, X):
        """Return the focal points and the indices of the x values in their accompanying windows"""
        x_focal = X.min()
        x_max = X.max()

        while x_focal <= x_max:
            yield x_focal, np.argwhere((X > (x_focal - self.window)) & (X < (x_focal + self.window)))
            x_focal += self.step_size

    def get_closest_points(self, X, x_focal):
        """return the indices of the nearest points in dataset X to datapoint x_focus"""
        knn = NearestNeighbors(n_neighbors=self.n_points).fit(X.reshape(-1, 1))
        return knn.kneighbors(x_focal)[1][0]

    def fit(self, X, y):
        sorting_idx = numpy.argsort(X)
        X = X[sorting_idx]
        y = y[sorting_idx]
        points_generator = self.generate_reg_indexes(X)
        y_focal = np.array([])

        x_focal_base = np.array([])
        for x_focal, idx_window in points_generator:
            #             x_focal = X[idx_focal].reshape(-1, 1)
            x_focal = x_focal.reshape(-1, 1)
            x_window = X[idx_window].reshape(-1, 1)

            if len(idx_window) < 2:  # Not possible to fit on less than 2 points
                idx_window = self.get_closest_points(X, x_focal)
                x_window = X[idx_window].reshape(-1, 1)

            y_window = y[idx_window]

            y_focal_new = self.model.fit(x_window, y_window).predict(x_focal)

            y_focal = np.append(y_focal, y_focal_new)
            x_focal_base = np.append(x_focal_base, x_focal)
        return y_focal, x_focal_base


def random_x(minimum_val, maximum_val, size):
    """
    Generate n random data-points of size between minimum_val and right bounds
    :param minimum_val: float, minimal value of the generated data
    :param maximum_val: float, maximum value of the generated data
    :param size: tuple or list, shape of the desired output
    :return: np.Array with generated data
    """
    return (maximum_val - minimum_val) * np.random.random(size=size) + minimum_val


def generate_data(x, generate_function, noise_std):
    """
    Return f(x) with added normal distributed noise with standard deviation noise_std.
    :param x: np.Array, input data
    :param generate_function: function to apply to input data
    :param noise_std: non-negative float, standard deviation of added noise.
    :return: np.Array, y = f(x) + N(0, std)
    """
    return generate_function(x) + np.random.normal(loc=0, scale=noise_std, size=x.shape)
