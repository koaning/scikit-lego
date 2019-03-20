import numpy as np
from sklearn.neighbors import NearestNeighbors


class PointSelector:
    def __init__(self, n_points=5, metric='euclidean'):
        self.n_points = n_points
        self.metric = metric

    def get_point_indexes(self, X):
        knn = NearestNeighbors(n_neighbors=self.n_points, metric=self.metric).fit()

        for point in X:
            yield knn.kneighbours(point.reshape(-1, 1))[1][0]


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
