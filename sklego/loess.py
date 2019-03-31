import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression


class LoessSmoother:
    def __init__(self, model='linear', n_points=20, point_extraction='knn'):
        self.n_points = n_points
        # This is to be adjusted:
        self.model = LinearRegression()
        self.point_extraction = point_extraction

    def get_point_sets(self, X):
        if self.point_extraction=='knn':
            knn = NearestNeighbors(n_neighbors=self.n_points, metric='euclidean').fit(X)

            # For every point return n nearest points:
            for idx, point in enumerate(X):
                yield knn.kneighbors(point.reshape(-1, 1))[1][0]
        elif self.point_extraction=='symmetric':
            for idx, point in enumerate(X):
                if idx < int(self.n_points / 2):
                    yield np.array(range(0, idx + self.n_points))
                elif (idx >= int(self.n_points / 2)) & ((idx <= len(X) - int(self.n_points / 2))):
                    yield np.array(range(idx - int(self.n_points / 2), idx + int(self.n_points / 2)))
                else:
                    yield np.array(range(len(X) - int(self.n_points / 2), len(X)))

        else:
            raise NotImplementedError('Sorry this method has not been implemented.')

    def fit(self, X, y):

        if len(X.shape) < 2:
            X = X.reshape(-1, 1)

        points_generator = self.get_point_sets(X)
        y_focal = np.array([])

        for idx_focal, idx_window in enumerate(points_generator):
            x_focal = X[idx_focal].reshape(-1, 1)
            x_window = X[idx_window].reshape(-1, 1)
            y_window = y[idx_window]
            #pf = PolynomialFeatures(degree=1, include_bias=False)
            y_focal_new = self.model.fit(x_window, y_window).predict(x_focal.reshape(-1, 1))
            y_focal = np.append(y_focal, y_focal_new)
        return y_focal


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
