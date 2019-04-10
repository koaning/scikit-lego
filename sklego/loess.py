from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
import scipy.spatial.distance as distance


class LoessRegressor(BaseEstimator, RegressorMixin):
    def __init__(self,
                 weighting_method=None,
                 span=.1):
        super().__init__()

        self.weighting_method = weighting_method
        self.span = span
        self.xs = None
        self.ys = None

    def fit(self, xs, ys):
        """

        :param x:
        :param y:
        :return:
        """
        self.xs = xs
        self.ys = ys

        return self

    def _get_window_indices(self, x):
        """

        :param x:
        :return:
        """
        n_points = int(len(self.xs) * self.span)

        knn = NearestNeighbors(n_neighbors=n_points).fit(self.xs.reshape(-1, 1))

        return knn.kneighbors(x)[1][0]

    def _create_weights(self, x, xs):

        if self.weighting_method == None:
            weights = np.ones(xs.shape)
        elif self.weighting_method == 'euclidean':
            weights = np.array([distance.euclidean(x, xsi) for xsi in xs])

        weights = weights/weights.max()

        return weights

    def predict(self, xs):
        """

        :param x:
        :param y:
        :return:
        """
        y_pred = np.array([])

        for x in xs:
            idx_list = self._get_window_indices(x.reshape(-1, 1))

            X = self.xs[idx_list].reshape(-1, 1)
            y = self.ys[idx_list]

            model = LinearRegression().fit(X, y, sample_weight=self._create_weights(self, x, X))

            y_pred = np.append(y_pred, model.predict(x.reshape(-1, 1)))

        return y_pred
