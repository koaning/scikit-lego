from sklearn.neighbors import NearestNeighbors


class PointSelector:
    def __init__(self, n_points=5, metric='euclidean'):
        self.n_points = n_points
        self.metric = metric

    def get_point_indexes(self, X):
        knn = NearestNeighbors(n_neighbors=self.n_points, metric=self.metric).fit()

        for point in X:
            yield knn.kneighbours(point.reshape(-1, 1))[1][0]
