from sklearn.neighbors import NearestNeighbors


class PointSelector:
    def __init__(self, n_points=5):
        self.n_points = n_points

    def get_point_sets(self, X):
        knn = NearestNeighbors(n_neighbors=self.n_points).fit()

        points_dict = {idx: (knn.kneighbours(point.reshape(-1, 1))[0], knn.kneighbours(point.reshape(-1, 1))[1]) for point in X}

        return points_dict
