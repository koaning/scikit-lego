import pytest
from sklearn import clone
from sklearn.base import BaseEstimator

from sklego.model_selection import KlusterFoldValidation
from sklearn.cluster import KMeans, MiniBatchKMeans
from tests.conftest import id_func
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

k_means_pipeline = make_pipeline(StandardScaler(), KMeans())


class DummyCluster(BaseEstimator):
    def __init__(self, n_splits=3):
        self.n_splits = n_splits

    def fit(self, X):
        return self

    def predict(self, X):
        return np.random.randint(0, self.n_splits, size=X.shape[0])

    def fit_predict(self, X):
        return self.predict(X)


@pytest.mark.parametrize(
    "cluster_method",
    [KMeans(), MiniBatchKMeans(), DummyCluster(), k_means_pipeline],
    ids=id_func,
)
def test_splits_not_fitted(cluster_method, random_xy_dataset_regr):
    cluster_method = clone(cluster_method)
    X, y = random_xy_dataset_regr
    kf = KlusterFoldValidation(cluster_method=cluster_method)
    for train_index, test_index in kf.split(X):
        assert len(train_index) > 0
        assert len(test_index) > 0


@pytest.mark.parametrize(
    "cluster_method",
    [KMeans(), MiniBatchKMeans(), DummyCluster(), k_means_pipeline],
    ids=id_func,
)
def test_splits_fitted(cluster_method, random_xy_dataset_regr):
    cluster_method = clone(cluster_method)
    X, y = random_xy_dataset_regr
    cluster_method = cluster_method.fit(X)
    kf = KlusterFoldValidation(cluster_method=cluster_method)
    for train_index, test_index in kf.split(X):
        assert len(train_index) > 0
        assert len(test_index) > 0


def test_no_split(random_xy_dataset_regr):
    X, y = random_xy_dataset_regr
    # With only one split, the method should raise a ValueError
    cluster_method = DummyCluster(n_splits=1)
    kf = KlusterFoldValidation(cluster_method=cluster_method)
    with pytest.raises(ValueError):
        for train_index, test_index in kf.split(X):
            assert len(train_index) > 0
            assert len(test_index) > 0
