import pytest
from sklego.model_selection import KlusterFoldValidation
from sklearn.cluster import KMeans, MiniBatchKMeans
from tests.conftest import id_func
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

k_means_pipeline = make_pipeline(StandardScaler(), KMeans())


class DummyCluster:
    def fit(self):
        return self

    def predict(self, X):
        return np.random.randint(0, 3, size=X.shape[0])

    def fit_predict(self, X):
        return self.predict(X)


@pytest.mark.parametrize("cluster_method", [KMeans(),
                                            MiniBatchKMeans(),
                                            DummyCluster(),
                                            k_means_pipeline], ids=id_func)
def test_splits(cluster_method, random_xy_dataset_regr):
    X, y = random_xy_dataset_regr
    kf = KlusterFoldValidation(cluster_method=cluster_method, random_state=123)

    for train_index, test_index in kf.split(X):
        assert len(train_index) > 0
        assert len(test_index) > 0
