import pytest
from sklego.model_selection import KMeansFold
from tests.conftest import id_func


@pytest.mark.parametrize("n_init", [5, 10], ids=id_func)
@pytest.mark.parametrize("algorithm", ['full', 'elkan'], ids=id_func)
@pytest.mark.parametrize("splits", [3, 5], ids=id_func)
def test_splits(n_init, algorithm, splits, random_xy_dataset_regr):
    X, y = random_xy_dataset_regr
    k_means_kwargs = {'n_init': n_init, 'algorithm': algorithm}
    kf = KMeansFold(n_splits=splits, random_state=123, k_means_kwargs=k_means_kwargs)

    for train_index, test_index in kf.split(X):
        assert len(train_index) > 0
        assert len(test_index) > 0
