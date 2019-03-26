import pytest
from sklego.model_selection import KMeansFold
from tests.conftest import id_func


@pytest.mark.parametrize("folder", [
    KMeansFold,
], ids=id_func)
@pytest.mark.parametrize("splits", [2, 5], ids=id_func)
def test_splits(folder, splits, random_xy_dataset_regr):
    X, y = random_xy_dataset_regr
    kf = folder(n_splits=splits, random_state=123)
    assert kf.get_n_splits(X) == splits
