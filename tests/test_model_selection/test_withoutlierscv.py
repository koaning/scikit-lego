import numpy as np
import pytest
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, StratifiedKFold

from sklego.model_selection import WithoutLiersCV


@pytest.mark.parametrize(
    "cv_strategy", [KFold(2), KFold(3, shuffle=True), StratifiedKFold(5), GroupKFold(2), GroupShuffleSplit(3)]
)
@pytest.mark.parametrize("anomalous_label", [-1, 1])
def test_split_without_anomalies(cv_strategy, anomalous_label):

    size = 1000

    X = np.random.randn(size, 3)
    y = (np.random.randn(size) > 1.5).astype(int)
    groups = np.random.randint(0, 10, size)

    y[y == 1] = anomalous_label

    cv = WithoutLiersCV(cv_strategy, anomalous_label=anomalous_label)

    for inliner_index, test_index in cv.split(X, y, groups):
        y_train = y[inliner_index]
        assert np.all(y_train != anomalous_label)

    assert cv.get_n_splits(X, y, groups) == cv_strategy.get_n_splits(X, y, groups)
