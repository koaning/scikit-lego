import numpy as np
import pytest
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, StratifiedKFold
from sklearn.model_selection._split import _BaseKFold

from sklego.model_selection import WithoutLiersCV


@pytest.mark.parametrize(
    "cv_strategy", [KFold(2), KFold(3, shuffle=True), StratifiedKFold(5), GroupKFold(2), GroupShuffleSplit(3)]
)
@pytest.mark.parametrize("anomalous_label", [-1, 1])
def test_split_without_anomalies(cv_strategy: _BaseKFold, anomalous_label: int) -> None:
    rng = np.random.default_rng()
    size = 1000

    X = rng.normal(size=(size, 3))
    y = (rng.normal(size=size) > 1.5).astype(int)
    groups = rng.integers(low=0, high=10, size=size)

    y[y == 1] = anomalous_label

    cv = WithoutLiersCV(cv_strategy, anomalous_label=anomalous_label)

    for train_index, _ in cv.split(X, y, groups):
        y_train = y[train_index]
        assert np.all(y_train != anomalous_label)

    assert cv.get_n_splits(X, y, groups) == cv_strategy.get_n_splits(X, y, groups)


@pytest.mark.parametrize("anomalous_label", [-1, 1])
def test_exclude_from_test(anomalous_label: int) -> None:
    rng = np.random.default_rng()
    size = 1000

    X = rng.normal(size=(size, 3))
    y = (rng.normal(size=size) > 1.5).astype(int)

    y[y == 1] = anomalous_label

    base_cv = KFold(3)
    cv_keep = WithoutLiersCV(base_cv, anomalous_label=anomalous_label)
    cv_drop = WithoutLiersCV(base_cv, anomalous_label=anomalous_label, exclude_from_test=True)

    splits = zip(base_cv.split(X, y), cv_keep.split(X, y), cv_drop.split(X, y))
    for (_, test_base), (train_keep, test_keep), (train_drop, test_drop) in splits:
        # By default the test folds are exactly the ones of the base splitter
        np.testing.assert_array_equal(test_keep, test_base)
        # `exclude_from_test` only affects the test folds
        np.testing.assert_array_equal(train_drop, train_keep)
        np.testing.assert_array_equal(test_drop, test_base[y[test_base] != anomalous_label])
        assert np.all(y[test_drop] != anomalous_label)


@pytest.mark.parametrize("exclude_from_test", [False, True])
def test_split_expected_indices(exclude_from_test: bool) -> None:
    X = np.arange(8).reshape(-1, 1)
    y = np.array([0, -1, 0, 0, -1, 0, 0, -1])

    # KFold(2) yields test folds [0, 1, 2, 3] and [4, 5, 6, 7], with the other half as train fold
    cv = WithoutLiersCV(KFold(n_splits=2), anomalous_label=-1, exclude_from_test=exclude_from_test)
    (train_0, test_0), (train_1, test_1) = cv.split(X, y)

    np.testing.assert_array_equal(train_0, [5, 6])
    np.testing.assert_array_equal(train_1, [0, 2, 3])

    if exclude_from_test:
        np.testing.assert_array_equal(test_0, [0, 2, 3])
        np.testing.assert_array_equal(test_1, [5, 6])
    else:
        np.testing.assert_array_equal(test_0, [0, 1, 2, 3])
        np.testing.assert_array_equal(test_1, [4, 5, 6, 7])
