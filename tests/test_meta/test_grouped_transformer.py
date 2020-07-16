
import pytest
import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import check_X_y

from sklego.common import flatten
from sklego.meta import GroupedTransformer
from tests.conftest import transformer_checks, nonmeta_checks, general_checks


@pytest.mark.parametrize(
    "test_fn", flatten([transformer_checks, nonmeta_checks, general_checks])
)
def test_estimator_checks(test_fn):
    trf = GroupedTransformer(StandardScaler())
    test_fn(GroupedTransformer.__name__, trf)


def test_values(random_xy_dataset_clf):
    X, y = random_xy_dataset_clf
    X, y = check_X_y(X, y)

    # Make sure all groups are present
    groups = np.repeat([0, 1], repeats=(X.shape[0] + 1) // 2)[:X.shape[0], np.newaxis]
    X_with_groups = np.concatenate([groups, X], axis=1)

    # Some weird interval to make sure we test the right values
    scaling_range = (13, 42)

    trf = MinMaxScaler(scaling_range)
    transformer = GroupedTransformer(clone(trf), groups=0)
    transformed = transformer.fit(X_with_groups, y).transform(X_with_groups)

    assert transformed.shape == X.shape

    df_with_groups = pd.concat([pd.Series(groups, name="G"), pd.DataFrame(transformed)], axis=1)

    assert (df_with_groups.groupby("G").min() == scaling_range[0]).all(None)
    assert (df_with_groups.groupby("G").max() == scaling_range[1]).all(None)


def test_set_params():
    trf = StandardScaler(with_std=False)
    transformer = GroupedTransformer(trf)

    transformer.set_params(transformer__with_std=True)
    assert trf.with_std


def test_get_params():
    trf = StandardScaler(with_std=False)
    transformer = GroupedTransformer(trf)

    assert transformer.get_params() == {
        "transformer__with_mean": True,
        "transformer__with_std": True,
        "transformer": trf,
        "predict_func": "predict",
    }


def test_non_transformer(random_xy_dataset_clf):
    X, y = random_xy_dataset_clf
    X, y = check_X_y(X, y)

    # This is not a transformer
    trf = LinearRegression()
    transformer = GroupedTransformer(trf)

    with pytest.raises(ValueError):
        transformer.fit(X, y)


def test_multiple_grouping_columns(random_xy_dataset_clf):
    X, y = random_xy_dataset_clf
    X, y = check_X_y(X, y)

    # Make sure all groups are present
    groups = np.tile(
        # 4x2 array, repeated until it fits
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        reps=((X.shape[0] + 1) // 4, 1)
    )[:X.shape[0], :]
    X_with_groups = np.concatenate([groups, X], axis=1)

    # Some weird interval to make sure we test the right values
    scaling_range = (13, 42)

    trf = MinMaxScaler(scaling_range)
    transformer = GroupedTransformer(clone(trf), groups=0)
    transformed = transformer.fit(X_with_groups, y).transform(X_with_groups)

    assert transformed.shape == X.shape

    df_with_groups = pd.concat([
        pd.DataFrame(groups, columns=["A", "B"]),
        pd.DataFrame(transformed)
    ], axis=1)

    assert (df_with_groups.groupby(["A", "B"]).min() == scaling_range[0]).all(None)
    assert (df_with_groups.groupby(["A", "B"]).max() == scaling_range[1]).all(None)


def test_missing_groups_transform(random_xy_dataset_clf):
    X, y = random_xy_dataset_clf
    X, y = check_X_y(X, y)

    # Make sure all groups are present
    groups = np.repeat([0, 1], repeats=(X.shape[0] + 1) // 2)[:X.shape[0], np.newaxis]
    X_with_groups = np.concatenate([groups, X], axis=1)

    # Some weird interval to make sure we test the right values
    scaling_range = (13, 42)

    trf = MinMaxScaler(scaling_range)
    transformer = GroupedTransformer(clone(trf), groups=0)
    transformer.fit(X_with_groups, y)

    # Array with 2 rows, first column a new group. Remaining top are out of range so should be the range
    X_test = np.concatenate([
        np.array([[3], [3]]), np.stack([X.min(axis=0) - 1, X.max(axis=0) + 1], axis=0)
    ], axis=1)

    transformed = transformer.transform(X_test)

    # Top row should all be equal to the small value of the range, bottom the other
    assert (transformed == scaling_range[0]).all(axis=1)[0]
    assert (transformed == scaling_range[1]).all(axis=1)[1]
