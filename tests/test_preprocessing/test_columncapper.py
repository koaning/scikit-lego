import pytest
import numpy as np
import pandas as pd

from sklearn.utils.validation import FLOAT_DTYPES
from sklego.common import flatten
from sklego.preprocessing import ColumnCapper
from tests.conftest import select_tests, transformer_checks, general_checks, nonmeta_checks


@pytest.mark.parametrize(
    "test_fn",
    select_tests(
        flatten([general_checks, nonmeta_checks, transformer_checks]),
        exclude=[
            "check_sample_weights_invariance",
            "check_estimators_nan_inf"
        ]
    )
)
def test_estimator_checks(test_fn):
    test_fn(ColumnCapper.__name__, ColumnCapper())


def test_quantile_range():
    def expect_type_error(quantile_range):
        with pytest.raises(TypeError):
            ColumnCapper(quantile_range)

    def expect_value_error(quantile_range):
        with pytest.raises(ValueError):
            ColumnCapper(quantile_range)

    # Testing quantile_range type
    expect_type_error(quantile_range=1)
    expect_type_error(quantile_range="a")
    expect_type_error(quantile_range={})
    expect_type_error(quantile_range=set())

    # Testing quantile_range values
    # Invalid type:
    expect_type_error(quantile_range=("a", 90))
    expect_type_error(quantile_range=(10, "a"))

    # Invalid limits
    expect_value_error(quantile_range=(-1, 90))
    expect_value_error(quantile_range=(10, 110))

    # Invalid order
    expect_value_error(quantile_range=(60, 40))


def test_interpolation():
    valid_interpolations = ("linear", "lower", "higher", "midpoint", "nearest")
    invalid_interpolations = ("test", 42, None, [], {}, set(), 0.42)

    for interpolation in valid_interpolations:
        ColumnCapper(interpolation=interpolation)

    for interpolation in invalid_interpolations:
        with pytest.raises(ValueError):
            ColumnCapper(interpolation=interpolation)


@pytest.fixture()
def valid_df():
    return pd.DataFrame(
        {"a": [1, np.nan, 3, 4], "b": [11, 12, np.inf, 14], "c": [21, 22, 23, 24]}
    )


def test_X_types_and_transformed_shapes(valid_df):
    def expect_value_error(X, X_transform=None):
        if X_transform is None:
            X_transform = X
        with pytest.raises(ValueError):
            capper = ColumnCapper().fit(X)
            capper.transform(X_transform)

    # Fitted and transformed arrays must have the same number of columns
    expect_value_error(valid_df, valid_df[["a", "b"]])

    invalid_dfs = [
        pd.DataFrame({"a": [np.nan, np.nan, np.nan], "b": [11, 12, 13]}),
        pd.DataFrame({"a": [np.inf, np.inf, np.inf], "b": [11, 12, 13]}),
    ]

    for invalid_df in invalid_dfs:
        expect_value_error(invalid_df)  # contains an invalid column ('a')
        expect_value_error(
            invalid_df["b"]
        )  # 1d arrays should be reshaped before fitted/transformed
        # Like this:
        ColumnCapper().fit_transform(invalid_df["b"].values.reshape(-1, 1))
        ColumnCapper().fit_transform(invalid_df["b"].values.reshape(1, -1))

    capper = ColumnCapper()
    for X in valid_df, valid_df.values:
        assert capper.fit_transform(X).shape == X.shape


def test_nan_inf(valid_df):
    # Capping infs
    capper = ColumnCapper(discard_infs=False)
    assert (capper.fit_transform(valid_df) == np.inf).sum().sum() == 0
    assert np.isnan(capper.fit_transform(valid_df)).sum() == 1

    # Discarding infs
    capper = ColumnCapper(discard_infs=True)
    assert (capper.fit_transform(valid_df) == np.inf).sum().sum() == 0
    assert np.isnan(capper.fit_transform(valid_df)).sum() == 2


def test_dtype_regression(random_xy_dataset_regr):
    X, y = random_xy_dataset_regr
    assert ColumnCapper().fit(X, y).transform(X).dtype in FLOAT_DTYPES


def test_dtype_classification(random_xy_dataset_clf):
    X, y = random_xy_dataset_clf
    assert ColumnCapper().fit(X, y).transform(X).dtype in FLOAT_DTYPES
