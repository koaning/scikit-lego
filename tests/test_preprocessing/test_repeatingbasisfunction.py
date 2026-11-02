import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer

from sklego.preprocessing import RepeatingBasisFunction
from sklego.preprocessing.repeatingbasis import _RepeatingBasisFunction


@pytest.fixture()
def df():
    return pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6],
            "b": np.log([10, 9, 8, 7, 6, 5]),
            "c": ["a", "b", "a", "b", "c", "c"],
            "d": ["b", "a", "a", "b", "a", "b"],
            "e": [0, 1, 0, 1, 0, 1],
        }
    )


@pytest.mark.parametrize("params", [{}, {"column": 1}, {"column": "b"}])
@pytest.mark.parametrize("remainder", ["drop", "passthrough"])
@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_column_selection_deprecation(df, params, remainder, method):
    with pytest.warns(UserWarning, match="SplineTransformer"):
        tf = RepeatingBasisFunction(**params, remainder=remainder)

    with pytest.warns(FutureWarning, match="column.*remainder.*ColumnTransformer"):
        getattr(tf, method)(df)


def test_deprecation_warning_lifecycle(df):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.warns(UserWarning, match="SplineTransformer"):
            tf = RepeatingBasisFunction()
        tf.get_params()
        with pytest.warns(UserWarning, match="SplineTransformer"):
            cloned = clone(tf)
    assert not any(issubclass(w.category, FutureWarning) for w in caught)

    for _ in range(2):
        with pytest.warns(FutureWarning, match="column.*remainder.*ColumnTransformer"):
            cloned.fit(df)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cloned.transform(df)
    assert not any(issubclass(w.category, FutureWarning) for w in caught)


@pytest.mark.parametrize("selector", [1, "b"])
@pytest.mark.parametrize("remainder", ["drop", "passthrough"])
def test_external_column_transformer_matches_legacy(df, selector, remainder):
    X = df if isinstance(selector, str) else df.values
    params = {"n_periods": 4, "input_range": (1, 3), "width": 0.5}
    with pytest.warns(UserWarning, match="SplineTransformer"):
        legacy = RepeatingBasisFunction(column=selector, remainder=remainder, **params)
        migrated = ColumnTransformer(
            [("periodic", RepeatingBasisFunction(**params), [selector])], remainder=remainder
        )

    with pytest.warns(FutureWarning, match="column.*remainder.*ColumnTransformer"):
        expected = legacy.fit_transform(X)
    with pytest.warns(UserWarning, match="SplineTransformer"):
        with pytest.warns(FutureWarning, match="column.*remainder.*ColumnTransformer"):
            result = migrated.fit_transform(X)

    np.testing.assert_array_equal(result, expected)
    if remainder == "passthrough":
        np.testing.assert_array_equal(result[:, 4:], df.drop(columns="b").values)


def test_int_indexing(df):
    X, y = df[["a", "b", "c", "d"]], df[["e"]]
    tf = RepeatingBasisFunction(column=0, n_periods=4, remainder="passthrough")
    with pytest.warns(FutureWarning, match="column.*remainder"):
        assert tf.fit(X, y).transform(X).shape == (6, 7)


def test_str_indexing(df):
    X, y = df[["a", "b", "c", "d"]], df[["e"]]
    tf = RepeatingBasisFunction(column="b", n_periods=4, remainder="passthrough")
    with pytest.warns(FutureWarning, match="column.*remainder"):
        assert tf.fit(X, y).transform(X).shape == (6, 7)


def test_drop_remainder(df):
    X, y = df[["a", "b", "c", "d"]], df[["e"]]
    tf = RepeatingBasisFunction(column="b", n_periods=4, remainder="drop")
    with pytest.warns(FutureWarning, match="column.*remainder"):
        assert tf.fit(X, y).transform(X).shape == (6, 4)


def test_dataframe_equals_array(df):
    X, y = df[["a", "b", "c", "d"]], df[["e"]]
    tf = RepeatingBasisFunction(column=1, n_periods=4, remainder="passthrough")
    with pytest.warns(FutureWarning, match="column.*remainder"):
        df_transformed = tf.fit(X, y).transform(X)
    with pytest.warns(FutureWarning, match="column.*remainder"):
        array_transformed = tf.fit(X.values, y).transform(X.values)
    np.testing.assert_array_equal(df_transformed, array_transformed)


def test_when_rbf_helper_receives_more_than_one_col_raises_value_error(df):
    X, y = df[["a", "b", "c", "d"]], df[["e"]]
    rbf_helper_tf = _RepeatingBasisFunction()
    with pytest.raises(ValueError):
        rbf_helper_tf.fit(X, y)


def test_discrete_input_range_distinct_encodings():
    df_week = pd.DataFrame({"weekday": [0, 1, 2, 3, 4, 5, 6]})
    tf = RepeatingBasisFunction(column="weekday", n_periods=4, input_range=(0, 7))
    with pytest.warns(FutureWarning, match="column.*remainder"):
        transformed = tf.fit_transform(df_week)

    assert not np.allclose(transformed[0], transformed[6])


def test_default_input_range_collapses_endpoints():
    df_week = pd.DataFrame({"weekday": [0, 1, 2, 3, 4, 5, 6]})
    tf = RepeatingBasisFunction(column="weekday", n_periods=4, input_range=None)
    with pytest.warns(FutureWarning, match="column.*remainder"):
        transformed = tf.fit_transform(df_week)

    np.testing.assert_allclose(transformed[0], transformed[6])
