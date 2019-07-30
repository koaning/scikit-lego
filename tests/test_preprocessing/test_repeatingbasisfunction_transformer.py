import pytest
import numpy as np
import pandas as pd

from sklego.preprocessing import RepeatingBasisFunction


@pytest.fixture()
def df():
    return pd.DataFrame({"a": [1, 2, 3, 4, 5, 6],
                         "b": np.log([10, 9, 8, 7, 6, 5]),
                         "c": ["a", "b", "a", "b", "c", "c"],
                         "d": ["b", "a", "a", "b", "a", "b"],
                         "e": [0, 1, 0, 1, 0, 1]})


def test_int_indexing(df):
    X, y = df[["a", "b", "c", "d"]], df[["e"]]
    tf = RepeatingBasisFunction(column=0,n_periods=4,remainder="passthrough")
    assert tf.fit(X, y).transform(X).shape == (6, 7)

def test_str_indexing(df):
    X, y = df[["a", "b", "c", "d"]], df[["e"]]
    tf = RepeatingBasisFunction(column="b",n_periods=4,remainder="passthrough")
    assert tf.fit(X, y).transform(X).shape == (6, 7)

def test_dataframe_equals_array(df):
    X, y = df[["a", "b", "c", "d"]], df[["e"]]
    tf = RepeatingBasisFunction(column=1, n_periods=4, remainer="passthrough")
    df_transformed = tf.fit(X,y).transform(X)
    array_transformed = tf.fit(X.values,y).transform(X.values)
    np.testing.assert_array_equal(df_transformed, array_transformed)