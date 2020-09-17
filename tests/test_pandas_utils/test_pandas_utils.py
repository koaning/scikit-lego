import pytest
import pandas as pd
import numpy as np
import logging

from sklego.pandas_utils import (
    log_step,
    add_lags,
    _add_lagged_pandas_columns,
    _add_lagged_numpy_columns,
)

logging.basicConfig(level=logging.INFO)


@pytest.fixture
def test_df():
    return pd.DataFrame({"X1": [0, 1, 2], "X2": [np.nan, "178", "154"]})


@pytest.fixture
def test_X():
    return np.array([[-4, 2], [-2, 0], [4, -6]])


def test_add_lags_wrong_inputs(test_df):
    invalid_df = [[1, 2, 3], [4, 5, 6]]
    invalid_lags = ["1", "2"]
    with pytest.raises(ValueError, match="lags must be a list of type: ?"):
        add_lags(test_df, ["X1"], invalid_lags)
    with pytest.raises(ValueError, match="X type should be one of: ?"):
        add_lags(invalid_df, ["X1"], 1)


def test_add_lags_correct_df(test_df):
    expected = pd.DataFrame({"X1": [1, 2], "X2": ["178", "154"], "X1-1": [0, 1]})
    ans = add_lags(test_df, "X1", -1)
    assert (ans.columns == expected.columns).all()
    assert (ans.values == expected.values).all()


def test_add_lags_correct_X(test_X):
    expected = np.array([[-4, 2, -2, 3, 0, -6]])
    assert (add_lags(test_X, [0, 1], [1, 2]) == expected).all()


def test_add_lagged_pandas_columns(test_df):
    with pytest.raises(KeyError, match="The column does not exist"):
        _add_lagged_pandas_columns(test_df, ["last_name"], 1, True)


def test_add_lagged_numpy_columns(test_X):
    err_indexed_integes = "Matrix columns are indexed by integers"
    err_column_not_exists = "The column does not exist"
    with pytest.raises(KeyError, match=err_column_not_exists):
        _add_lagged_numpy_columns(test_X, [15], 1, True)
    with pytest.raises(ValueError, match=err_indexed_integes):
        _add_lagged_numpy_columns(test_X, ["test"], 1, True)
    with pytest.raises(ValueError, match=err_indexed_integes):
        _add_lagged_numpy_columns(test_X, ["test"], 1, True)


def test_logging(caplog, test_df):
    caplog.clear()

    @log_step
    def do_something(df):
        return df.drop(0)

    @log_step
    def do_nothing(df, *args, **kwargs):
        return df

    (test_df.pipe(do_nothing).pipe(do_nothing, a="1").pipe(do_something))

    assert caplog.messages[0].startswith("[do_nothing(df)] n_obs=3 n_col=2 ")
    assert caplog.messages[1].startswith(
        "[do_nothing(df, kwargs = {'a': '1'})] n_obs=3 n_col=2 "
    )
    assert caplog.messages[2].startswith("[do_something(df)] n_obs=2 n_col=2 ")
