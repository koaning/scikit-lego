import pytest
import pandas as pd
import numpy as np
import logging

from sklego.pandas import log_step, _add_lagged_pandas_columns, _add_lagged_numpy_columns

logging.basicConfig(level=logging.INFO)


@pytest.fixture
def test_df():
    return pd.DataFrame({
        'X1': [0, 1, 2],
        'X2': [np.nan, '178', '154']
    })


@pytest.fixture
def test_X():
    return np.array([[-4, 2],
                     [-2, 0],
                     [4, -6]])


def test_add_lagged_pandas_columns(test_df):
    with pytest.raises(KeyError, message='The column does not exist'):
        _add_lagged_pandas_columns(test_df, ['last_name'], 1)


def test_add_lagged_numpy_columns(test_X):
    err_indexed_integes = 'Matrix columns are indexed by integers'
    err_column_not_exists = 'The column does not exist'
    with pytest.raises(KeyError, message=err_column_not_exists):
        _add_lagged_numpy_columns(test_X, [15], 1)
    with pytest.raises(ValueError, message=err_indexed_integes):
        _add_lagged_numpy_columns(test_X, ['test'], 1)
    with pytest.raises(ValueError, message=err_indexed_integes):
        _add_lagged_numpy_columns(test_X, ['test'], 1)


def test_logging(caplog, test_df):
    caplog.clear()

    @log_step
    def do_something(df):
        return df.drop(0)

    @log_step
    def do_nothing(df, *args, **kwargs):
        return df

    (test_df
        .pipe(do_nothing)
        .pipe(do_nothing, a='1')
        .pipe(do_something))

    assert caplog.messages[0].startswith("[ do_nothing(df) ] n_obs=3 n_col=2 ")
    assert caplog.messages[1].startswith("[ do_nothing(df, kwargs = {'a': '1'}) ] n_obs=3 n_col=2 ")
    assert caplog.messages[2].startswith("[ do_something(df) ] n_obs=2 n_col=2 ")
