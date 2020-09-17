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

    assert caplog.messages[0].startswith("[do_nothing(df)]")
    assert caplog.messages[1].startswith("[do_nothing(df, kwargs = {'a': '1'})]")
    assert caplog.messages[2].startswith("[do_something(df)]")


@pytest.mark.parametrize("time_taken", [True, False])
def test_log_time(time_taken, caplog, test_df):
    caplog.clear()

    @log_step(time_taken=time_taken)
    def do_nothing(df, *args, **kwargs):
        return df

    test_df.pipe(do_nothing)

    assert ("time=" in caplog.messages[0]) == time_taken


@pytest.mark.parametrize("shape", [True, False])
def test_log_shape(shape, caplog, test_df):
    caplog.clear()

    @log_step(shape=shape)
    def do_nothing(df, *args, **kwargs):
        return df

    test_df.pipe(do_nothing)

    message = caplog.messages[0]

    assert (f"n_obs={test_df.shape[0]}" in message) == shape
    assert (f"n_col={test_df.shape[1]}" in message) == shape


def test_log_shape_delta(caplog, test_df):
    caplog.clear()

    @log_step(shape_delta=True)
    def do_nothing(df, *args, **kwargs):
        return df

    @log_step(shape_delta=True)
    def add_row(df, *args, **kwargs):
        df = df.copy()
        df.loc["new_row", :] = df.iloc[0, :]
        return df

    @log_step(shape_delta=True)
    def remove_row(df, *args, **kwargs):
        return df.drop(index="new_row")

    @log_step(shape_delta=True)
    def add_column(df, *args, **kwargs):
        return df.assign(new_column=42)

    @log_step(shape_delta=True)
    def remove_column(df, *args, **kwargs):
        return df.drop(columns="new_column")

    (
        test_df
        .pipe(do_nothing)
        .pipe(add_row)
        .pipe(remove_row)
        .pipe(add_column)
        .pipe(remove_column)
    )

    assert "delta=(0, 0)" in caplog.messages[0]
    assert "delta=(+1, 0)" in caplog.messages[1]
    assert "delta=(-1, 0)" in caplog.messages[2]
    assert "delta=(0, +1)" in caplog.messages[3]
    assert "delta=(0, -1)" in caplog.messages[4]


@pytest.mark.parametrize("names", [True, False])
def test_log_names(names, caplog, test_df):
    caplog.clear()

    @log_step(names=names)
    def do_nothing(df, *args, **kwargs):
        return df

    test_df.pipe(do_nothing)

    message = caplog.messages[0]

    assert ("names=" in message) == names

    if names:
        assert all(col in message for col in test_df.columns)


@pytest.mark.parametrize("dtypes", [True, False])
def test_log_dtypes(dtypes, caplog, test_df):
    caplog.clear()

    @log_step(dtypes=dtypes)
    def do_nothing(df, *args, **kwargs):
        return df

    test_df.pipe(do_nothing)

    message = caplog.messages[0]

    assert ("dtypes=" in message) == dtypes

    if dtypes:
        assert str(test_df.dtypes.to_dict()) in message


def test_log_not_names_and_dtypes(caplog, test_df):
    caplog.clear()

    @log_step(names=True, dtypes=True)
    def do_nothing(df, *args, **kwargs):
        return df

    test_df.pipe(do_nothing)

    assert "names=" not in caplog.messages[0]


def test_log_custom_logger(caplog, test_df):
    caplog.clear()

    logger_name = "my_custom_logger"

    my_logger = logging.getLogger(logger_name)

    @log_step(logger=my_logger)
    def do_nothing(df, *args, **kwargs):
        return df

    test_df.pipe(do_nothing)

    assert logger_name in caplog.text
