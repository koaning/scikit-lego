import logging

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest

from sklego.pandas_utils import (
    _add_lagged_dataframe_columns,
    _add_lagged_numpy_columns,
    add_lags,
    log_step,
    log_step_extra,
)

logging.basicConfig(level=logging.INFO)


@pytest.fixture
def data():
    return {"X1": [0, 1, 2], "X2": [float("nan"), "178", "154"]}


@pytest.fixture
def test_X():
    return np.array([[-4, 2], [-2, 0], [4, -6]])


@pytest.mark.parametrize("frame_func", [
    pd.DataFrame,
    lambda data: pl.DataFrame(data, strict=False)
])
def test_add_lags_wrong_inputs(data, frame_func):
    invalid_df = [[1, 2, 3], [4, 5, 6]]
    invalid_lags = ["1", "2"]
    test_df = frame_func(data)
    with pytest.raises(ValueError, match="lags must be a list of type: ?"):
        add_lags(test_df, ["X1"], invalid_lags)
    with pytest.raises(ValueError, match="X type should be one of: ?"):
        add_lags(invalid_df, ["X1"], 1)


@pytest.mark.parametrize("frame_func", [pd.DataFrame, pl.DataFrame, pl.LazyFrame])
def test_add_lags_correct_df(data, frame_func):
    test_df = frame_func(data)
    expected = frame_func({"X1": [1, 2], "X2": ["178", "154"], "X1-1": [0, 1]})
    ans = add_lags(test_df, "X1", -1)
    if isinstance(ans, pl.LazyFrame):
        ans = ans.collect()
    if isinstance(expected, pl.LazyFrame):
        expected = expected.collect()
    assert [x for x in ans.columns] == [x for x in expected.columns]
    assert (ans.to_numpy() == expected.to_numpy()).all()


def test_add_lags_correct_X(test_X):
    expected = np.array([[-4, 2, -2, 3, 0, -6]])
    assert (add_lags(test_X, [0, 1], [1, 2]) == expected).all()


@pytest.mark.parametrize("frame_func", [pd.DataFrame, pl.DataFrame])
def test_add_lagged_dataframe_columns(data, frame_func):
    test_df = nw.from_native(frame_func(data))
    with pytest.raises(KeyError, match="The column does not exist"):
        _add_lagged_dataframe_columns(test_df, ["last_name"], 1, True)


def test_add_lagged_numpy_columns(test_X):
    err_indexed_integes = "Matrix columns are indexed by integers"
    err_column_not_exists = "The column does not exist"
    with pytest.raises(KeyError, match=err_column_not_exists):
        _add_lagged_numpy_columns(test_X, [15], 1, True)
    with pytest.raises(ValueError, match=err_indexed_integes):
        _add_lagged_numpy_columns(test_X, ["test"], 1, True)
    with pytest.raises(ValueError, match=err_indexed_integes):
        _add_lagged_numpy_columns(test_X, ["test"], 1, True)


def test_log_step(capsys, data):
    """Base test of log_step without any arguments to the logger"""
    test_df = pd.DataFrame(data)

    @log_step
    def do_something(df):
        return df.drop(0)

    @log_step
    def do_nothing(df, *args, **kwargs):
        return df

    (test_df.pipe(do_nothing).pipe(do_nothing, a="1").pipe(do_something))

    captured = capsys.readouterr()
    print_statements = captured.out.split("\n")

    assert print_statements[0].startswith("[do_nothing(df)]")
    assert print_statements[1].startswith("[do_nothing(df, kwargs = {'a': '1'})]")
    assert print_statements[2].startswith("[do_something(df)]")


def test_log_step_display_args(capsys, data):
    """Test that we can disable printing function arguments in the log_step"""
    test_df = pd.DataFrame(data)

    @log_step(display_args=False)
    def do_something(df):
        return df.drop(0)

    @log_step(display_args=False)
    def do_nothing(df, *args, **kwargs):
        return df

    (test_df.pipe(do_nothing).pipe(do_nothing, a="1").pipe(do_something))

    captured = capsys.readouterr()
    print_statements = captured.out.split("\n")

    assert print_statements[0].startswith("[do_nothing]")
    assert "kwargs = {'a': '1'}" not in print_statements[1]
    assert print_statements[2].startswith("[do_something]")


def test_log_step_logger(caplog, data):
    """Base test of log_step with a logger supplied instead of default print"""
    test_df = pd.DataFrame(data)
    caplog.clear()

    @log_step(print_fn=logging.info)
    def do_something(df):
        return df.drop(0)

    @log_step(print_fn=logging.info)
    def do_nothing(df, *args, **kwargs):
        return df

    with caplog.at_level(logging.INFO):
        (test_df.pipe(do_nothing).pipe(do_nothing, a="1").pipe(do_something))

    assert caplog.messages[0].startswith("[do_nothing(df)]")
    assert caplog.messages[1].startswith("[do_nothing(df, kwargs = {'a': '1'})]")
    assert caplog.messages[2].startswith("[do_something(df)]")


@pytest.mark.parametrize("time_taken", [True, False])
def test_log_time(time_taken, capsys, data):
    """Test logging of time taken can be switched on and off"""
    test_df = pd.DataFrame(data)

    @log_step(time_taken=time_taken)
    def do_nothing(df, *args, **kwargs):
        return df

    test_df.pipe(do_nothing)

    captured = capsys.readouterr()
    print_statements = captured.out.split("\n")

    assert ("time=" in print_statements[0]) == time_taken


@pytest.mark.parametrize("shape", [True, False])
def test_log_shape(shape, capsys, data):
    """Test logging of shape can be switched on and off"""
    test_df = pd.DataFrame(data)

    @log_step(shape=shape)
    def do_nothing(df, *args, **kwargs):
        return df

    test_df.pipe(do_nothing)

    captured = capsys.readouterr()

    assert (f"n_obs={test_df.shape[0]}" in captured.out) == shape
    assert (f"n_col={test_df.shape[1]}" in captured.out) == shape


def test_log_shape_delta(capsys, data):
    """Test logging of shape delta can be switched on and off"""
    test_df = pd.DataFrame(data)

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

    (test_df.pipe(do_nothing).pipe(add_row).pipe(remove_row).pipe(add_column).pipe(remove_column))

    captured = capsys.readouterr()
    print_statements = captured.out.split("\n")

    assert "delta=(0, 0)" in print_statements[0]
    assert "delta=(+1, 0)" in print_statements[1]
    assert "delta=(-1, 0)" in print_statements[2]
    assert "delta=(0, +1)" in print_statements[3]
    assert "delta=(0, -1)" in print_statements[4]


@pytest.mark.parametrize("names", [True, False])
def test_log_names(names, capsys, data):
    """Test logging of names can be switched on and off"""
    test_df = pd.DataFrame(data)

    @log_step(names=names)
    def do_nothing(df, *args, **kwargs):
        return df

    test_df.pipe(do_nothing)

    captured = capsys.readouterr()

    assert ("names=" in captured.out) == names

    if names:
        assert all(col in captured.out for col in test_df.columns)


@pytest.mark.parametrize("dtypes", [True, False])
def test_log_dtypes(dtypes, capsys, data):
    """Test logging of dtypes can be switched on and off"""
    test_df = pd.DataFrame(data)

    @log_step(dtypes=dtypes)
    def do_nothing(df, *args, **kwargs):
        return df

    test_df.pipe(do_nothing)

    captured = capsys.readouterr()

    assert ("dtypes=" in captured.out) == dtypes

    if dtypes:
        assert str(test_df.dtypes.to_dict()) in captured.out


def test_log_not_names_and_dtypes(capsys, data):
    """
    Test that not both names and types are logged, even if we set both to True
    We don't want this because dtypes also prints the names
    """
    test_df = pd.DataFrame(data)

    @log_step(names=True, dtypes=True)
    def do_nothing(df, *args, **kwargs):
        return df

    test_df.pipe(do_nothing)

    captured = capsys.readouterr()

    assert "names=" not in captured.out


def test_log_custom_logger(caplog, data):
    """Test that we can supply a custom logger to the log_step"""
    test_df = pd.DataFrame(data)
    caplog.clear()

    logger_name = "my_custom_logger"

    my_logger = logging.getLogger(logger_name)

    @log_step(print_fn=my_logger.info)
    def do_nothing(df, *args, **kwargs):
        return df

    with caplog.at_level(logging.INFO):
        test_df.pipe(do_nothing)

    assert logger_name in caplog.text


@pytest.mark.parametrize("log_error", [True, False])
def test_log_error(log_error, capsys, data):
    """Test logging of shape can be switched on and off"""
    test_df = pd.DataFrame(data)

    err_msg = "This is a test Exception"

    @log_step(log_error=log_error)
    def do_nothing(df, *args, **kwargs):
        raise RuntimeError(err_msg)

    err_reraised = False
    try:
        test_df.pipe(do_nothing)
    except RuntimeError:
        err_reraised = True

    captured = capsys.readouterr()

    assert err_reraised
    assert "FAILED" in captured.out
    assert (f"FAILED with error: {err_msg}" in captured.out) == log_error


def test_log_extra(capsys):
    """Base test for the log_step_extra function"""
    n_cats = 3
    n_dogs = 2

    test_df = pd.DataFrame({"id": range(n_cats + n_dogs), "animals": ["dog"] * n_dogs + ["cat"] * n_cats})

    def cat_counter(df):
        return f"cats={(df['animals']=='cat').sum()}"

    @log_step_extra(cat_counter)
    def do_nothing(df, *args, **kwargs):
        return df

    @log_step_extra(cat_counter)
    def double_df(df, *args, **kwargs):
        return pd.concat([df, df], axis=0)

    test_df.pipe(do_nothing).pipe(double_df)

    captured = capsys.readouterr()
    print_statements = captured.out.split("\n")

    assert f"cats={n_cats}" in print_statements[0]
    assert f"cats={2*n_cats}" in print_statements[1]


def test_log_extra_kwargs(capsys):
    """Test that we can supply kwargs to user-specified logging functions"""
    n_cats = 3
    n_dogs = 2

    test_df = pd.DataFrame({"id": range(n_cats + n_dogs), "animals": ["dog"] * n_dogs + ["cat"] * n_cats})

    def animal_counter(df, animal="cat"):
        return f"{animal}s={(df['animals']==animal).sum()}"

    @log_step_extra(animal_counter, animal="dog")
    def do_nothing(df, *args, **kwargs):
        return df

    @log_step_extra(animal_counter, animal="dog")
    def double_df(df, *args, **kwargs):
        return pd.concat([df, df], axis=0)

    test_df.pipe(do_nothing).pipe(double_df)

    captured = capsys.readouterr()
    print_statements = captured.out.split("\n")

    assert f"dogs={n_dogs}" in print_statements[0]
    assert f"dogs={2*n_dogs}" in print_statements[1]


def test_log_extra_multiple(capsys, data):
    """Test that we can add multiple logging functions"""
    test_df = pd.DataFrame(data)

    @log_step_extra(len, type)
    def do_nothing(df, *args, **kwargs):
        return df

    test_df.pipe(do_nothing)

    captured = capsys.readouterr()

    assert str(len(test_df)) in captured.out
    assert str(type(test_df)) in captured.out


def test_log_extra_no_func(data):
    """We need at least one logging function"""
    test_df = pd.DataFrame(data)
    with pytest.raises(ValueError) as e:

        @log_step_extra()
        def do_nothing(df, *args, **kwargs):
            return df

        test_df.pipe(do_nothing)

        assert "log_function" in str(e)


def test_log_extra_not_callable_func(data):
    """Make sure the logging functions are checked to be callable"""
    test_df = pd.DataFrame(data)
    with pytest.raises(ValueError) as e:

        @log_step_extra(1)
        def do_nothing(df, *args, **kwargs):
            return df

        test_df.pipe(do_nothing)

        assert "callable" in str(e)
        assert "int" in str(e)


def test_log_extra_custom_logger(caplog, data):
    """Test that we can supply a custom logger to the log_step_extra"""
    test_df = pd.DataFrame(data)
    caplog.clear()

    logger_name = "my_custom_logger"

    my_logger = logging.getLogger(logger_name)

    @log_step_extra(len, print_fn=my_logger.info)
    def do_nothing(df, *args, **kwargs):
        return df

    with caplog.at_level(logging.INFO):
        test_df.pipe(do_nothing)

    assert logger_name in caplog.text
