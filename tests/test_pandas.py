import pytest
import pandas as pd
import numpy as np
import logging

from sklego.pandas import log_step

logging.basicConfig(level=logging.INFO)


@pytest.fixture
def test_df():
    return pd.DataFrame({
        'X1': [0, 1, 2],
        'X2': [np.nan, '178', '154']
    })


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
