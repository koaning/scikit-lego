import pandas as pd
import numpy as np
import logging

from sklego.pandas_utils import log_step

logging.basicConfig(level=logging.INFO)


def test_logging(caplog):
    caplog.clear()
    test_df = pd.DataFrame(
        {'IDs': [0, 1, 2],
         'length': [np.nan, '178', '154']}
    )

    @log_step
    def do_something(df):
        return df

    do_something(test_df)
    assert caplog.messages[0].startswith('[do_something]  n_obs=3 n_col=2 ')
