from contextlib import nullcontext as does_not_raise

import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal as pandas_assert_frame_equal
from polars.testing import assert_frame_equal as polars_assert_frame_equal
from sklearn.pipeline import Pipeline, make_pipeline

from sklego.preprocessing import ColumnSelector


@pytest.fixture()
def data():
    return {
        "a": [1, 2, 3, 4, 5, 6],
        "b": [10, 9, 8, 7, 6, 5],
        "c": ["a", "b", "a", "b", "c", "c"],
        "d": ["b", "a", "a", "b", "a", "b"],
        "e": [0, 1, 0, 1, 0, 1],
    }


@pytest.mark.parametrize(
    "frame_func, assert_func",
    [
        (pd.DataFrame, pandas_assert_frame_equal),
        (pl.DataFrame, polars_assert_frame_equal),
    ],
)
@pytest.mark.parametrize(
    "select, context",
    [
        (["a", "b"], does_not_raise()),  # two
        (["e"], does_not_raise()),  # one
        (["a", "b", "c", "d", "e"], does_not_raise()),  # all)
        ([], pytest.raises(ValueError)),  # none
        (["f"], pytest.raises(KeyError)),  # not in data
    ],
)
@pytest.mark.parametrize("wrapper", [lambda x: x, make_pipeline])
def test_drop(data, frame_func, assert_func, select, context, wrapper):
    sub_data = {k: v for k, v in data.items() if k in select}

    with context:
        transformer = wrapper(ColumnSelector(select))
        result_df = transformer.fit_transform(frame_func(data))
        expected_df = frame_func(sub_data)

        assert_func(result_df, expected_df)

        if not isinstance(transformer, Pipeline):
            assert transformer.get_feature_names() == list(sub_data.keys())
