import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.pipeline import make_pipeline
import pytest
from sklego.preprocessing import ColumnDropper


@pytest.fixture()
def df():
    return pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6],
            "b": [10, 9, 8, 7, 6, 5],
            "c": ["a", "b", "a", "b", "c", "c"],
            "d": ["b", "a", "a", "b", "a", "b"],
            "e": [0, 1, 0, 1, 0, 1],
        }
    )


def test_drop_two(df):
    result_df = ColumnDropper(["a", "b"]).fit_transform(df)
    expected_df = pd.DataFrame(
        {
            "c": ["a", "b", "a", "b", "c", "c"],
            "d": ["b", "a", "a", "b", "a", "b"],
            "e": [0, 1, 0, 1, 0, 1],
        }
    )

    assert_frame_equal(result_df, expected_df)


def test_drop_one(df):
    result_df = ColumnDropper(["e"]).fit_transform(df)
    expected_df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6],
            "b": [10, 9, 8, 7, 6, 5],
            "c": ["a", "b", "a", "b", "c", "c"],
            "d": ["b", "a", "a", "b", "a", "b"],
        }
    )

    assert_frame_equal(result_df, expected_df)


def test_drop_all(df):
    with pytest.raises(ValueError):
        ColumnDropper(["a", "b", "c", "d", "e"]).fit_transform(df)


def test_drop_none(df):
    result_df = ColumnDropper([]).fit_transform(df)
    assert_frame_equal(result_df, df)


def test_drop_not_in_frame(df):
    with pytest.raises(KeyError):
        ColumnDropper(["f"]).fit_transform(df)


def test_drop_one_in_pipeline(df):
    pipe = make_pipeline(ColumnDropper(["e"]))
    result_df = pipe.fit_transform(df)
    expected_df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6],
            "b": [10, 9, 8, 7, 6, 5],
            "c": ["a", "b", "a", "b", "c", "c"],
            "d": ["b", "a", "a", "b", "a", "b"],
        }
    )

    assert_frame_equal(result_df, expected_df)


def test_get_feature_names():
    df = pd.DataFrame({"a": [4, 5, 6], "b": ["4", "5", "6"]})
    transformer = ColumnDropper("a").fit(df)
    assert transformer.get_feature_names() == ["b"]
