import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.pipeline import make_pipeline
import pytest
from sklego.preprocessing import ColumnSelector


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


def test_select_two(df):
    result_df = ColumnSelector(["d", "e"]).fit_transform(df)
    expected_df = pd.DataFrame(
        {"d": ["b", "a", "a", "b", "a", "b"], "e": [0, 1, 0, 1, 0, 1]}
    )

    assert_frame_equal(result_df, expected_df)


def test_select_one(df):
    result_df = ColumnSelector(["e"]).fit_transform(df)
    expected_df = pd.DataFrame({"e": [0, 1, 0, 1, 0, 1]})

    assert_frame_equal(result_df, expected_df)


def test_select_all(df):
    result_df = ColumnSelector(["a", "b", "c", "d", "e"]).fit_transform(df)
    assert_frame_equal(result_df, df)


def test_select_none(df):
    with pytest.raises(ValueError):
        ColumnSelector([]).fit_transform(df)


def test_select_not_in_frame(df):
    with pytest.raises(KeyError):
        ColumnSelector(["f"]).fit_transform(df)


def test_select_one_in_pipeline(df):
    pipe = make_pipeline(ColumnSelector(["d"]))
    result_df = pipe.fit_transform(df)
    expected_df = pd.DataFrame({"d": ["b", "a", "a", "b", "a", "b"]})

    assert_frame_equal(result_df, expected_df)


def test_get_feature_names():
    df = pd.DataFrame({"a": [4, 5, 6], "b": ["4", "5", "6"]})
    transformer = ColumnSelector("a").fit(df)
    assert transformer.get_feature_names() == ["a"]
