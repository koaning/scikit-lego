import pandas as pd
from sklego.preprocessing import DateTimeFeatures
import pytest


@pytest.fixture()
def timeseries_df():
    # Generate example dataset
    X = pd.DataFrame({
        "date": pd.to_datetime(["2017-1-19", "2017-12-25", "2018-4-1", "2018-8-8 21:00",
                                "2019-4-21 12:15:00"]),
        "amount": [85, 61, 38, 74, 18],
    })
    y = [0, 1, 1, 0, 0]
    return X, y


def test_timeseries_feature_adder(timeseries_df):
    """
    Test that the DateTimeFeatures function works as expected.
    :param timeseries_df: produces the samples data frame
    :return:
    """
    X, y = timeseries_df

    tsfa = DateTimeFeatures()
    tsfa.fit(X, y)
    result = tsfa.transform(X, y)
    expected_result_weekday = pd.Series([3, 0, 6, 2, 6], name="weekday")

    pd.testing.assert_series_equal(result["weekday"], expected_result_weekday)
    assert result.shape == (5, 10)


def test_timeseries_wrong_date_feature(timeseries_df):
    """
    Test that the TimeFeatures function raises an error because the supplied date_feature does not
    exist as one of the dt attributes.
    :param timeseries_df: produces the samples data frame
    :return:
    """
    X, y = timeseries_df

    tsfa = DateTimeFeatures(date_features=["year", "weekday", "holiday"])

    with pytest.raises(ValueError):
        tsfa.fit(X, y)
