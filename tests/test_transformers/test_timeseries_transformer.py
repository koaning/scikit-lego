import pandas as pd
from sklego.preprocessing import TimeFeatures
import pytest


def test_timeseries_feature_adder():

    # Generate example dataset
    X = pd.DataFrame({
        "date": pd.to_datetime(["2017-1-19", "2017-12-25", "2018-4-1", "2018-8-8 21:00",
                                "2019-4-21 12:15:00"]),
        "amount": [85, 61, 38, 74, 18],
    })
    y = [0, 1, 1, 0, 0]

    # Test TimeFeatures
    tsfa = TimeFeatures()
    tsfa.fit(X, y)
    result = tsfa.transform(X, y)
    expected_result_weekday = pd.Series([3, 0, 6, 2, 6], name="weekday")

    pd.testing.assert_series_equal(result["weekday"], expected_result_weekday)
    assert result.shape == (5, 10)

    # Test TimeFeatures that raise error
    tsfa = TimeFeatures(date_features=["year", "weekday", "holiday"])

    with pytest.raises(ValueError):
        tsfa.fit(X, y)
