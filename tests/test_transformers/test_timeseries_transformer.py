import datetime as dt
import pandas as pd

from sklego.preprocessing import TimeSeriesFeatureAdder


def test_timeseries_feature_adder():

    # Generate example dataset

    start_date = dt.datetime(2019, 1, 1)
    end_date = dt.datetime(2019, 1, 10)

    X = pd.DataFrame({
        "date": pd.date_range(start_date, end_date, freq="2D"),
        "amount": [85, 61, 38, 74, 18],
    })
    y = [0, 1, 1, 0, 0]

    # Test TimeSeriesFeatureAdder without dummies
    tsfa = TimeSeriesFeatureAdder()
    tsfa.fit(X, y)
    result = tsfa.transform(X, y)
    expected_result = pd.Series([1, 3, 5, 0, 2], name="day_of_week")

    pd.testing.assert_series_equal(result["day_of_week"], expected_result)

    # Test TimeSeriesFeatureAdder with dummies
    tsfa = TimeSeriesFeatureAdder(make_dummies=True)
    tsfa.fit(X, y)
    result = tsfa.transform(X, y)
    expected_result = (X
                       .assign(
                        day_0=[0, 0, 0, 1, 0],
                        day_1=[1, 0, 0, 0, 0],
                        day_2=[0, 0, 0, 0, 1],
                        day_3=[0, 1, 0, 0, 0],
                        day_5=[0, 0, 1, 0, 0]))

    pd.testing.assert_frame_equal(result, expected_result)
