from operator import attrgetter
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class TimeSeriesFeatureAdder(TransformerMixin, BaseEstimator):
    """
    Add common time-series features to data frame in your scikit-learn pipeline.
    There are a number of date features that can be added (weekday, weekofyear, year,
    quarter, month, day, hour and minute. By default it will generate all the time features. They
    can be individually excluded.

    :param date_col: name of the date column (default= 'date')


    """

    def __init__(self, date_col="date", weekday=True, weekofyear=True, year=True, quarter=True,
                 month=True, day=True, hour=True, minute=True):
        date_features = np.array(
            ["weekday", "weekofyear", "year", "quarter", "month", "day", "hour", "minute"])
        date_mask = [weekday, weekofyear, year, quarter, month, day, hour, minute]

        self.date_col = date_col
        self.date_features = date_features[date_mask]

    def fit(self, X, y):
        _ = X.get(self.date_col,
                  f"Date column {self.date_col} doesn't exist. Please provide another one.")
        if isinstance(_, str):
            print(_)

        # TODO: check that column is date time, below doesn't work
        # assert isinstance(X[date_col], np.datetime64)
        return self

    def transform(self, X, y):
        return self._get_time_features(X)

    def _get_time_features(self, dataf: pd.DataFrame):
        time_attr = pd.DataFrame(attrgetter(*self.date_features)(dataf[self.date_col].dt)).T
        time_attr.columns = self.date_features
        return pd.concat([dataf, time_attr], axis=1)
