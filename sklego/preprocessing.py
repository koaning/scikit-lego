import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TimeSeriesFeatureAdder(TransformerMixin, BaseEstimator):
    """
    Add common time-series features to data frame in your scikit-learn pipeline.
    Currently, it returns the day of the week.

    :param date_col: name of the date column (default= 'date')
    :param make_dummies: boolean whether the days of week will be added as dummy variables or as
    one column. In case of one column, Mondays = 0 and Sundays = 6. (default= false).

    """

    def __init__(self, date_col="date", make_dummies=False):
        self.date_col = date_col
        self.make_dummies = make_dummies

    def fit(self, X, y):
        _ = X.get(self.date_col,
                  f"Date column {self.date_col} doesn't exist. Please provide another one.")
        if isinstance(_, str):
            print(_)

        # TODO: check that column is date time, below doesn't work
        # assert isinstance(X[date_col], np.datetime64)
        return self

    def transform(self, X, y):
        X = self._get_days_of_week(X, self.date_col)

        return X

    def _get_days_of_week(self, dataf: pd.DataFrame, date_col: str = "date"):
        """
        The day of the week with Monday=0, Sunday=6.

        :param dataf: data frame that contains a date column. (type pandas.DataFrame)
        :returns pandas.DataFrame with an extra column day_of_week
        """
        df_with_days_of_week = dataf.assign(day_of_week=lambda d: d[date_col].dt.dayofweek)
        if self.make_dummies:
            df_with_days_of_week = pd.get_dummies(df_with_days_of_week, columns=["day_of_week"],
                                                  prefix="day", dtype=int)

        return df_with_days_of_week
