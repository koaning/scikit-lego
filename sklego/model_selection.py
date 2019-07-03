import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_array

from sklego.base import Clusterer


class TimeGapSplit:
    """
    Time Series cross-validator
    ---------------------------
    Provides train/test indices to split time series data samples.
    This cross-validation object is a variation of TimeSeriesSplit with the following differences:
    1. The splits are made based on datetime duration, instead of number of rows.
    2. The user specifies the training and the validation durations
    3. The user can specify a 'gap' duration that is omitted from the end part of the training split
    Those 3 parameters can be used to really replicate how the model
    is going to be used in production in batch learning.
    i.e. you can fix:
    1. The historical training data
    2. The retraining frequency
    3. The period of the forward looking window necessary to create the target.
    This period at the end of your training fold is dropped due to lack of recent data to create your target.
    Each validation fold doesn't overlap. The entire 'window' moves by 1 valid_duration until there is not enough data.
    The number of folds is automatically defined that way.
    :param pandas.DataFrame df: DataFrame that should have all the indices of X used in split()
    :param str date_col: Name of the column of datetime in the df
    :param datetime.timedelta train_duration: historical training data
    :param datetime.timedelta valid_duration: retraining frequency
    :param datetime.timedelta gap_duration: forward looking window of the target
    """

    def __init__(self, df, date_col, train_duration, valid_duration, gap_duration=timedelta(0)):
        if train_duration <= gap_duration:
            raise AssertionError("gap_duration is longer than train_duration, it should be shorter.")

        df[date_col] = pd.to_datetime(df[date_col])
        self.df = df
        self.date_col = date_col
        self.train_duration = train_duration
        self.valid_duration = valid_duration
        self.gap_duration = gap_duration

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.
        :param pandas.DataFrame X:
        :param y: Always ignored, exists for compatibility
        :param groups: Always ignored, exists for compatibility
        """
        date_series = self.df.loc[X.index][self.date_col]
        date_series = date_series.sort_values(ascending=True)
        date_min = date_series.min()
        date_max = date_series.max()

        current_date = date_min
        while True:
            if current_date + self.train_duration + self.valid_duration > date_max:
                break

            train_i = date_series[
                (date_series >= current_date) &
                (date_series < current_date + self.train_duration - self.gap_duration)].index.values
            valid_i = date_series[
                (date_series >= current_date + self.train_duration) &
                (date_series < current_date + self.train_duration + self.valid_duration)].index.values

            current_date = current_date + self.valid_duration

            yield (np.array([X.index.get_loc(i) for i in train_i]),
                   np.array([X.index.get_loc(i) for i in valid_i]))

    def get_n_splits(self, X=None, y=None, groups=None):

        return sum(1 for x in self.split(X, y, groups))


class KlusterFoldValidation:
    """
    KlusterFold cross validator

    - Create folds based on provided cluster method

    :param cluster_method: Clustering method with fit_predict attribute
    """

    def __init__(self, cluster_method=None):
        if not isinstance(cluster_method, Clusterer):
            raise ValueError("The KlusterFoldValidation only works on cluster methods with .fit_predict.")

        self.cluster_method = cluster_method
        self.n_splits = None

    def split(self, X, y=None, groups=None):
        """
        Generator to iterate over the indices

        :param X: Array to split on
        :param y: Always ignored, exists for compatibility
        :param groups: Always ignored, exists for compatibility
        """

        X = check_array(X)

        if not self._method_is_fitted(X):
            self.cluster_method.fit(X)
        clusters = self.cluster_method.predict(X)

        self.n_splits = len(np.unique(clusters))

        if self.n_splits < 2:
            raise ValueError(
                f"Clustering method resulted in {self.n_splits} cluster, too few for fold validation"
            )

        for label in np.unique(clusters):
            yield (np.where(clusters != label)[0], np.where(clusters == label)[0])

    def _method_is_fitted(self, X):
        """
        :param X: Array to use if the method is fitted
        :return: True if fitted, else False
        """
        try:
            self.cluster_method.predict(X[0:1, :])
            return True
        except NotFittedError:
            return False
