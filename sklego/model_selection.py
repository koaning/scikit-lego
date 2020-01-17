import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_array

from sklego.base import Clusterer


class TimeGapSplit:
    """
    Provides train/test indices to split time series data samples.

    This cross-validation object is a variation of TimeSeriesSplit with the following differences:

    - The splits are made based on datetime duration, instead of number of rows.
    - The user specifies the training and the validation durations
    - The user can specify a 'gap' duration that is omitted from the end part of the training split

    The 3 duration parameters can be used to really replicate how the model
    is going to be used in production in batch learning.

    Each validation fold doesn't overlap. The entire 'window' moves by 1 valid_duration until there is not enough data.
    The number of folds is automatically defined that way.

    :param pandas.Series date_serie: Series with the date, that should have all the indices of X used in split()
    :param datetime.timedelta train_duration: historical training data.
    :param datetime.timedelta valid_duration: retraining period.
    :param datetime.timedelta gap_duration: forward looking window of the target.
        The period of the forward looking window necessary to create your target variable.
        This period is dropped at the end of your training folds due to lack of recent data.
        In production you would have not been able to create the target for that period, and you would have drop it from
        the training data.
    """

    def __init__(
        self, date_serie, train_duration, valid_duration, gap_duration=timedelta(0)
    ):
        if train_duration <= gap_duration:
            raise ValueError(
                "gap_duration is longer than train_duration, it should be shorter."
            )

        if not date_serie.index.is_unique:
            raise ValueError("date_serie doesn't have a unique index")

        self.date_serie = date_serie.copy()
        self.date_serie = self.date_serie.rename("__date__")
        self.train_duration = train_duration
        self.valid_duration = valid_duration
        self.gap_duration = gap_duration

    def join_date_and_x(self, X):
        """
        Make a DataFrame indexed by the pandas index (the same as date_series) with date column joined with that index
        and with the 'numpy index' column (i.e. just a range) that is required for the output and the rest of sklearn
        :param pandas.DataFrame X:
        """
        X_index_df = pd.DataFrame(range(len(X)), columns=["np_index"], index=X.index)
        X_index_df = X_index_df.join(self.date_serie)

        return X_index_df

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.
        :param pandas.DataFrame X:
        :param y: Always ignored, exists for compatibility
        :param groups: Always ignored, exists for compatibility
        """

        X_index_df = self.join_date_and_x(X)
        X_index_df = X_index_df.sort_values("__date__", ascending=True)

        if len(X) != len(X_index_df):
            raise AssertionError(
                "X and X_index_df are not the same length, "
                "there must be some index missing in 'self.date_serie'"
            )

        date_min = X_index_df["__date__"].min()
        date_max = X_index_df["__date__"].max()

        current_date = date_min
        while True:
            if current_date + self.train_duration + self.valid_duration > date_max:
                break

            X_train_df = X_index_df[
                (X_index_df["__date__"] >= current_date)
                & (
                    X_index_df["__date__"]
                    < current_date + self.train_duration - self.gap_duration
                )
            ]
            X_valid_df = X_index_df[
                (X_index_df["__date__"] >= current_date + self.train_duration)
                & (
                    X_index_df["__date__"]
                    < current_date + self.train_duration + self.valid_duration
                )
            ]

            current_date = current_date + self.valid_duration

            yield (X_train_df["np_index"].values, X_valid_df["np_index"].values)

    def get_n_splits(self, X=None, y=None, groups=None):

        return sum(1 for x in self.split(X, y, groups))

    def summary(self, X):
        """
        Describe all folds
        :param pandas.DataFrame X:
        :returns: ``pd.DataFrame`` summary of all folds
        """
        summary = []
        X_index_df = self.join_date_and_x(X)

        def get_split_info(X, indices, j, part, summary):
            dates = X_index_df.iloc[indices]["__date__"]
            mindate = dates.min()
            maxdate = dates.max()

            s = pd.Series(
                {
                    "Start date": mindate,
                    "End date": maxdate,
                    "Period": pd.to_datetime(maxdate, format="%Y%m%d")
                    - pd.to_datetime(mindate, format="%Y%m%d"),
                    "Unique days": len(dates.unique()),
                    "nbr samples": len(indices),
                },
                name=(j, part),
            )
            summary.append(s)
            return summary

        j = 0
        for i in self.split(X):
            summary = get_split_info(X, i[0], j, "train", summary)
            summary = get_split_info(X, i[1], j, "valid", summary)
            j = j + 1

        return pd.DataFrame(summary)


class KlusterFoldValidation:
    """
    KlusterFold cross validator

    - Create folds based on provided cluster method

    :param cluster_method: Clustering method with fit_predict attribute
    """

    def __init__(self, cluster_method=None):
        if not isinstance(cluster_method, Clusterer):
            raise ValueError(
                "The KlusterFoldValidation only works on cluster methods with .fit_predict."
            )

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
