import numbers
from datetime import timedelta
from itertools import combinations
from warnings import warn

import narwhals.stable.v1 as nw
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils.validation import indexable
from sklearn_compat.utils.validation import check_array

from sklego.base import Clusterer
from sklego.common import sliding_window


class TimeGapSplit:
    r"""Provides train/test indices to split time series data samples.

    This cross-validation object is a variation of TimeSeriesSplit with the following  differences:

    - The splits are made based on datetime duration, instead of number of rows.
    - The user specifies the `valid_duration` and either `train_duration` or `n_splits`.
    - The user can specify a `gap_duration` that is added after the training split and before the validation split.

    The 3 duration parameters can be used to really replicate how the model is going to be used in production in batch
    learning.

    Each validation fold doesn't overlap. The entire `window` moves by 1 `valid_duration` until there is not enough
    data.

    If this would lead to more splits than specified with `n_splits`, the `window` moves by `valid_duration` times the
    fraction of possible splits and requested splits:

    - `n_possible_splits = (total_length - train_duration - gap_duration) // valid_duration`
    - `time_shift = valid_duration * n_possible_splits / n_splits`

    so the CV spans the whole dataset.

    If `train_duration` is not passed but `n_splits` is, the training duration is increased to:

    `train_duration = total_length - (gap_duration + valid_duration * n_splits)`

    such that the shifting the entire window by one validation duration spans the whole training set.

    Parameters
    ----------
    date_serie : Series
        Series with the date, that should have all the indices of X used in the split() method.
        If the Series is not pandas-like (for example, if it's a Polars Series, which does not have
        an index) then it must the same length as the `X` and `y` objects passed to `split`.
    valid_duration : datetime.timedelta
        Retraining period.
    stride_duration : datetime.timedelta | None
        The time shift for the training period. If `None`, then it fallbacks to `valid_duration` value.
    train_duration : datetime.timedelta | None, default=None
        Historical training data.
    gap_duration : datetime.timedelta, default=timedelta(0)
        Forward-looking window of the target. The period of the forward-looking window necessary to create your target
        variable. This period is dropped at the end of your training folds due to lack of recent data. In production you
        would have not been able to create the target for that period, and you would have to drop it from the training data.
    n_splits : int | None, default=None
        Number of splits.
    window : Literal["rolling", "expanding"], default="rolling"
        Type of moving window to use.

        - `"rolling"` window has fixed size and is shifted entirely.
        - `"expanding"` left side of window is fixed, right border increases each fold.

    Notes
    -----
    Native cross-dataframe support is achieved using
    [Narwhals](https://narwhals-dev.github.io/narwhals/){:target="_blank"}.
    Supported dataframes are:

    - pandas
    - Polars (eager)
    - Modin
    - cuDF

    See [Narwhals docs](https://narwhals-dev.github.io/narwhals/extending/){:target="_blank"} for an up-to-date list
    (and to learn how you can add your dataframe library to it!), though note that only those
    convertible to `numpy` arrays will work with this class.
    """

    def __init__(
        self,
        date_serie,
        valid_duration,
        stride_duration=None,
        train_duration=None,
        gap_duration=timedelta(0),
        n_splits=None,
        window="rolling",
    ):
        # If stride length is not defined, set it equal to the length validation set
        if stride_duration is None:
            stride_duration = valid_duration

        if stride_duration <= timedelta(milliseconds=0):
            msg = f"`stride_duration` should be a positive timedelta, found {stride_duration}"
            raise ValueError(msg)

        if (train_duration is None) and (n_splits is None):
            raise ValueError("Either train_duration or n_splits have to be defined")

        if (train_duration is not None) and (train_duration <= gap_duration):
            raise ValueError("gap_duration is longer than train_duration, it should be shorter.")

        if (train_duration is not None) and (train_duration <= stride_duration):
            raise ValueError("stride_duration is longer than train_duration, it should be shorter.")

        self.date_serie = nw.from_native(date_serie, series_only=True).alias("__date__")
        self.train_duration = train_duration
        self.valid_duration = valid_duration
        self.gap_duration = gap_duration
        self.stride_duration = stride_duration
        self.n_splits = n_splits
        self.window = window

    def _join_date_and_x(self, X):
        """Creates a DataFrame indexed by the pandas index (the same as `date_serie`) with date column joined with that
        index and with the 'numpy index' column (i.e. just a range) that is required for the output and the rest of
        sklearn.

        If the user is working with index-less dataframes (e.g. Polars), then `self.date_series` needs to be the same
        length as `X`.

        Parameters
        ----------
        X : DataFrame
            Dataframe with the data to split
        """
        X_index_df = nw.maybe_align_index(self.date_serie, X).to_frame().with_row_index("np_index")

        return X_index_df

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : DataFrame
            Dataframe with the data to split.
        y : array-like | None, default=None
            Ignored, present for compatibility.
        groups : array-like | None, default=None
            Ignored, present for compatibility.

        Yields
        -------
        tuple[np.ndarray, np.ndarray]
            Train and test indices of the same fold.
        """

        X = nw.from_native(X, eager_only=True)
        X_index_df = self._join_date_and_x(X)
        X_index_df = X_index_df.sort("__date__", descending=False)

        if len(X) != len(X_index_df):
            raise AssertionError(
                "X and X_index_df are not the same length, there must be some index missing in 'self.date_serie'"
            )

        date_min = X_index_df["__date__"].min()
        date_max = X_index_df["__date__"].max()
        date_length = X_index_df["__date__"].max() - X_index_df["__date__"].min()

        if (self.train_duration is None) and (self.n_splits is not None):
            self.train_duration = date_length - (
                self.gap_duration + self.valid_duration + (self.n_splits - 1) * self.stride_duration
            )

        if (self.train_duration is not None) and (self.train_duration <= self.gap_duration):
            raise ValueError("gap_duration is longer than train_duration, it should be shorter.")

        n_split_max = (
            1 + (date_length - self.train_duration - self.gap_duration - self.valid_duration) / self.stride_duration
        )

        if self.n_splits:
            if n_split_max < self.n_splits:
                raise ValueError(
                    (
                        "Number of folds requested = {1} are greater"
                        " than maximum  ={0} possible"
                        " based on the given values."
                    ).format(n_split_max, self.n_splits)
                )

        current_date = date_min
        start_date = date_min
        # if the n_splits is smaller than what would usually be done for train, validation, stride and gap duration,
        # the next fold is slightly further in time than just stride_duration
        if self.n_splits is not None:
            time_shift = self.stride_duration * n_split_max / self.n_splits
        else:
            time_shift = self.stride_duration
        while True:
            if current_date + self.train_duration + time_shift + self.gap_duration > date_max:
                break

            X_train_df = X_index_df.filter(
                nw.col("__date__") >= start_date, nw.col("__date__") < current_date + self.train_duration
            )
            X_valid_df = X_index_df.filter(
                nw.col("__date__") >= current_date + self.train_duration + self.gap_duration,
                nw.col("__date__") < current_date + self.train_duration + self.valid_duration + self.gap_duration,
            )

            current_date = current_date + time_shift
            if self.window == "rolling":
                start_date = current_date
            yield (
                X_train_df["np_index"].to_numpy(),
                X_valid_df["np_index"].to_numpy(),
            )

    def get_n_splits(self, X=None, y=None, groups=None):
        """Get the number of splits

        Parameters
        ----------
        X : DataFrame
            Dataframe with the data to split.
        y : array-like | None, default=None
            Ignored, present for compatibility.
        groups : array-like | None, default=None
            Ignored, present for compatibility.

        Returns
        -------
        int
            Number of splits.
        """
        return sum(1 for x in self.split(X, y, groups))

    def summary(self, X):
        """Describe all folds.

        Parameters
        ----------
        X : DataFrame
            Dataframe with the data to split.

        Returns
        -------
        DataFrame
            Summary of all folds.
        """
        summary = []
        X = nw.from_native(X, eager_only=True)
        X_index_df = self._join_date_and_x(X)

        summary = {
            "Start date": [],
            "End date": [],
            "Period": [],
            "Unique days": [],
            "nbr samples": [],
            "part": [],
            "fold": [],
        }
        native_namespace = nw.get_native_namespace(X)

        def update_split_info(indices, j, part, summary):
            dates = X_index_df["__date__"][indices]
            mindate = dates.min()
            maxdate = dates.max()
            n_unique = dates.n_unique()

            summary["Start date"].append(mindate)
            summary["End date"].append(maxdate)
            summary["Period"].append(maxdate - mindate)
            summary["Unique days"].append(n_unique)
            summary["nbr samples"].append(len(indices))
            summary["part"].append(part)
            summary["fold"].append(j)

        j = 0
        for i in self.split(nw.to_native(X)):
            train_info = nw.to_native(nw.new_series(name="tmp", values=i[0], native_namespace=native_namespace))
            valid_info = nw.to_native(nw.new_series(name="tmp", values=i[1], native_namespace=native_namespace))
            update_split_info(train_info, j, "train", summary)
            update_split_info(valid_info, j, "valid", summary)
            j = j + 1

        result = nw.from_dict(summary, native_namespace=native_namespace)
        result = nw.maybe_set_index(result, ["fold", "part"])
        return nw.to_native(result)


def KlusterFoldValidation(**kwargs):
    warn(
        "Please use `ClusterFoldValidation` instead of `KlusterFoldValidation`."
        "We will use correct spelling going forward and `KlusterFoldValidation` will be deprecated.",
        DeprecationWarning,
    )
    return ClusterFoldValidation(**kwargs)


class ClusterFoldValidation:
    """Cross validator that creates folds based on provided cluster method.
    This ensures that data points in the same cluster are not split across different folds.

    !!! info "New in version 0.8.2"

    Parameters
    ----------
    cluster_method : Clusterer
        Clustering method to use for the fold validation.
    """

    def __init__(self, cluster_method=None):
        if not isinstance(cluster_method, Clusterer):
            raise ValueError("The KlusterFoldValidation only works on cluster methods with .fit_predict.")

        self.cluster_method = cluster_method
        self.n_splits = None

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Array to split on.
        y : array-like of shape (n_samples,) | None, default=None
            Ignored, present for compatibility.
        groups : array-like of shape (n_samples,) | None, default=None
            Ignored, present for compatibility.

        Yields
        -------
        tuple[np.ndarray, np.ndarray]
            Train and test indices of the same fold.
        """

        X = check_array(X)

        if not self._method_is_fitted(X):
            self.cluster_method.fit(X)
        clusters = self.cluster_method.predict(X)

        self.n_splits = len(np.unique(clusters))

        if self.n_splits < 2:
            raise ValueError(f"Clustering method resulted in {self.n_splits} cluster, too few for fold validation")

        for label in np.unique(clusters):
            yield (
                np.where(clusters != label)[0],
                np.where(clusters == label)[0],
            )

    def _method_is_fitted(self, X):
        """Check if the `cluster_method` is fitted

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Array to use if the method is fitted

        Returns
        -------
        bool
            True if fitted, else False
        """

        try:
            self.cluster_method.predict(X[0:1, :])
            return True
        except NotFittedError:
            return False


class GroupTimeSeriesSplit(_BaseKFold):
    """Sliding window time series split.

    Create `n_splits` folds with an as equally possible size through a smart variant of a brute force search.
    Groups parameter in `.split()` method should be filled with the time groups (e.g. years)

    If `n_splits` is 3 ("*" = train, "x" = test):

    ```console
    * * * x x x - - - - -
    - - - * * * x x x - -
    - - - - - - * * * x x
    ```

    Parameters
    ----------
    n_splits : int
        Amount of (train, test) splits to generate.
    """

    # table above inspired by sktime

    def __init__(self, n_splits):
        if not isinstance(n_splits, numbers.Integral):
            raise ValueError(
                "The number of folds must be of Integral type. %s of type %s was passed." % (n_splits, type(n_splits))
            )
        n_splits = int(n_splits)

        if n_splits <= 1:
            raise ValueError(
                "k-fold cross-validation requires at least one"
                " train/test split by setting n_splits=2 or more,"
                " got n_splits={0}.".format(n_splits)
            )

        self.n_splits = n_splits

    def summary(self):
        """Describe all folds in a pd.DataFrame which displays the groups splits and extra statistics about it.

        Can only be run after having applied the `.split()` method to the `GroupTimeSeriesSplit` instance.

        Returns
        -------
        pd.DataFrame
            Summary of all folds.
        """
        try:
            return (
                self._grouped_df.sort_index()
                .assign(group=lambda df: df["group"].astype(int))
                .assign(obs_per_group=lambda df: df.groupby("group")["observations"].transform("sum"))
                .assign(ideal_group_size=round(self._ideal_group_size))
                .assign(diff_from_ideal_group_size=lambda df: df["obs_per_group"] - df["ideal_group_size"])
            )
        except AttributeError:
            raise AttributeError(".summary() only works after having ran .split(X, y, groups).")

    def split(self, X=None, y=None, groups=None):
        """Generate the train-test splits of all the folds

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
            Data to split.
        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into train/test set,

        Returns
        -------
        List[np.ndarray]
            List containing train-test split indices of each fold.
        """
        if groups is None:
            raise ValueError("Groups cannot be None")
        X, y, groups = indexable(X, y, groups)
        n_groups = np.unique(groups).shape[0]
        if self.n_splits >= n_groups:
            raise ValueError(
                ("n_splits({0}) must be less than the amount of unique groups({1}).").format(self.n_splits, n_groups)
            )
        return list(self._iter_test_indices(X, y, groups))

    def get_n_splits(self, X=None, y=None, groups=None):
        """Get the amount of splits

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
            Ignored, present for compatibility.
        y : array-like of shape (n_samples,), default=None
            Ignored, present for compatibility.
        groups : array-like of shape (n_samples,), default=None
            Ignored, present for compatibility.

        Returns
        -------
        int
            Amount of splits.
        """
        return self.n_splits

    def _check_for_long_estimated_runtime(self, groups):
        """Check for combinations of n_splits and unique groups and raises `UserWarning` if runtime is expected to take
        over one minute.

        Parameters
        ----------
        groups : array-like of shape (n_samples,)
            Groups to check for unique groups and n_splits.

        Raises
        ------
        UserWarning
            If runtime is expected to take over one minute.
        """
        unique_groups = len(set(groups))
        warning = (
            "Finding the optimal split points with {0} unique groups and n_splits at {1} can take several minutes."
        ).format(unique_groups, self.n_splits)
        if self.n_splits == 4 and unique_groups > 250:
            warn(
                warning + " Consider to decrease n_splits to 3 or lower.",
                UserWarning,
            )

        elif self.n_splits == 5 and unique_groups > 100:
            warn(
                warning + " Consider to decrease n_splits to 4 or lower.",
                UserWarning,
            )

        elif self.n_splits > 5 and unique_groups > 30:
            warn(
                warning + " Consider to decrease n_splits to 5 or lower.",
                UserWarning,
            )

    def _iter_test_indices(self, X=None, y=None, groups=None):
        """Calculate the optimal division of groups into folds so that every fold is as equally large as possible.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
            Ignored, present for compatibility.
        y : array-like of shape (n_samples,), default=None
            Ignored, present for compatibility.
        groups : array-like of shape (n_samples,), default=None
            Array with groups.

        Yields
        ------
        tuple[np.ndarray, np.ndarray]
            Train and test indices of the same fold.
        """
        self._check_for_long_estimated_runtime(groups)
        self._first_split_index, self._last_split_index = self._calc_first_and_last_split_index(groups=groups)
        self._best_splits = self._get_split_indices()
        groups = self._regroup(groups)
        for i in range(self.n_splits):
            yield np.where(groups == i)[0], np.where(groups == i + 1)[0]

    def _calc_first_and_last_split_index(self, X=None, y=None, groups=None):
        """Calculate an approximate first and last split point to reduce the amount of options during a brute force
        search.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
            Ignored, present for compatibility.
        y : array-like of shape (n_samples,), default=None
            Ignored, present for compatibility.
        groups : array-like of shape (n_samples,), default=None
            Array with groups.

        Returns
        -------
        tuple[int, int]
            Approximate first and last split indexes.
        """

        # get the counts (=amount of rows) for each group
        self._grouped_df = (
            pd.DataFrame(np.array(groups))
            .rename(columns={0: "index"})
            .groupby("index")
            .size()
            .sort_index()
            .to_frame()
            .rename(columns={0: "observations"})
        )

        # set the ideal group_size and reduce it to 90% to have some leverage
        self._ideal_group_size = np.sum(self._grouped_df["observations"]) / (self.n_splits + 1)
        init_ideal_group_size = self._ideal_group_size * 0.9

        # initialize the index of the first split, to reduce the amount of possible index split options
        first_split_index = (
            self._grouped_df.assign(cumsum_obs=lambda df: df["observations"].cumsum())
            .assign(group_id=lambda df: (df["cumsum_obs"] - 1) // init_ideal_group_size)
            .reset_index()
            .loc[lambda df: df["group_id"] != 0]
            .iloc[0]
            .name
        )
        # initialize the index of the last split point, to reduce the amount of possible index split options
        last_split_index = len(self._grouped_df) - (
            self._grouped_df.assign(
                observations=lambda df: df["observations"].to_numpy()[::-1],
                cumsum_obs=lambda df: df["observations"].cumsum(),
            )
            .reset_index()
            .assign(group_id=lambda df: (df["cumsum_obs"] - 1) // init_ideal_group_size)
            .loc[lambda df: df["group_id"] != 0]
            .iloc[0]
            .name
            - 1
        )
        return first_split_index, last_split_index

    def _get_split_indices(self):
        """Calculate for each possible splits the total absolute different of the groups to the ideal group size and
        saves the split with the least absolute difference.

        Returns
        -------
        int
            Index of the best split point.
        """
        # set the index range to search possible splits for
        index_range = range(self._first_split_index, self._last_split_index)

        observations = self._grouped_df["observations"].tolist()

        # create generator with all the possible index splits
        # e.g. for [0, 1, 3, 5, 8] and self.n_splits = 2
        # [(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
        # with for the first split:
        # group1 = [:1]
        # group2 = [1:2]
        # group3 = [2:]
        splits_generator_shifted = combinations(index_range, self.n_splits)

        # get the first iteration
        first_splits = next(splits_generator_shifted)

        # create a new generator that starts from the beginning again
        splits_generator = combinations(index_range, self.n_splits)

        # generate a list, with for every group the difference between them and the ideal group size, e.g.
        # ideal_group_size = 100
        # group_sizes = [10,20,270]
        # diff_from_ideal_list = [-90, -80, 170]
        diff_from_ideal_list = [sum(observations[: first_splits[0]]) - self._ideal_group_size]
        for split in sliding_window(first_splits, window_size=2, step_size=1):
            try:
                diff_from_ideal_list += [sum(observations[split[0] : split[1]]) - self._ideal_group_size]
            except IndexError:
                diff_from_ideal_list += [sum(observations[split[0] :]) - self._ideal_group_size]

        # keep track of the minimum of the total difference from all groups to the ideal group size
        min_diff = sum([abs(diff) for diff in diff_from_ideal_list])
        best_splits = first_splits

        # loop through all possible split points and check whether a new split
        # has a less total difference from all groups to the ideal group size
        for prev_splits, new_splits in zip(splits_generator, splits_generator_shifted):
            diff_from_ideal_list = self._calc_new_diffs(observations, diff_from_ideal_list, prev_splits, new_splits)
            new_diff = sum([abs(diff) for diff in diff_from_ideal_list])

            # if with the new split the difference is less than the current most optimal, save the new split
            if new_diff < min_diff:
                min_diff = new_diff
                best_splits = new_splits
        return best_splits

    @staticmethod
    def _calc_new_diffs(values, diff_list, prev_splits, new_splits):
        """Calculate the new group size differences compared to the optimal group size.

        Parameters
        ----------
        values : list
            List of values.
        diff_list : list
            List of values with for each index split its difference from the optimal group size.
        prev_splits : tuple
            Indices of the previous splits, excluding index 0 and the last index.
        new_splits : tuple
            Indices of the new splits, excluding index 0 and the last index.

        Returns
        -------
        list
            List of values with for each index split its difference from the optimal group size.
        """
        # calculate which indices have changed, e.g.:
        # new_index = (1,2,5)
        # prev_index = (1,2,4)
        # index_diffs = (0,0,1)
        index_diffs = [new_index - prev_index for prev_index, new_index in zip(prev_splits, new_splits)]
        new_diff_list = diff_list.copy()

        # calculate the effects of the index change to the groups
        for index, diff in enumerate(index_diffs):
            if diff != 0:
                start_index, end_index = (
                    (prev_splits[index], new_splits[index])
                    if prev_splits[index] < new_splits[index]
                    else (new_splits[index], prev_splits[index])
                )

                # calculate the value change from one group to another
                value_change = sum(values[start_index:end_index])

                # if diff < 0 the previous group gains values, so change value_change to -value_change
                value_change = value_change if diff > 0 else -value_change

                # change the values of the current and next group
                new_diff_list[index] += value_change
                new_diff_list[index + 1] -= value_change

        return new_diff_list

    def _regroup(self, groups):
        """Specify in which group every observation belongs.

        Parameters
        ----------
        groups : array-like of shape (n_samples,)
            Array with original groups.

        Returns
        -------
        np.ndarray
            Indices for the train and test splits of each fold
        """

        df = self._grouped_df.copy().reset_index()
        # set each unique group to the right group_id to group them into folds
        df.loc[: self._best_splits[0], "group"] = 0
        for group_id, splits in enumerate(sliding_window(self._best_splits, 2, 1)):
            try:
                df.loc[splits[0] : splits[1], "group"] = group_id + 1
            except IndexError:
                df.loc[splits[0] :, "group"] = group_id + 1

        self._grouped_df = df
        # create a mapper to set every group to the right group_id
        mapper = dict(zip(df["index"], df["group"]))
        return np.vectorize(mapper.get)(groups)
