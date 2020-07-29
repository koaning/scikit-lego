import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.utils.validation import (
    check_is_fitted,
    check_array,
)

from sklego.common import as_list, expanding_list
from ._grouped_utils import relative_shrinkage, constant_shrinkage, min_n_obs_shrinkage
from ._grouped_utils import _split_groups_and_values


class GroupedPredictor(BaseEstimator):
    """
    Construct an estimator per data group. Splits data by values of a
    single column and fits one estimator per such column.

    :param estimator: the model/pipeline to be applied per group
    :param groups: the column(s) of the matrix/dataframe to select as a grouping parameter set
    :param shrinkage: How to perform shrinkage.
                      None: No shrinkage (default)
                      {"constant", "min_n_obs", "relative"} or a callable
                      * constant: shrunk prediction for a level is weighted average of its prediction and its
                                  parents prediction
                      * min_n_obs: shrunk prediction is the prediction for the smallest group with at least
                                   n observations in it
                      * relative: each group-level is weight according to its size
                      * function: a function that takes a list of group lengths and returns an array of the
                                  same size with the weights for each group
    :param use_global_model: With shrinkage: whether to have a model over the entire input as first group
                             Without shrinkage: whether or not to fall back to a general model in case the group
                             parameter is not found during `.predict()`
    :param **shrinkage_kwargs: keyword arguments to the shrinkage function
    """

    # Number of features in value df can be 0, e.g. for dummy models
    _check_kwargs = {"ensure_min_features": 0, "accept_large_sparse": False}
    _global_col_name = "a-column-that-is-constant-for-all-data"
    _global_col_value = "global"

    def __init__(
        self,
        estimator,
        groups,
        shrinkage=None,
        use_global_model=True,
        **shrinkage_kwargs,
    ):
        self.estimator = estimator
        self.groups = groups
        self.shrinkage = shrinkage
        self.use_global_model = use_global_model
        self.shrinkage_kwargs = shrinkage_kwargs

    def __set_shrinkage_function(self):
        if (
            self.shrinkage
            and len(as_list(self.groups)) == 1
            and not self.use_global_model
        ):
            raise ValueError(
                "Cannot do shrinkage with a single group if use_global_model is False"
            )

        if isinstance(self.shrinkage, str):
            # Predefined shrinkage functions
            shrink_options = {
                "constant": constant_shrinkage,
                "relative": relative_shrinkage,
                "min_n_obs": min_n_obs_shrinkage,
            }

            try:
                self.shrinkage_function_ = shrink_options[self.shrinkage]
            except KeyError:
                raise ValueError(
                    f"The specified shrinkage function {self.shrinkage} is not valid, "
                    f"choose from {list(shrink_options.keys())} or supply a callable."
                )
        elif callable(self.shrinkage):
            self.__check_shrinkage_func()
            self.shrinkage_function_ = self.shrinkage
        else:
            raise ValueError(
                "Invalid shrinkage specified. Should be either None (no shrinkage), str or callable."
            )

    def __check_shrinkage_func(self):
        """Validate the shrinkage function if a function is specified"""
        group_lengths = [10, 5, 2]
        expected_shape = np.array(group_lengths).shape
        try:
            result = self.shrinkage(group_lengths)
        except Exception as e:
            raise ValueError(
                f"Caught an exception while checking the shrinkage function: {str(e)}"
            ) from e
        else:
            if not isinstance(result, np.ndarray):
                raise ValueError(
                    f"shrinkage_function({group_lengths}) should return an np.ndarray"
                )
            if result.shape != expected_shape:
                raise ValueError(
                    f"shrinkage_function({group_lengths}).shape should be {expected_shape}"
                )

    def __get_shrinkage_factor(self, X_group):
        """Get for all complete groups an array of shrinkages"""
        group_colnames = X_group.columns.to_list()
        counts = X_group.groupby(group_colnames).size()

        # Groups that are split on all
        most_granular_groups = [
            grp for grp in self.groups_ if len(as_list(grp)) == len(group_colnames)
        ]

        # For each hierarchy level in each most granular group, get the number of observations
        hierarchical_counts = {
            granular_group: [
                counts[tuple(subgroup)].sum()
                for subgroup in expanding_list(granular_group, tuple)
            ]
            for granular_group in most_granular_groups
        }

        # For each hierarchy level in each most granular group, get the shrinkage factor
        shrinkage_factors = {
            group: self.shrinkage_function_(counts, **self.shrinkage_kwargs)
            for group, counts in hierarchical_counts.items()
        }

        # Make sure that the factors sum to one
        shrinkage_factors = {
            group: value / value.sum() for group, value in shrinkage_factors.items()
        }

        return shrinkage_factors

    def __fit_single_group(self, group, X, y=None):
        try:
            return clone(self.estimator).fit(X, y)
        except Exception as e:
            raise type(e)(f"Exception for group {group}: {e}")

    def __fit_grouped_estimator(self, X_group, X_value, y=None, columns=None):
        # Reset indices such that they are the same in X and y
        if not columns:
            columns = X_group.columns.tolist()

        # Make the groups based on the groups dataframe, use the indices on the values array
        try:
            group_indices = X_group.groupby(columns).indices
        except TypeError:
            # This one is needed because of line #918 of sklearn/utils/estimator_checks
            raise TypeError("argument must be a string, date or number")

        if y is not None:
            if isinstance(y, pd.Series):
                y.index = X_group.index

            grouped_estimators = {
                # Fit a clone of the transformer to each group
                group: self.__fit_single_group(group, X_value[indices, :], y[indices])
                for group, indices in group_indices.items()
            }
        else:
            grouped_estimators = {
                group: self.__fit_single_group(group, X_value[indices, :])
                for group, indices in group_indices.items()
            }

        return grouped_estimators

    def __fit_shrinkage_groups(self, X_group, X_value, y):
        estimators = {}

        for grouping_colnames in self.group_colnames_hierarchical_:
            # Fit a grouped estimator to each (sub)group hierarchically
            estimators.update(
                self.__fit_grouped_estimator(
                    X_group, X_value, y, columns=grouping_colnames
                )
            )

        return estimators

    def __add_shrinkage_column(self, X_group):
        """Add global group as first column if needed for shrinkage"""

        if self.shrinkage is not None and self.use_global_model:
            return pd.concat(
                [
                    pd.Series(
                        [self._global_col_value] * len(X_group),
                        name=self._global_col_name,
                    ),
                    X_group,
                ],
                axis=1,
            )

        return X_group

    def fit(self, X, y=None):
        """
        Fit the model using X, y as training data. Will also learn the groups that exist within the dataset.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :param y: array-like, shape=(n_samples,) training data.
        :return: Returns an instance of self.
        """

        X_group, X_value = _split_groups_and_values(
            X, self.groups, min_value_cols=0, **self._check_kwargs
        )

        X_group = self.__add_shrinkage_column(X_group)

        if y is not None:
            y = check_array(y, ensure_2d=False)

        if self.shrinkage is not None:
            self.__set_shrinkage_function()

        # List of all hierarchical subsets of columns
        self.group_colnames_hierarchical_ = expanding_list(X_group.columns, list)

        self.fallback_ = None

        if self.shrinkage is None and self.use_global_model:
            self.fallback_ = clone(self.estimator).fit(X_value, y)

        if self.shrinkage is not None:
            self.estimators_ = self.__fit_shrinkage_groups(X_group, X_value, y)
        else:
            self.estimators_ = self.__fit_grouped_estimator(X_group, X_value, y)

        self.groups_ = as_list(self.estimators_.keys())

        if self.shrinkage is not None:
            self.shrinkage_factors_ = self.__get_shrinkage_factor(X_group)

        return self

    def __predict_shrinkage_groups(self, X_group, X_value):
        """Make predictions for all shrinkage groups"""
        # DataFrame with predictions for each hierarchy level, per row. Missing groups errors are thrown here.
        hierarchical_predictions = pd.concat(
            [
                pd.Series(self.__predict_groups(X_group, X_value, level_columns))
                for level_columns in self.group_colnames_hierarchical_
            ],
            axis=1,
        )

        # This is a Series with values the tuples of hierarchical grouping
        prediction_groups = X_group.agg(func=tuple, axis=1)

        # This is a Series of arrays
        shrinkage_factors = prediction_groups.map(self.shrinkage_factors_)

        # Convert the Series of arrays it to a DataFrame
        shrinkage_factors = pd.DataFrame.from_dict(shrinkage_factors.to_dict()).T

        return (hierarchical_predictions * shrinkage_factors).sum(axis=1)

    def __predict_single_group(self, group, X):
        """Predict a single group by getting its estimator from the fitted dict"""
        # Keep track of the original index such that we can sort in __predict_groups
        index = X.index
        try:
            group_predictor = self.estimators_[group]
        except KeyError:
            if self.fallback_:
                group_predictor = self.fallback_
            else:
                raise ValueError(
                    f"Found new group {group} during predict with use_global_model = False"
                )

        return pd.DataFrame(group_predictor.predict(X)).set_index(index)

    def __predict_groups(
        self, X_group: pd.DataFrame, X_value: np.array, group_cols=None
    ):
        """Predict for all groups"""
        # Reset indices such that they are the same in X_group (reset in __check_grouping_columns),
        # this way we can track the order of the result
        X_value = pd.DataFrame(X_value).reset_index(drop=True)

        if group_cols is None:
            group_cols = X_group.columns.tolist()

        # Make the groups based on the groups dataframe, use the indices on the values array
        group_indices = X_group.groupby(group_cols).indices

        return (
            pd.concat(
                [
                    self.__predict_single_group(group, X_value.loc[indices, :])
                    for group, indices in group_indices.items()
                ],
                axis=0,
            )
            .sort_index()
            .values.squeeze()
        )

    def predict(self, X):
        """
        Predict on new data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :return: array, shape=(n_samples,) the predicted data
        """

        check_is_fitted(self, ["estimators_", "groups_", "fallback_"])

        X_group, X_value = _split_groups_and_values(
            X, self.groups, min_value_cols=0, **self._check_kwargs
        )

        X_group = self.__add_shrinkage_column(X_group)

        if self.shrinkage is None:
            return self.__predict_groups(X_group, X_value)
        else:
            return self.__predict_shrinkage_groups(X_group, X_value)
