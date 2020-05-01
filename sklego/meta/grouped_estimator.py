import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.utils.validation import (
    check_is_fitted,
    check_X_y,
    check_array,
)

from sklego.common import as_list, expanding_list


def constant_shrinkage(group_sizes: list, alpha: float) -> np.ndarray:
    r"""
    The augmented prediction for each level is the weighted average between its prediction and the augmented
    prediction for its parent.

    Let $\hat{y}_i$ be the prediction at level $i$, with $i=0$ being the root, than the augmented prediction
    $\hat{y}_i^* = \alpha \hat{y}_i + (1 - \alpha) \hat{y}_{i-1}^*$, with $\hat{y}_0^* = \hat{y}_0$.
    """
    return np.array(
        [alpha ** (len(group_sizes) - 1)]
        + [
            alpha ** (len(group_sizes) - 1 - i) * (1 - alpha)
            for i in range(1, len(group_sizes) - 1)
        ]
        + [(1 - alpha)]
    )


def relative_shrinkage(group_sizes: list) -> np.ndarray:
    """Weigh each group according to it's size"""
    return np.array(group_sizes)


def min_n_obs_shrinkage(group_sizes: list, min_n_obs) -> np.ndarray:
    """Use only the smallest group with a certain amount of observations"""
    if min_n_obs > max(group_sizes):
        raise ValueError(
            f"There is no group with size greater than or equal to {min_n_obs}"
        )

    res = np.zeros(len(group_sizes))
    res[np.argmin(np.array(group_sizes) >= min_n_obs) - 1] = 1
    return res


class GroupedEstimator(BaseEstimator):
    """
    Construct an estimator per data group. Splits data by values of a
    single column and fits one estimator per such column.

    :param estimator: the model/pipeline to be applied per group
    :param groups: the column(s) of the matrix/dataframe to select as a grouping parameter set
    :param value_columns: Columns to use in the prediction. If None (default), use all non-grouping columns
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

    def __init__(
        self,
        estimator,
        groups,
        value_columns=None,
        shrinkage=None,
        use_global_model=True,
        **shrinkage_kwargs,
    ):
        self.estimator = estimator
        self.groups = groups
        self.value_columns = value_columns
        self.shrinkage = shrinkage
        self.use_global_model = use_global_model
        self.shrinkage_kwargs = shrinkage_kwargs

    def __set_shrinkage_function(self):
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
                f"Invalid shrinkage specified. Should be either None (no shrinkage), str or callable."
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

    @staticmethod
    def __check_cols_exist(X, cols):
        """Check whether the specified grouping columns are in X"""
        if X.shape[1] == 0:
            raise ValueError(
                f"0 feature(s) (shape=({X.shape[0]}, 0)) while a minimum of 1 is required."
            )

        # X has been converted to a DataFrame
        x_cols = set(X.columns)
        diff = set(as_list(cols)) - x_cols

        if len(diff) > 0:
            raise ValueError(f"{diff} not in columns of X {x_cols}")

    @staticmethod
    def __check_missing_and_inf(X):
        """Check that all elements of X are non-missing and finite, needed because check_array cannot handle strings"""
        if np.any(pd.isnull(X)):
            raise ValueError("X has NaN values")
        try:
            if np.any(np.isinf(X)):
                raise ValueError("X has infinite values")
        except TypeError:
            # if X cannot be converted to numeric, checking infinites does not make sense
            pass

    def __validate(self, X, y=None):
        """Validate the input, used in both fit and predict"""
        if (
            self.shrinkage
            and len(as_list(self.groups)) == 1
            and not self.use_global_model
        ):
            raise ValueError(
                "Cannot do shrinkage with a single group if use_global_model is False"
            )

        self.__check_cols_exist(X, self.value_colnames_)
        self.__check_cols_exist(X, self.group_colnames_)

        # Split the model data from the grouping columns, this part is checked `regularly`
        X_data = X.loc[:, self.value_colnames_]

        # y can be None because __validate used in predict, X can have no columns if the estimator only uses y
        if X_data.shape[1] > 0 and y is not None:
            check_X_y(X_data, y, multi_output=True)
        elif y is not None:
            check_array(y, ensure_2d=False)
        elif X_data.shape[1] > 0:
            check_array(X_data)

        self.__check_missing_and_inf(X)

    def __fit_grouped_estimator(self, X, y, value_columns, group_columns):
        # Reset indices such that they are the same in X and y
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)

        group_indices = X.groupby(group_columns).indices

        grouped_estimations = {
            group: clone(self.estimator).fit(
                X.loc[indices, value_columns], y.loc[indices]
            )
            for group, indices in group_indices.items()
        }

        return grouped_estimations

    def __get_shrinkage_factor(self, X):
        """Get for all complete groups an array of shrinkages"""
        counts = X.groupby(self.group_colnames_).size()

        # Groups that are split on all
        most_granular_groups = [
            grp
            for grp in self.groups_
            if len(as_list(grp)) == len(self.group_colnames_)
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

    def __prepare_input_data(self, X, y=None):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[str(_) for _ in range(X.shape[1])])

        if self.shrinkage is not None and self.use_global_model:
            global_col = "a-column-that-is-constant-for-all-data"
            X = X.assign(**{global_col: "global"})
            self.groups = [global_col] + as_list(self.groups)

        if y is not None:
            if isinstance(y, np.ndarray):
                pred_col = (
                    "the-column-that-i-want-to-predict-but-dont-have-the-name-for"
                )
                cols = (
                    pred_col
                    if y.ndim == 1
                    else ["_".join([pred_col, i]) for i in range(y.shape[1])]
                )
                y = (
                    pd.Series(y, name=cols)
                    if y.ndim == 1
                    else pd.DataFrame(y, columns=cols)
                )

            return X, y

        return X

    def fit(self, X, y=None):
        """
        Fit the model using X, y as training data. Will also learn the groups that exist within the dataset.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :param y: array-like, shape=(n_samples,) training data.
        :return: Returns an instance of self.
        """
        X, y = self.__prepare_input_data(X, y)

        if self.shrinkage is not None:
            self.__set_shrinkage_function()

        self.group_colnames_ = [str(_) for _ in as_list(self.groups)]

        if self.value_columns is not None:
            self.value_colnames_ = [str(_) for _ in as_list(self.value_columns)]
        else:
            self.value_colnames_ = [
                _ for _ in X.columns if _ not in self.group_colnames_
            ]
        self.__validate(X, y)

        # List of all hierarchical subsets of columns
        self.group_colnames_hierarchical_ = expanding_list(self.group_colnames_, list)

        self.fallback_ = None

        if self.shrinkage is None and self.use_global_model:
            subset_x = X[self.value_colnames_]
            self.fallback_ = clone(self.estimator).fit(subset_x, y)

        if self.shrinkage is not None:
            self.estimators_ = {}

            for level_colnames in self.group_colnames_hierarchical_:
                self.estimators_.update(
                    self.__fit_grouped_estimator(
                        X, y, self.value_colnames_, level_colnames
                    )
                )
        else:
            self.estimators_ = self.__fit_grouped_estimator(
                X, y, self.value_colnames_, self.group_colnames_
            )

        self.groups_ = as_list(self.estimators_.keys())

        if self.shrinkage is not None:
            self.shrinkage_factors_ = self.__get_shrinkage_factor(X)

        return self

    def __predict_group(self, X, group_colnames):
        """Make predictions for all groups"""
        try:
            return (
                X.groupby(group_colnames, as_index=False)
                .apply(
                    lambda d: pd.DataFrame(
                        self.estimators_.get(d.name, self.fallback_).predict(
                            d[self.value_colnames_]
                        ),
                        index=d.index,
                    )
                )
                .values.squeeze()
            )
        except AttributeError:
            # Handle new groups
            culprits = set(X[self.group_colnames_].agg(func=tuple, axis=1)) - set(
                self.estimators_.keys()
            )

            if self.shrinkage is not None and self.use_global_model:
                # Remove the global group from the culprits because the user did not specify
                culprits = {culprit[1:] for culprit in culprits}

            raise ValueError(
                f"found a group(s) {culprits} in `.predict` that was not in `.fit`"
            )

    def __predict_shrinkage_groups(self, X):
        """Make predictions for all shrinkage groups"""
        # DataFrame with predictions for each hierarchy level, per row. Missing groups errors are thrown here.
        hierarchical_predictions = pd.concat(
            [
                pd.Series(self.__predict_group(X, level_columns))
                for level_columns in self.group_colnames_hierarchical_
            ],
            axis=1,
        )

        # This is a Series with values the tuples of hierarchical grouping
        prediction_groups = X[self.group_colnames_].agg(func=tuple, axis=1)

        # This is a Series of arrays
        shrinkage_factors = prediction_groups.map(self.shrinkage_factors_)

        # Convert the Series of arrays it to a DataFrame
        shrinkage_factors = pd.DataFrame.from_dict(shrinkage_factors.to_dict()).T

        return (hierarchical_predictions * shrinkage_factors).sum(axis=1)

    def predict(self, X):
        """
        Predict on new data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :return: array, shape=(n_samples,) the predicted data
        """
        X = self.__prepare_input_data(X)
        self.__validate(X)

        check_is_fitted(
            self,
            [
                "estimators_",
                "groups_",
                "group_colnames_",
                "value_colnames_",
                "fallback_",
            ],
        )

        if self.shrinkage is None:
            return self.__predict_group(X, group_colnames=self.group_colnames_)
        else:
            return self.__predict_shrinkage_groups(X)
