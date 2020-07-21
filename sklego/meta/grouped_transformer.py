import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from sklego.common import as_list


class GroupedTransformer(BaseEstimator, TransformerMixin):
    """
    Construct a transformer per data group. Splits data by groups from single or multiple columns
    and transforms remaining columns using the transformers corresponding to the groups.

    :param transformer: the transformer to be applied per group
    :param groups: the column(s) of the matrix/dataframe to select as a grouping parameter set
    :param use_global_model: Whether or not to fall back to a general transformation in case a group
                             is not found during `.transform()`
    """

    _check_kwargs = {"accept_large_sparse": False}

    def __init__(self, transformer, groups=0, use_global_model=True):
        self.transformer = transformer
        self.groups = groups
        self.use_global_model = use_global_model

    def __check_value_columns(self, X):
        """Do basic checks on the value columns"""
        try:
            if isinstance(X, pd.DataFrame):
                X_value = X.drop(columns=self.groups).values
            else:
                X_value = np.delete(X, as_list(self.groups), axis=1)
        except Exception:
            # Check if we can leverage check_array for standard exceptions
            check_array(X, **self._check_kwargs)
            raise ValueError(f"Could not drop groups {self.groups} from columns of X")

        return check_array(X_value, **self._check_kwargs)

    def __check_grouping_columns(self, X):
        """Do basic checks on grouping columns"""
        if isinstance(X, pd.DataFrame):
            X_group = X.loc[:, as_list(self.groups)]
        else:
            X_group = pd.DataFrame(X[:, as_list(self.groups)])

        # Do regular checks on numeric columns
        X_group_num = X_group.select_dtypes(include="number")
        if X_group_num.shape[1]:
            check_array(X_group.select_dtypes(include="number"), **self._check_kwargs)

        # Only check missingness in object columns
        if X_group.select_dtypes(exclude="number").isnull().any(axis=None):
            raise ValueError("X has NaN values")

        # The grouping part we always want as a DataFrame with range index
        return X_group.reset_index(drop=True)

    def __fit_single_group(self, group, X, y=None):
        try:
            return clone(self.transformer).fit(X, y)
        except Exception as e:
            raise type(e)(f"Exception for group {group}: {e}")

    def __fit_grouped_transformer(
        self, X_group: pd.DataFrame, X_value: np.array, y=None
    ):
        """Fit a transformer to each group"""
        # Make the groups based on the groups dataframe, use the indices on the values array
        try:
            group_indices = X_group.groupby(X_group.columns.tolist()).indices
        except TypeError:
            # This one is needed because of line #918 of sklearn/utils/estimator_checks
            raise TypeError("argument must be a string, date or number")

        if y:
            grouped_transformers = {
                # Fit a clone of the transformer to each group
                group: self.__fit_single_group(group, X_value[indices, :], y[indices])
                for group, indices in group_indices.items()
            }
        else:
            grouped_transformers = {
                group: self.__fit_single_group(group, X_value[indices, :])
                for group, indices in group_indices.items()
            }

        return grouped_transformers

    def __check_transformer(self):
        if not hasattr(self.transformer, "transform"):
            raise ValueError(
                "The supplied transformer should have a 'transform' method"
            )

    def fit(self, X, y=None):
        """
        Fit the transformers to the groups in X

        :param X: Array-like with at least two columns, of which at least one corresponds to groups defined in init,
                  and the remaining columns represent the values to transform.
        :param y: (Optional) target variable
        """
        self.__check_transformer()

        X_value = self.__check_value_columns(X)
        X_group = self.__check_grouping_columns(X)

        self.fallback_ = None

        if self.use_global_model:
            self.fallback_ = clone(self.transformer).fit(X_value)

        self.transformers_ = self.__fit_grouped_transformer(X_group, X_value)

        return self

    def __transform_single_group(self, group, X):
        """Transform a single group by getting its transformer from the fitted dict"""
        # Keep track of the original index such that we can sort in __transform_groups
        index = X.index
        try:
            group_transformer = self.transformers_[group]
        except KeyError:
            if self.fallback_:
                group_transformer = self.fallback_
            else:
                raise ValueError(
                    f"Found new group {group} during transform with use_global_model = False"
                )

        return pd.DataFrame(group_transformer.transform(X)).set_index(index)

    def __transform_groups(self, X_group: pd.DataFrame, X_value: np.array):
        """Transform all groups"""
        # Reset indices such that they are the same in X_group (reset in __check_grouping_columns),
        # this way we can track the order of the result
        X_value = pd.DataFrame(X_value).reset_index(drop=True)

        # Make the groups based on the groups dataframe, use the indices on the values array
        group_indices = X_group.groupby(X_group.columns.tolist()).indices

        return (
            pd.concat(
                [
                    self.__transform_single_group(group, X_value.loc[indices, :])
                    for group, indices in group_indices.items()
                ],
                axis=0,
            )
            .sort_index()
            .values
        )

    def transform(self, X):
        """
        Fit the transformers to the groups in X

        :param X: Array-like with columns corresponding to the ones in .fit()
        """
        check_is_fitted(self, ["fallback_", "transformers_"])

        X_value = self.__check_value_columns(X)
        X_group = self.__check_grouping_columns(X)

        return self.__transform_groups(X_group, X_value)
