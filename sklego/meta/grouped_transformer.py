import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted

from sklego.meta._grouped_utils import _split_groups_and_values


class GroupedTransformer(BaseEstimator, TransformerMixin):
    """Construct a transformer per data group. Splits data by groups from single or multiple columns and transforms
    remaining columns using the transformers corresponding to the groups.

    Parameters
    ----------
    transformer : scikit-learn compatible transformer
        The transformer to be applied per group.
    groups : int | str | List[int] | List[str] | None
        The column(s) of the array/dataframe to select as a grouping parameter set. If `None`, the transformer will be
        applied to the entire input without grouping.
    use_global_model : bool, default=True
        Whether or not to fall back to a general transformation in case a group is not found during `.transform()`.
    check_X : bool, default=True
        Whether or not to check the input data. If False, the checks are delegated to the wrapped estimator.

    Attributes
    ----------
    transformers_ : scikit-learn compatible transformer | dict[..., scikit-learn compatible transformer]
        The fitted transformers per group or a single fitted transformer if `groups` is `None`.
    fallback_ : scikit-learn compatible transformer | None
        The fitted transformer to fall back to in case a group is not found during `.transform()`. Only present if
        `use_global_model` is `True`.
    """

    _check_kwargs = {"accept_large_sparse": False}

    def __init__(self, transformer, groups, use_global_model=True, check_X=True):
        self.transformer = transformer
        self.groups = groups
        self.use_global_model = use_global_model
        self.check_X = check_X

    def __fit_single_group(self, group, X, y=None):
        """Fit transformer to the given group.

        Parameters
        ----------
        group : tuple
            The group to fit the transformer to.
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), default=None
            Target values.

        Returns
        -------
        transformer : scikit-learn compatible transformer
            The fitted transformer for the group.
        """
        try:
            return clone(self.transformer).fit(X, y)
        except Exception as e:
            raise type(e)(f"Exception for group {group}: {e}")

    def __fit_grouped_transformer(self, X_group: pd.DataFrame, X_value: np.ndarray, y=None):
        """Fit a transformer to each group"""
        # Make the groups based on the groups dataframe, use the indices on the values array
        try:
            group_indices = X_group.groupby(X_group.columns.tolist()).indices
        except TypeError:
            # This one is needed because of line #918 of sklearn/utils/estimator_checks
            raise TypeError("argument must be a string, date or number")

        if y is not None:
            if isinstance(y, pd.Series):
                y.index = X_group.index

            grouped_transformers = {
                # Fit a clone of the transformer to each group
                group: self.__fit_single_group(group, X_value[indices, :], y[indices])
                for group, indices in group_indices.items()
            }
        else:
            grouped_transformers = {
                group: self.__fit_single_group(group, X_value[indices, :]) for group, indices in group_indices.items()
            }

        return grouped_transformers

    def __check_transformer(self):
        """Check if the supplied transformer has a `transform` method and raise a `ValueError` if not."""
        if not hasattr(self.transformer, "transform"):
            raise ValueError("The supplied transformer should have a 'transform' method")

    def fit(self, X, y=None):
        """Fit one transformer for each group of training data `X`.

        Will also learn the groups that exist within the dataset.

        If `use_global_model=True` a fallback transformer will be fitted on the entire dataset in case a group is not
        found during `.transform()`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. If `groups` is not `None`, X should have at least two columns, of which at least one
            corresponds to groups defined in `groups`, and the remaining columns represent the values to transform.
        y : array-like of shape (n_samples,), default=None
            Target values.

        Returns
        -------
        self : GroupedTransformer
            The fitted transformer.
        """
        self.__check_transformer()

        self.fallback_ = None

        if self.groups is None:
            self.transformers_ = clone(self.transformer).fit(X, y)
            return self

        X_group, X_value = _split_groups_and_values(X, self.groups, check_X=self.check_X, **self._check_kwargs)
        self.transformers_ = self.__fit_grouped_transformer(X_group, X_value, y)

        if self.use_global_model:
            self.fallback_ = clone(self.transformer).fit(X_value, y)

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
                raise ValueError(f"Found new group {group} during transform with use_global_model = False")

        return pd.DataFrame(group_transformer.transform(X)).set_index(index)

    def __transform_groups(self, X_group: pd.DataFrame, X_value: np.ndarray):
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
            .to_numpy()
        )

    def transform(self, X):
        """Transform new data `X` by transforming on each group. If a group is not found during `.transform()` and
        `use_global_model=True` the fallback transformer will be used. If `use_global_model=False` a `ValueError` will
        be raised.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        array-like of shape (n_samples, n_features)
            Data transformed per group.
        """
        check_is_fitted(self, ["fallback_", "transformers_"])

        if self.groups is None:
            return self.transformers_.transform(X)

        X_group, X_value = _split_groups_and_values(X, self.groups, **self._check_kwargs)

        return self.__transform_groups(X_group, X_value)
