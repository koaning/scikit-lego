from typing import Union

import narwhals as nw
import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted

from sklego.common import as_list
from sklego.meta._grouped_utils import parse_X_y


class GroupedTransformer(TransformerMixin, MetaEstimatorMixin, BaseEstimator):
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
    _required_parameters = ["transformer", "groups"]

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

    def __fit_grouped_transformer(self, frame: nw.DataFrame, y: Union[np.ndarray, None]):
        """Fit a transformer to each group"""

        grouped_transformers = {
            # Fit a clone of the transformer to each group
            group_name: self.__fit_single_group(
                group_name,
                X=nw.to_native(X_grp.drop(["__sklego_target__", *self.groups_])),
                y=(nw.to_native(X_grp["__sklego_target__"]) if y is not None else None),
            )
            for group_name, X_grp in frame.group_by(self.groups_)
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
        self.groups_ = as_list(self.groups) if self.groups is not None else None

        X = nw.from_native(X, strict=False, eager_only=True)
        if not isinstance(X, nw.DataFrame) and self.groups_ is not None:
            # Accounts for negative indices if X is an array
            self.groups_ = [
                X.shape[1] + group if isinstance(group, int) and group < 0 else group for group in self.groups_
            ]

        frame = parse_X_y(X, y, self.groups_, check_X=self.check_X, **self._check_kwargs)

        if self.groups is None:
            X_, y_ = (
                nw.to_native(frame.drop("__sklego_target__")),
                nw.to_native(frame["__sklego_target__"]) if y is not None else None,
            )
            self.transformers_ = clone(self.transformer).fit(X_, y=y_)
            return self

        self.transformers_ = self.__fit_grouped_transformer(frame, y)

        if self.use_global_model:
            X_, y_ = (
                nw.to_native(frame.drop(["__sklego_target__", *self.groups_])),
                nw.to_native(frame["__sklego_target__"]) if y is not None else None,
            )
            self.fallback_ = clone(self.transformer).fit(X_, y_)

        self.n_features_in_ = X.shape[1]
        return self

    def __transform_single_group(self, group, X):
        """Transform a single group by getting its transformer from the fitted dict"""
        try:
            group_transformer = self.transformers_[group]
        except KeyError:
            if self.fallback_:
                group_transformer = self.fallback_
            else:
                raise ValueError(f"Found new group {group} during transform with use_global_model = False")

        return np.asarray(group_transformer.transform(X))

    def __transform_groups(self, frame: nw.DataFrame):
        """Transform all groups"""

        n_samples = frame.shape[0]
        frame = frame.with_columns(__sklego_index__=np.arange(n_samples))

        results = [
            (
                nw.to_native(X_grp.select("__sklego_index__")).to_numpy().squeeze().astype(int),
                self.__transform_single_group(
                    group_name, nw.to_native(X_grp.drop(["__sklego_index__", *self.groups_]))
                ),
            )
            for group_name, X_grp in frame.group_by(self.groups_)
        ]

        output = np.zeros(shape=(n_samples, results[0][1].shape[1]))
        for grp_index, grp_result in results:
            output[grp_index, :] = grp_result

        return output

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

        X = nw.from_native(X, strict=False, eager_only=True)
        frame = parse_X_y(X, y=None, groups=self.groups_, check_X=self.check_X, **self._check_kwargs).drop(
            "__sklego_target__"
        )

        if self.groups is None:
            X_ = nw.to_native(frame)
            return self.transformers_.transform(X_)

        return self.__transform_groups(frame)

    def _more_tags(self):
        return {"allow_nan": True}
