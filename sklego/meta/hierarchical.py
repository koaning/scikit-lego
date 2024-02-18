from warnings import warn

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn import clone
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MetaEstimatorMixin,
    RegressorMixin,
    is_classifier,
    is_regressor,
)
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_array, check_is_fitted

from sklego.common import as_list, expanding_list
from sklego.meta._grouped_utils import constant_shrinkage, min_n_obs_shrinkage, relative_shrinkage


def is_transformer(estimator):
    """Check if an estimator is a transformer."""
    return hasattr(estimator, "transform")


def _get_estimator(estimators, grp_values, grp_names, return_level, fallback_method):
    """Recursive function to get the estimator for the given group values.

    Parameters
    ----------
    estimators : dict[tuple, scikit-learn compatible estimator/pipeline]
        Dictionary with group values as keys and estimators as values.
    grp_values : tuple
        List of group values - keys to the estimators dictionary.
    grp_names : list
        List of group names, used only if `fallback_method="raise"`.
    return_level : int
        The level of the group values to return the estimator for.
    fallback_method : Literal["parent", "raise"]
        Defines which fallback strategy to use if a group is not found at prediction time.

    Returns
    -------
    estimator : scikit-learn compatible estimator/pipeline
        The estimator for the given group values.
    return_level : int
        The level of the group values for which the estimator was found.

    Raises
    ------
    KeyError
        If `fallback_method="raise"` and no fallback/parent estimator is found for the given group values.
    """
    try:
        return estimators[grp_values], return_level
    except KeyError:
        if fallback_method == "parent":
            return _get_estimator(estimators, grp_values[:-1], grp_names[:-1], return_level - 1, fallback_method)
        else:  # fallback_method == "raise"
            raise KeyError(f"No fallback/parent estimator found for the given group values: {grp_names}={grp_values}")


class HierarchicalPredictor(MetaEstimatorMixin, BaseEstimator):
    _CHECK_KWARGS = {
        "ensure_min_features": 0,
        "accept_large_sparse": False,
    }
    _ALLOWED_SHRINKAGE = {
        "constant": constant_shrinkage,
        "relative": relative_shrinkage,
        "min_n_obs": min_n_obs_shrinkage,
    }
    _ALLOWED_FALLBACK = {"parent", "raise"}

    _GLOBAL_NAME = "__sklego_global_estimator__"
    _TARGET_NAME = "__skelgo_target_value__"

    def __init__(
        self,
        estimator,
        groups,
        *,
        shrinkage=None,
        fallback_method="parent",
        n_jobs=None,
        check_X=True,
        **shrinkage_kwargs,
    ):
        self.estimator = estimator
        self.groups = groups
        self.shrinkage = shrinkage
        self.fallback_method = fallback_method
        self.n_jobs = n_jobs
        self.check_X = check_X
        self.shrinkage_kwargs = shrinkage_kwargs

    @property
    def _estimator_type(self):
        """Computes `_estimator_type` dynamically from the wrapped model."""
        return self.estimator._estimator_type

    def fit(self, X, y=None):
        """Fit one estimator for each hierarchical group of training data `X` and `y`.

        Will also learn the groups that exist within the training dataset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), default=None
            Target values, if applicable.

        Returns
        -------
        self : BaseHierarchicalEstimator
            The fitted estimator.

        Raises
        -------
        ValueError
            - If `check_X` is not a boolean.
            - If group columns contain NaN values.
            - If `shrinkage` is not one of `None`, `"constant"`, `"min_n_obs"`, `"relative"`, or a callable.
            - If `fallback_method` is not `"parent"` or `"raise"`.
        """

        if self.fallback_method not in self._ALLOWED_FALLBACK:
            raise ValueError(f"`fallback_method` should be either `parent` or `raise`. Found {self.fallback_method}")

        if not isinstance(self.check_X, bool):
            raise ValueError(f"`check_X` should be a boolean. Found {type(self.check_X)}")

        self.groups_ = [self._GLOBAL_NAME] + as_list(self.groups)
        self.fitted_levels_ = expanding_list(self.groups_)
        self.shrinkage_function_ = self._set_shrinkage_function()  # If invalid shrinkage, will raise ValueError

        frame = (
            pd.DataFrame(X)
            .assign(**{self._TARGET_NAME: np.array(y), self._GLOBAL_NAME: 1})
            .reset_index(drop=True)
            .pipe(self.__validate_frame)
        )

        self.estimators_ = self._fit_estimators(frame)
        self.shrinkage_factors_ = self._fit_shrinkage_factors(frame)

        self.n_groups_ = len(self.groups_)
        self.n_features_ = frame.shape[1] - self.n_groups_ - 1
        self.n_features_in_ = frame.shape[1] - 2  # target and global columns
        self.n_levels_ = len(self.fitted_levels_)

        return self

    def _inference(self, X, method_name):
        """Calls `method_name` on each level and apply shrinkage if necessary"""
        check_is_fitted(self, ["estimators_", "groups_"])

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X should have {self.n_features_in_} features, got {X.shape[1]}")

        frame = pd.DataFrame(X).reset_index(drop=True).assign(**{self._GLOBAL_NAME: 1})

        if not is_classifier(self.estimator):
            # regressor or outlier detector
            n_out = 1
        else:
            if self.n_classes_ > 2 or method_name == "predict_proba":
                n_out = self.n_classes_
            else:
                # binary case with `method_name = "decision_function"`
                n_out = 1

        preds = np.zeros((X.shape[0], self.n_levels_, n_out), dtype=float)
        shrinkage = np.zeros((X.shape[0], self.n_levels_), dtype=float)

        for level_idx, grp_names in enumerate(self.fitted_levels_):
            for grp_values, grp_frame in frame.groupby(grp_names):
                grp_idx = grp_frame.index

                _estimator, _level = _get_estimator(
                    estimators=self.estimators_,
                    grp_values=grp_values,
                    grp_names=grp_names,
                    return_level=len(grp_names),
                    fallback_method=self.fallback_method,
                )
                _shrinkage_factor = self.shrinkage_factors_[grp_values[:_level]]

                last_dim_ix = _estimator.classes_ if is_classifier(self.estimator) else [0]

                raw_pred = getattr(_estimator, method_name)(grp_frame.drop(columns=self.groups_))

                preds[np.ix_(grp_idx, [level_idx], last_dim_ix)] = np.atleast_3d(raw_pred[:, None])
                shrinkage[np.ix_(grp_idx)] = _shrinkage_factor

        return (preds * np.atleast_3d(shrinkage)).sum(axis=1).squeeze()

    def _fit_single_estimator(self, grp_frame):
        """Shortcut to fit an estimator on a single group"""
        _X = grp_frame.drop(columns=self.groups_ + [self._TARGET_NAME])
        _y = grp_frame[self._TARGET_NAME]
        return clone(self.estimator).fit(_X, _y)

    def _fit_estimators(self, frame):
        """Fits one estimator per level of the group column(s), and returns a dictionary of the fitted estimators.

        The keys of the dictionary are the group values, and the values are the fitted estimators.
        The if-else block is used to parallelize the fitting process if `n_jobs` is greater than 1.
        """
        # Question: Should the `estimators_` keys be named tuples instead of plain tuples?
        if self.n_jobs is None or self.n_jobs == 1:
            estimators_ = {
                grp_values: self._fit_single_estimator(grp_frame)
                for grp_names in self.fitted_levels_
                for grp_values, grp_frame in frame.groupby(grp_names)
            }
        else:
            fit_func = lambda grp_values, grp_frame: (grp_values, self._fit_single_estimator(grp_frame))

            estimators_ = dict(
                Parallel(n_jobs=self.n_jobs)(
                    delayed(fit_func)(grp_values, grp_frame)
                    for grp_names in self.fitted_levels_
                    for grp_values, grp_frame in frame.groupby(grp_names)
                )
            )

        return estimators_

    def _set_shrinkage_function(self):
        """Set the shrinkage function and validate it if it is a custom callable"""
        if self.shrinkage in self._ALLOWED_SHRINKAGE.keys():
            shrinkage_function_ = self._ALLOWED_SHRINKAGE[self.shrinkage]

        elif callable(self.shrinkage):
            self.__check_shrinkage_func()
            shrinkage_function_ = self.shrinkage

        elif self.shrinkage is None:
            """Instead of keeping two different behaviors for shrinkage and non-shrinkage cases, this conditional block
            maps no shrinkage to a constant shrinkage function, wit  all the weight on the grouped passed,
            independently from the level sizes, as expected from the other shrinkage functions (*).
            This allows the rest of the code to be agnostic to the shrinkage function, and the shrinkage factors.

            (*) Consider the following example:

            - groups = ["a", "b"] with values (0, 0), (0, 1) and (1, 0) of respective sizes 6, 5, 9.
            - Considering these sizes, in `__fit_shrinkage_factors` the hierarchical_counts will be:
                - (1, 0, 0): [20, 11, 6]
                - (1, 0, 1): [20, 11, 5]
                - (1, 1, 0): [20, 9, 9]

                Notice that we always have the same total count (20), and the shrinkage factors will reflect that.
            - For `shrinkage = "relative"`, we get the following shrinkage factors:
                {
                    (1,): array([1.]),
                    (1, 0): array([0.64, 0.35]),
                    (1, 1): array([0.69, 0.31]),
                    (1, 0, 0): array([0.54, 0.30 , 0.16]),
                    (1, 0, 1): array([0.56, 0.30, 0.14]),
                    (1, 1, 0): array([0.52, 0.24, 0.24])
                }
            - For `shrinkage = None`, we get the following shrinkage factors:
                {
                    (1,): array([1., 0., 0.]),
                    (1, 0): array([0., 1., 0.]),
                    (1, 1): array([0., 1., 0.]),
                    (1, 0, 0): array([0., 0., 1.]),
                    (1, 0, 1): array([0., 0., 1.]),
                    (1, 1, 0): array([0., 0., 1.])
                }
            """

            def no_shrinkage_function(x):
                n = len(self.fitted_levels_[-1])
                return np.lib.pad([1], (len(x) - 1, n - len(x)), "constant", constant_values=(0))

            shrinkage_function_ = no_shrinkage_function

        else:
            raise ValueError(
                f"`shrinkage` should be either `None`, {self._ALLOWED_SHRINKAGE.keys()}, or a callable. "
                f"Found {self.shrinkage} of type {type(self.shrinkage)}"
            )
        return shrinkage_function_

    def __check_shrinkage_func(self):
        """Validate the shrinkage function if a function is specified"""
        group_lengths = [10, 5, 2]
        expected_shape = np.array(group_lengths).shape
        try:
            result = self.shrinkage(group_lengths)
        except Exception as e:
            raise ValueError(f"Caught an exception while checking the shrinkage function: {str(e)}") from e
        else:
            if not isinstance(result, np.ndarray):
                raise ValueError(f"shrinkage_function({group_lengths}) should return an np.ndarray")
            if result.shape != expected_shape:
                raise ValueError(f"shrinkage_function({group_lengths}).shape should be {expected_shape}")

    def _fit_shrinkage_factors(self, frame):
        """Computes the shrinkage coefficients for all fitted levels (corresponding to the keys of self.estimators_)"""

        check_is_fitted(self, ["estimators_", "groups_"])
        counts = frame.groupby(self.groups_).size().rename("counts")
        all_grp_values = list(self.estimators_.keys())

        hierarchical_counts = {
            grp_value: [counts.loc[subgroup].sum() for subgroup in expanding_list(grp_value, tuple)]
            for grp_value in all_grp_values
        }

        shrinkage_factors = {
            grp_value: self.shrinkage_function_(counts, **self.shrinkage_kwargs)
            for grp_value, counts in hierarchical_counts.items()
        }

        # Normalize and pad
        return {grp_value: shrink_array / shrink_array.sum() for grp_value, shrink_array in shrinkage_factors.items()}

    def __validate_frame(self, frame):
        """Validate the input arrays"""

        if self.check_X:
            X_values = frame.drop(columns=self.groups_ + [self._TARGET_NAME]).copy()
            check_array(X_values, **self._CHECK_KWARGS)

        X_groups = frame.loc[:, self.groups_].copy()

        X_group_num = X_groups.select_dtypes(include="number")
        if X_group_num.shape[1]:
            check_array(X_group_num, **self._CHECK_KWARGS)

        # Only check missingness in object columns
        if X_groups.select_dtypes(exclude="number").isnull().any(axis=None):
            raise ValueError("Group columns contain NaN values")

        return frame


class HierarchicalRegressor(HierarchicalPredictor, RegressorMixin):
    def fit(self, X, y):
        if not is_regressor(self.estimator):
            raise ValueError("The supplied estimator should be a regressor")

        super().fit(X, y)
        return self

    def predict(self, X):
        return self._inference(X, "predict")


class HierarchicalClassifier(HierarchicalPredictor, ClassifierMixin):
    def fit(self, X, y):
        if not is_classifier(self.estimator):
            raise ValueError("The supplied estimator should be a classifier")

        if not hasattr(self.estimator, "predict_proba"):
            raise ValueError("The supplied estimator should have a 'predict_proba' method")

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        super().fit(X, y)
        return self

    def predict(self, X):
        preds = self._inference(X, method_name="predict_proba")
        return self.classes_[np.argmax(preds, axis=1)]

    def predict_proba(self, X):
        """Predict probabilities on new data `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict.

        Returns
        -------
        array-like of shape (n_samples, n_classes)
            Predicted probabilities per class.
        """
        return self._inference(X, method_name="predict_proba")

    @available_if(lambda self: hasattr(self.estimator, "decision_function"))
    def decision_function(self, X):
        """Predict confidence scores for samples in `X`.

        !!! warning
            Available only if the underlying estimator implements `.decision_function()` method.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict.

        Returns
        -------
        array-like of shape (n_samples,) or (n_samples, n_classes)
            Confidence scores per (n_samples, n_classes) combination.
            In the binary case, confidence score for self.classes_[1] where > 0 means this class would be
            predicted.
        """
        warn(
            "`decision_function` will lead to inconsistent results in cases where the estimators are not all fitted "
            "on the same target values.",
            UserWarning,
        )
        return self.__predict_estimators(X, method_name="decision_function")


# class HierarchicalTransformer(TransformerMixin):
#
#     def fit(self, X, y=None):
#         if not is_transformer(self.estimator):
#             raise ValueError("The supplied transformer should have a 'transform' method")
#         ...
#         return self
#
#     def transform(self, X):
#         return self._inference(X, "transform")
