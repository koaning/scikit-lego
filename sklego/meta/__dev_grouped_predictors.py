import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    is_classifier,
    is_regressor,
)
from sklearn.utils.validation import check_is_fitted

from sklego.common import as_list, expanding_list
from sklego.meta._grouped_utils import (
    constant_shrinkage,
    min_n_obs_shrinkage,
    relative_shrinkage,
)


def _get_estimator(estimators, grp_values, grp_names, return_level, fallback_method):
    """
    Recursive function to get the estimator for the given group values.

    Parameters
    ----------
    estimators : dict
        Dictionary with group values as keys and estimators as values.
    grp_values : list
        List of group values - keys to the estimators dictionary.
    grp_names : list
        List of group names
    return_level : int
        The level of the group values to return the estimator for.
    """
    if fallback_method == "raise":
        return estimators[grp_values], return_level
    elif fallback_method == "next":
        try:
            return estimators[grp_values], return_level
        except KeyError:
            if len(grp_values) == 1:
                raise ValueError(
                    f"No fallback/parent estimator found for the given group values: {grp_names}={grp_values}"
                )
            return _get_estimator(estimators, grp_values[:-1], grp_names[:-1], return_level - 1, fallback_method)

    elif fallback_method == "global":
        try:
            return estimators[grp_values], return_level
        except KeyError:
            return estimators[(1,)], 1


class GroupedPredictor(BaseEstimator):
    """Construct an estimator per data group. Splits data by values of a single column and fits one estimator per such
    column.

    Parameters
    ----------
    estimator : scikit-learn compatible estimator/pipeline
        The estimator/pipeline to be applied per group.
    groups : int | str | List[int] | List[str]
        The column(s) of the array/dataframe to select as a grouping parameter set.
    shrinkage : Literal["constant", "min_n_obs", "relative"] | Callable | None, default=None
        How to perform shrinkage:

        - `None`: No shrinkage (default)
        - `"constant"`: shrunk prediction for a level is weighted average of its prediction and its parents prediction
        - `"min_n_obs"`: shrunk prediction is the prediction for the smallest group with at least n observations in it
        - `"relative"`: each group-level is weight according to its size
        - `Callable`: a function that takes a list of group lengths and returns an array of the same size with the
            weights for each group
    use_global_model : bool, default=True

        - With shrinkage: whether to have a model over the entire input as first group
        - Without shrinkage: whether or not to fall back to a general model in case the group parameter is not found
            during `.predict()`
    check_X : bool, default=True
        Whether to validate `X` to be non-empty 2D array of finite values and attempt to cast `X` to float.
        If disabled, the model/pipeline is expected to handle e.g. missing, non-numeric, or non-finite values.
    **shrinkage_kwargs : dict
        Keyword arguments to the shrinkage function
    """

    _check_kwargs = {"ensure_min_features": 0, "accept_large_sparse": False}

    _ALLOWED_SHRINKAGE = {
        "constant": constant_shrinkage,
        "relative": relative_shrinkage,
        "min_n_obs": min_n_obs_shrinkage,
    }
    _ALLOWED_FALLBACK = {"global", "next", "raise"}

    def __init__(
        self,
        estimator,
        groups,
        shrinkage=None,
        use_global_model=True,
        check_X=True,
        fallback_method="raise",
        **shrinkage_kwargs,
    ):
        self.estimator = estimator
        self.groups = groups
        self.shrinkage = shrinkage
        self.use_global_model = use_global_model
        self.check_X = check_X
        self.fallback_method = fallback_method
        self.shrinkage_kwargs = shrinkage_kwargs

    @property
    def _estimator_type(self):
        """Computes `_estimator_type` dynamically from the wrapped model."""
        return self.estimator._estimator_type

    def fit(self, X, y=None):
        # TODO: Params and X,y checks

        self.groups_ = as_list(self.groups)
        frame = pd.DataFrame(X).assign(__target_value__=y)
        frame.index = pd.RangeIndex(start=0, stop=frame.shape[0], step=1)

        if self.use_global_model:
            frame = frame.assign(__global_model__=1)
            self.groups_ = ["__global_model__"] + self.groups_

        self.fitted_levels_ = expanding_list(self.groups_)
        self.n_levels_ = len(self.fitted_levels_)

        self.estimators_ = self.__fit_estimators(frame)
        self.shrinkage_function_ = self.__set_shrinkage_function()
        self.shrinkage_factors_ = self.__fit_shrinkage_factors(frame)
        return self

    def __fit_estimators(self, frame):
        estimators_ = {}
        for grp_names in self.fitted_levels_:
            for grp_values, grp_frame in frame.groupby(grp_names):
                _X = grp_frame.drop(columns=self.groups_ + ["__target_value__"])
                _y = grp_frame["__target_value__"]

                if _y.isnull().all():
                    estimators_[grp_values] = clone(self.estimator).fit(_X)
                else:
                    estimators_[grp_values] = clone(self.estimator).fit(_X, _y)
        return estimators_

    def __fit_shrinkage_factors(self, frame):
        n_groups = len(self.groups_)
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
        return {
            grp_value: np.pad(shrink_array / shrink_array.sum(), (0, n_groups - shrink_array.size))
            for grp_value, shrink_array in shrinkage_factors.items()
        }

    # def __set_fit_levels(self):
    #     """Based on the combination of parameters passed to the class, it defines the groups/levels that were fitted.
    #
    #     This function should be called only after assigning self.groups_ during fit.
    #
    #     We have a few combinations:
    #     - `fallback_level = 0`, `use_global_model = True`: then one global model is fitted as well as the highest
    #         level of granularity for the groups passed.
    #     - `fallback_level = 0`, `use_global_model = False`: then only the highest level of granularity is fitted for the
    #         groups passed.
    #     - `fallback_level = 1`: then all levels of granularity for the groups passed.
    #     """
    #     check_is_fitted(self, ["groups_"])
    #     if self.fallback_level == 0 and self.use_global_model:
    #         levels_ = [["__global_model__"], self.groups_]
    #     elif self.fallback_level == 0 and not self.use_global_model:
    #         levels_ = self.groups_
    #     elif self.fallback_level == 1:
    #         levels_ = expanding_list(self.groups_)
    #     else:
    #         raise ValueError(f"`fallback_level` should be 0 or 1, not {self.fallback_level}")
    #     return levels_

    def __set_shrinkage_function(self):
        if self.shrinkage and len(as_list(self.groups)) == 1 and not self.use_global_model:
            raise ValueError("Cannot do shrinkage with a single group and `use_global_model=False`")

        if self.shrinkage in self._ALLOWED_SHRINKAGE.keys():
            shrinkage_function_ = self._ALLOWED_SHRINKAGE[self.shrinkage]

        elif callable(self.shrinkage):
            self.__check_shrinkage_func()
            shrinkage_function_ = self.shrinkage

        elif self.shrinkage is None:
            shrinkage_function_ = lambda x: np.pad(np.zeros(len(x) - 1), pad_width=(0, 1), constant_values=1)

        else:
            raise ValueError(f"`shrinkage` should be either `None`, {self._ALLOWED_SHRINKAGE.keys()}, or a callable")
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


class GroupedClassifier(GroupedPredictor, ClassifierMixin):
    def fit(self, X, y):
        if not is_classifier(self.estimator):
            raise ValueError(f"estimator should be a classifier, not {self.estimator._estimator_type}")

        self.classes_ = np.sort(np.unique(y))
        self.n_classes_ = len(self.classes_)

        super().fit(X, y)
        return self

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

    def predict_proba(self, X):
        check_is_fitted(self, ["estimators_", "groups_"])

        frame = pd.DataFrame(X)
        frame.index = pd.RangeIndex(start=0, stop=frame.shape[0], step=1)

        if self.use_global_model:
            frame = frame.assign(__global_model__=1)

        raw_preds = np.zeros((X.shape[0], self.n_levels_, self.n_classes_), dtype=float)
        shrinkage = np.zeros((X.shape[0], self.n_levels_, self.n_levels_), dtype=float)

        for level, grp_names in enumerate(self.fitted_levels_):
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
                raw_preds[np.ix_(grp_idx, [level], _estimator.classes_)] = _estimator.predict_proba(
                    grp_frame.drop(columns=self.groups_)
                )[:, None, :]
                shrinkage[np.ix_(grp_idx, [level])] = _shrinkage_factor

        return (raw_preds * shrinkage[:, -1, :][:, :, None]).sum(axis=1)


class GroupedRegressor(GroupedPredictor, ClassifierMixin):
    def fit(self, X, y):
        if not is_regressor(self.estimator):
            raise ValueError(f"estimator should be a regressor, not {self.estimator._estimator_type}")
        super().fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self, ["estimators_", "groups_"])

        frame = pd.DataFrame(X)
        frame.index = pd.RangeIndex(start=0, stop=frame.shape[0], step=1)

        if self.use_global_model:
            frame = frame.assign(__global_model__=1)

        raw_preds = np.empty((X.shape[0], self.n_levels_), dtype=float)
        shrinkage = np.empty((X.shape[0], self.n_levels_, self.n_levels_), dtype=float)

        for level, grp_names in enumerate(self.fitted_levels_):
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
                raw_preds[grp_idx, level : level + 1] = _estimator.predict(grp_frame.drop(columns=self.groups_))[
                    :, None
                ]
                shrinkage[grp_idx, level : level + 1] = _shrinkage_factor

        preds = np.average(raw_preds, axis=1, weights=shrinkage[:, -1, :])
        return preds
