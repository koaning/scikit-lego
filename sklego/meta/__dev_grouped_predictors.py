import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.base import (
    BaseEstimator,
    is_classifier,
)
from sklearn.utils.metaestimators import available_if
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

    @property
    def n_levels_(self):
        check_is_fitted(self, ["fitted_levels_"])
        return len(self.fitted_levels_)

    def fit(self, X, y=None):
        # TODO: Params and X,y checks

        if is_classifier(self.estimator):
            self.classes_ = np.sort(np.unique(y))
            self.n_classes_ = len(self.classes_)

        self.groups_ = as_list(self.groups)
        frame = pd.DataFrame(X).assign(__target_value__=y)
        frame.index = pd.RangeIndex(start=0, stop=frame.shape[0], step=1)

        if self.use_global_model:
            frame = frame.assign(__global_model__=1)
            self.groups_ = ["__global_model__"] + self.groups_

        self.fitted_levels_ = self.__get_fit_levels()  # expanding_list(self.groups_)

        self.estimators_ = self.__fit_estimators(frame)
        self.shrinkage_function_ = self.__set_shrinkage_function()
        self.shrinkage_factors_ = self.__fit_shrinkage_factors(frame)
        return self

    def predict(self, X):
        preds = self.__predict_estimators(X)

        if is_classifier(self.estimator):
            return self.classes_[np.argmax(preds, axis=1)]
        else:
            return preds.squeeze()

    @available_if(lambda self: hasattr(self.estimator, "predict_proba"))
    def predict_proba(self, X):
        return self.__predict_estimators(X)

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

    def __predict_estimators(self, X):
        check_is_fitted(self, ["estimators_", "groups_"])

        frame = pd.DataFrame(X)
        frame.index = pd.RangeIndex(start=0, stop=frame.shape[0], step=1)

        if self.use_global_model:
            frame = frame.assign(__global_model__=1)

        depth = getattr(self, "n_classes_", 1)

        preds = np.zeros((X.shape[0], self.n_levels_, depth), dtype=float)
        shrinkage = np.zeros((X.shape[0], self.n_levels_, self.n_levels_), dtype=float)

        predict_method = "predict_proba" if is_classifier(self.estimator) else "predict"
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

                last_dim_ix = _estimator.classes_ if is_classifier(self.estimator) else [0]
                raw_pred = getattr(_estimator, predict_method)(grp_frame.drop(columns=self.groups_))
                _shrinkage_factor = self.shrinkage_factors_[grp_values[:_level]]
                preds[np.ix_(grp_idx, [level], last_dim_ix)] = np.atleast_3d(raw_pred[:, None])
                shrinkage[np.ix_(grp_idx, [level])] = _shrinkage_factor

        return (preds * shrinkage[:, -1, :][:, :, None]).sum(axis=1)

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

    def __get_fit_levels(self):
        """Based on the combination of parameters passed to the class, it defines the groups/levels that were fitted.

        This function should be called only after assigning self.groups_ during fit.
        """
        check_is_fitted(self, ["groups_"])

        if self.fallback_method == "raise":
            levels_ = self.groups_ if self.shrinkage is None else expanding_list(self.groups_)

        elif self.fallback_method == "next":
            levels_ = expanding_list(self.groups_)

        elif self.fallback_method == "global":
            if not self.use_global_model:
                raise ValueError("`fallback_method`='global' requires `use_global_model=True`")
            elif self.shrinkage is None:
                levels_ = [["__global_model__"], self.groups_]
            else:
                levels_ = expanding_list(self.groups_)

        else:
            raise ValueError(f"`fallback_method` should be one of {self._ALLOWED_FALLBACK}, not {self.fallback_method}")

        return levels_
