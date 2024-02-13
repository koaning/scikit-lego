import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.base import BaseEstimator, is_classifier
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_array, check_is_fitted

from sklego.common import as_list, expanding_list
from sklego.meta._grouped_utils import _get_estimator, constant_shrinkage, min_n_obs_shrinkage, relative_shrinkage


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
    fallback_method : Literal["global", "next", "raise"], default="global"
        Defines which fallback strategy to use if a group is not found at prediction time:

        - "global": use global model to make the prediction, it requires to have `use_global_model=True` flag.
        - "next": if `groups` has length more than 1, then it fallback to the first available "parent".
            Example: let `groups=["a", "b"]` with values `(0, 0)`, `(0, 1)` and `(1, 0)`. If we try to predict the group
            value `(0,2)`, we fallback to the model trained on `a=0` since there is no model trained on `(a=0, b=2)`.
        - "raise": if a group value is not found an error is raised.
    **shrinkage_kwargs : dict[str, Any]
        Keyword arguments to the shrinkage function

    Attributes
    ----------
    estimators_ : dict[tuple, scikit-learn compatible estimator/pipeline]
        Dictionary with group values as keys and estimators as values.
    groups_ : list[str] | list[int]
        The list of group names/indexes
    fitted_levels_ : list[list[str] | list[int]]
        The list of group names/indexes that were fitted
    shrinkage_function_ : Callable
        The shrinkage function that was used
    shrinkage_factors_ : dict[tuple, np.ndarray]
        Dictionary with group values as keys and shrinkage factors as values for all fitted levels
    classes_ : np.ndarray
        The classes of the target variable, applicable only for classification tasks
    n_classes_ : int
        The number of classes of the target variable, applicable only for classification tasks
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
        fallback_method="global",
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
        """Fit one estimator for each group of training data `X` and `y`.

        Will also learn the groups that exist within the dataset.

        If `use_global_model=True` a fallback estimator will be fitted on the entire dataset in case a group is not
        found during `.predict()`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), default=None
            Target values.

        Returns
        -------
        self : GroupedPredictor
            The fitted estimator.
        """

        # TODO: Validate class params?

        if is_classifier(self.estimator):
            self.classes_ = np.sort(np.unique(y))  # TODO: Must be sequential for the rest of the code to work
            self.n_classes_ = len(self.classes_)

        self.groups_ = as_list(self.groups)

        # TODO: __grouped_predictor_target_value__?
        frame = pd.DataFrame(X).assign(__target_value__=np.array(y)).reset_index(drop=True)

        self.__validate_inputs(frame)

        if self.use_global_model:
            # TODO: __grouped_predictor_global_model__?
            frame = frame.assign(__global_model__=1)
            self.groups_ = ["__global_model__"] + self.groups_

        self.fitted_levels_ = self.__set_fit_levels()

        self.estimators_ = self.__fit_estimators(frame)
        self.shrinkage_function_ = self.__set_shrinkage_function()
        self.shrinkage_factors_ = self.__fit_shrinkage_factors(frame)
        return self

    def predict(self, X):
        """Predict target values on new data `X` by predicting on each group. If a group is not found during
        `.predict()` and `use_global_model=True` the fallback estimator will be used. If `use_global_model=False` a
        `ValueError` will be raised.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict.

        Returns
        -------
        array-like of shape (n_samples,)
            Predicted target values.
        """
        if is_classifier(self.estimator):
            preds = self.__predict_estimators(X, method_name="predict_proba")
            return self.classes_[np.argmax(preds, axis=1)]
        else:
            preds = self.__predict_estimators(X, method_name="predict")
            return preds

    @available_if(lambda self: hasattr(self.estimator, "predict_proba"))
    def predict_proba(self, X):
        """Predict probabilities on new data `X`.

        !!! warning
            Available only if the underlying estimator implements `.predict_proba()` method.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict.

        Returns
        -------
        array-like of shape (n_samples, n_classes)
            Predicted probabilities per class.
        """
        return self.__predict_estimators(X, method_name="predict_proba")

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
        return self.__predict_estimators(X, method_name="decision_function")

    def __validate_inputs(self, frame):
        """Validate the input arrays"""

        if frame.shape[1] <= len(self.groups_) + 1:
            raise ValueError("`X` contains no features")

        if self.check_X:
            X_values = frame.drop(columns=self.groups_ + ["__target_value__"]).copy()
            check_array(X_values, **self._check_kwargs)

        X_groups = frame.loc[:, self.groups_].copy()

        X_group_num = X_groups.select_dtypes(include="number")
        if X_group_num.shape[1]:
            check_array(X_group_num, **self._check_kwargs)

        # Only check missingness in object columns
        if X_groups.select_dtypes(exclude="number").isnull().any(axis=None):
            raise ValueError("Group columns contain NaN values")

        return self

    def __set_fit_levels(self):
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

    def __fit_estimators(self, frame):
        """Fit one estimator per level of the group column(s)"""
        estimators_ = {}
        for grp_names in self.fitted_levels_:
            for grp_values, grp_frame in frame.groupby(grp_names):
                _X = grp_frame.drop(columns=self.groups_ + ["__target_value__"])
                _y = grp_frame["__target_value__"]

                if _y.isnull().all():
                    estimators_[grp_values] = clone(self.estimator).fit(_X)  # TODO: or .fit(_X, None) ?
                else:
                    estimators_[grp_values] = clone(self.estimator).fit(_X, _y)

        return estimators_

    def __fit_shrinkage_factors(self, frame):
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
        return {
            grp_value: np.pad(shrink_array / shrink_array.sum(), (0, max(0, self.n_levels_ - shrink_array.size)))
            for grp_value, shrink_array in shrinkage_factors.items()
        }

    def __set_shrinkage_function(self):
        """Set the shrinkage function and validate it if it is a custom callable"""

        if self.shrinkage and len(as_list(self.groups)) == 1 and not self.use_global_model:
            raise ValueError("Cannot do shrinkage with a single group and `use_global_model=False`")

        if self.shrinkage in self._ALLOWED_SHRINKAGE.keys():
            shrinkage_function_ = self._ALLOWED_SHRINKAGE[self.shrinkage]

        elif callable(self.shrinkage):
            self.__check_shrinkage_func()
            shrinkage_function_ = self.shrinkage

        elif self.shrinkage is None:
            shrinkage_function_ = lambda x: np.pad(np.zeros(self.n_levels_ - 1), pad_width=(0, 1), constant_values=1)

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

    def __predict_estimators(self, X, method_name):
        """Predict on each level and apply shrinkage if necessary"""
        check_is_fitted(self, ["estimators_", "groups_"])

        frame = pd.DataFrame(X).reset_index(drop=True)

        if self.use_global_model:
            frame = frame.assign(__global_model__=1)

        depth = getattr(self, "n_classes_", 1)

        preds = np.zeros((X.shape[0], self.n_levels_, depth), dtype=float)
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
