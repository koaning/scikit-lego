from copy import deepcopy
from typing import List, Union

import narwhals.stable.v1 as nw
import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, RegressorMixin, is_classifier, is_regressor
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from sklego.common import as_list, expanding_list
from sklego.meta._grouped_utils import parse_X_y
from sklego.meta._shrinkage_utils import (
    ShrinkageMixin,
    constant_shrinkage,
    equal_shrinkage,
    min_n_obs_shrinkage,
    relative_shrinkage,
)


class GroupedPredictor(ShrinkageMixin, MetaEstimatorMixin, BaseEstimator):
    """`GroupedPredictor` is a meta-estimator that fits a separate estimator for each group in the input data.

    The input data is split into a group and a value part: for each unique combination of the group columns, a separate
    estimator is fitted to the corresponding value rows. The group columns are specified by the `groups` parameter.

    If `use_global_model=True` a fallback estimator will be fitted on the entire dataset in case a group is not found
    during `.predict()`.

    If `shrinkage` is not `None`, the predictions of the group-level models are combined using a shrinkage method. The
    shrinkage method can be one of the predefined methods `"constant"`, `"equal"`, `"min_n_obs"`, `"relative"` or a
    custom shrinkage function. The shrinkage method is specified by the `shrinkage` parameter.

    !!! warning "Shrinkage"
        Shrinkage is only available for regression models.

    Parameters
    ----------
    estimator : scikit-learn compatible estimator/pipeline
        The estimator/pipeline to be applied per group.
    groups : int | str | List[int] | List[str]
        The column(s) of the array/dataframe to select as a grouping parameter set.
    shrinkage : Literal["constant", "equal", "min_n_obs", "relative"] | Callable | None, default=None
        How to perform shrinkage:

        - `None`: No shrinkage (default)
        - `"constant"`: the augmented prediction for each level is the weighted average between its prediction and the
            augmented prediction for its parent.
        - `"equal"`: each group is weighed equally.
        - `"min_n_obs"`: use only the smallest group with a certain amount of observations.
        - `"relative"`: weigh each group according to its size.
        - `Callable`: a function that takes a list of group lengths and returns an array of the same size with the
            weights for each group.
    use_global_model : bool, default=True

        - With shrinkage: whether to have a model over the entire input as first group
        - Without shrinkage: whether or not to fall back to a general model in case the group parameter is not found
            during `.predict()`
    check_X : bool, default=True
        Whether to validate `X` to be non-empty 2D array of finite values and attempt to cast `X` to float.
        If disabled, the model/pipeline is expected to handle e.g. missing, non-numeric, or non-finite values.
    **shrinkage_kwargs : dict
        Keyword arguments to the shrinkage function

    Attributes
    ----------
    estimators_ : dict
        A dictionary with the fitted estimators per group
    groups_ : list
        A list of all the groups that were found during fitting
    fallback_ : estimator
        A fallback estimator that is used when `use_global_model=True` and a group is not found during `.predict()`
    shrinkage_function_ : callable
        The shrinkage function that is used to calculate the shrinkage factors
    shrinkage_factors_ : dict
        A dictionary with the shrinkage factors per group

    Example
    -------
    ```py
    import pandas as pd
    from sklearn.dummy import DummyRegressor
    from sklego.meta import GroupedPredictor

    results_df = pd.DataFrame(
        {
            "Grade": ["11", "11", "11", "11", "12", "12", "12", "12"],
            "Course": ["Algebra", "Algebra", "English", "English", "Algebra", "Algebra", "English", "English"],
            "Name": ["Mary", "Helen", "Mary", "Helen", "Mary", "Helen", "Mary", "Helen"],
            "Result": [100, 94, 88, 92, 96, 98, 90, 90],
        }
    )

    groups = ["Grade", "Name"]
    target = "Result"

    # We will use the DummyRegressor() to calculate the mean of each grouping
    grouped_pred = GroupedPredictor(DummyRegressor(), groups)
    grouped_pred.fit(results_df[groups], results_df[target])

    # What is the average for Mary in Grade 12?
    pred_df = pd.DataFrame(
        {
            "Grade": ["12"],
            "Name": ["Mary"],
        }
    )

    # Predicts the mean result of each student in each grade
    pred = grouped_pred.predict(pred_df)

    print(f"The average result of {pred_df["Name"][0]} in Grade {pred_df["Grade"][0]} was {pred}")
    ### The average result of Mary in Grade 12 was 93.0
    ```
    """

    # Number of features in value df can be 0, e.g. for dummy models
    _check_kwargs = {"ensure_min_features": 0, "accept_large_sparse": False}
    _global_col_name = "a-column-that-is-constant-for-all-data"
    _global_col_value = "global"

    _ALLOWED_SHRINKAGE = {
        "constant": constant_shrinkage,
        "relative": relative_shrinkage,
        "min_n_obs": min_n_obs_shrinkage,
        "equal": equal_shrinkage,
    }

    _required_parameters = ["estimator", "groups"]

    def __init__(
        self,
        estimator,
        groups,
        shrinkage=None,
        use_global_model=True,
        check_X=True,
        shrinkage_kwargs=None,
    ):
        self.estimator = estimator
        self.groups = groups
        self.shrinkage = shrinkage
        self.use_global_model = use_global_model
        self.shrinkage_kwargs = shrinkage_kwargs
        self.check_X = check_X

    def __fit_single_group(self, group, X, y=None):
        """Fit estimator to the given group."""
        try:
            return clone(self.estimator).fit(X, y)
        except Exception as e:
            raise type(e)(f"Exception for group {group}: {e}")

    def __fit_grouped_estimator(
        self, frame: nw.DataFrame, y: Union[np.ndarray, None] = None, columns: Union[List[int], List[str], None] = None
    ):
        """Fit an estimator to each group"""

        if columns is None:
            columns = self._groups

        to_drop = list(set(["__sklego_target__", *columns, *as_list(self.groups)]))
        grouped_estimators = {
            # Fit a clone of the estimators to each group
            (group_name[0] if len(group_name) == 1 else group_name): self.__fit_single_group(
                group=(group_name[0] if len(group_name) == 1 else group_name),
                X=nw.to_native(X_grp.drop(to_drop)),
                y=(X_grp.select("__sklego_target__").to_numpy().reshape(-1) if y is not None else None),
            )
            for group_name, X_grp in frame.group_by(columns)
        }

        return grouped_estimators

    def __fit_shrinkage_groups(self, frame, y):
        estimators = {}

        for grouping_colnames in self.group_colnames_hierarchical_:
            # Fit a grouped estimator to each (sub)group hierarchically
            estimators.update(self.__fit_grouped_estimator(frame, y, columns=grouping_colnames))

        return estimators

    def __add_shrinkage_column(self, frame, groups=None):
        """Add global group as first column if needed for shrinkage"""

        if self.shrinkage is not None and self.use_global_model:
            n_samples = frame.shape[0]

            frame = frame.select(
                nw.from_dict(
                    data={self._global_col_name: np.full(shape=n_samples, fill_value=self._global_col_value)},
                    native_namespace=nw.get_native_namespace(frame),
                )[self._global_col_name],
                nw.all(),
            )
            groups = [self._global_col_name] if groups is None else [self._global_col_name, *groups]

        return frame, groups

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
        if self.shrinkage is not None and not is_regressor(self.estimator):
            raise ValueError("Shrinkage is only available for regression models")

        _group_cols = as_list(deepcopy(self.groups)) if self.groups is not None else None

        if (
            self.shrinkage is not None
            and _group_cols is not None
            and len(_group_cols) == 1
            and not self.use_global_model
        ):
            raise ValueError("Shrinkage is not null, but found a total of 1 groups")

        X = nw.from_native(X, strict=False, eager_only=True)

        frame = parse_X_y(X, y, _group_cols, check_X=self.check_X, **self._check_kwargs)
        frame, _group_cols = self.__add_shrinkage_column(frame, _group_cols)
        self.n_features_in_ = frame.shape[1] - 1
        self.n_fitted_levels_ = 1 + self.use_global_model

        self.shrinkage_function_ = self._set_shrinkage_function()

        # List of all hierarchical subsets of columns
        self.group_colnames_hierarchical_ = expanding_list(_group_cols, list)
        self.fallback_ = None

        if self.shrinkage is None and self.use_global_model:
            X_ = nw.to_native(frame.drop([*_group_cols, "__sklego_target__"]))
            y_ = nw.to_native(frame["__sklego_target__"])

            self.fallback_ = clone(self.estimator).fit(X_, y_)

        if self.shrinkage is not None:
            self.estimators_ = self.__fit_shrinkage_groups(frame, y)
        else:
            self.estimators_ = self.__fit_grouped_estimator(frame, y, columns=_group_cols)

        self.groups_ = as_list(self.estimators_.keys())

        if self.shrinkage is not None:
            _groups = (
                [self._global_col_name, *as_list(deepcopy(self.groups))]
                if self.use_global_model
                else as_list(deepcopy(self.groups))
            )

            self.shrinkage_factors_ = self._fit_shrinkage_factors(frame, groups=_groups, most_granular_only=True)
            self.shrinkage_factors_ = {(k[0] if len(k) == 1 else k): v for k, v in self.shrinkage_factors_.items()}

        return self

    def __predict_shrinkage_groups(self, frame, method="predict", groups=None):
        """Make predictions for all shrinkage groups"""
        # DataFrame with predictions for each hierarchy level, per row. Missing groups errors are thrown here.
        hierarchical_predictions = pd.concat(
            [
                pd.Series(self.__predict_groups(frame, method=method, groups=level_columns))
                for level_columns in self.group_colnames_hierarchical_
            ],
            axis=1,
        )

        # This is a Series with values the tuples of hierarchical grouping
        prediction_groups = pd.Series([tuple(_) for _ in frame.select(groups).to_pandas().itertuples(index=False)])

        # This is a Series of arrays
        shrinkage_factors = prediction_groups.map(self.shrinkage_factors_)

        # Convert the Series of arrays it to a DataFrame
        shrinkage_factors = pd.DataFrame.from_dict(shrinkage_factors.to_dict()).T

        return (hierarchical_predictions * shrinkage_factors).sum(axis=1)

    def __predict_single_group(self, group, X, method="predict"):
        """Predict a single group by getting its estimator from the fitted dict"""

        try:
            group_predictor = self.estimators_[group]
        except KeyError:
            if self.fallback_:
                group_predictor = self.fallback_
            else:
                raise ValueError(f"Found new group {group} during predict with use_global_model = False")

        is_predict_proba = is_classifier(group_predictor) and method == "predict_proba"
        # Ensure to provide pd.DataFrame with the correct label name
        extra_kwargs = {"columns": group_predictor.classes_} if is_predict_proba else {}

        # getattr(group_predictor, method) returns the predict method of the fitted model
        # if the method argument is "predict" and the predict_proba method if method argument is "predict_proba"
        return pd.DataFrame(getattr(group_predictor, method)(X), **extra_kwargs)

    def __predict_groups(self, frame: nw.DataFrame, method="predict", groups=None):
        """Predict for all groups"""

        n_samples = frame.shape[0]
        frame = frame.with_columns(__sklego_index__=np.arange(n_samples))
        return (
            pd.concat(
                [
                    self.__predict_single_group(
                        (group_value[0] if len(group_value) == 1 else group_value),
                        nw.to_native(X_grp.drop(["__sklego_index__", *groups, *as_list(self.groups)])),
                        method=method,
                    ).set_index(X_grp["__sklego_index__"].to_numpy().reshape(-1).astype(int))
                    for group_value, X_grp in frame.group_by(groups)
                ],
                axis=0,
            )
            .fillna(0)
            .sort_index()
            .to_numpy()
            .squeeze()
        )

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
        check_is_fitted(self, ["estimators_", "groups_", "fallback_"])

        _group_cols = as_list(deepcopy(self.groups)) if self.groups is not None else None
        X = nw.from_native(X, strict=False, eager_only=True)
        frame = parse_X_y(X, y=None, groups=_group_cols, check_X=self.check_X, **self._check_kwargs).drop(
            "__sklego_target__"
        )
        frame, _group_cols = self.__add_shrinkage_column(frame, _group_cols)

        if self.shrinkage is None:
            return self.__predict_groups(frame, method="predict", groups=_group_cols)
        else:
            return self.__predict_shrinkage_groups(frame, method="predict", groups=_group_cols)

    # This ensures that the meta-estimator only has the predict_proba method if the estimator has it
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
        check_is_fitted(self, ["estimators_", "groups_", "fallback_"])

        _group_cols = as_list(deepcopy(self.groups)) if self.groups is not None else None
        X = nw.from_native(X, strict=False, eager_only=True)
        frame = parse_X_y(X, y=None, groups=_group_cols, check_X=self.check_X, **self._check_kwargs).drop(
            "__sklego_target__"
        )
        frame, _group_cols = self.__add_shrinkage_column(frame, _group_cols)

        if self.shrinkage is None:
            return self.__predict_groups(frame, method="predict_proba", groups=_group_cols)
        else:
            return self.__predict_shrinkage_groups(frame, method="predict_proba", groups=_group_cols)

    # This ensures that the meta-estimator only has the predict_proba method if the estimator has it
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
        check_is_fitted(self, ["estimators_", "groups_", "fallback_"])

        _group_cols = as_list(deepcopy(self.groups)) if self.groups is not None else None
        X = nw.from_native(X, strict=False, eager_only=True)

        frame = parse_X_y(X, y=None, groups=_group_cols, check_X=self.check_X, **self._check_kwargs).drop(
            "__sklego_target__"
        )
        frame, _group_cols = self.__add_shrinkage_column(frame, _group_cols)

        if self.shrinkage is None:
            return self.__predict_groups(frame, method="decision_function", groups=_group_cols)
        else:
            return self.__predict_shrinkage_groups(frame, method="decision_function", groups=_group_cols)

    @property
    def _estimator_type(self):
        """Computes `_estimator_type` dynamically from the wrapped model."""
        return self.estimator._estimator_type

    def _more_tags(self):
        return {"allow_nan": True}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


class GroupedRegressor(RegressorMixin, GroupedPredictor):
    """`GroupedRegressor` is a meta-estimator that fits a separate regressor for each group in the input data.

    Its spec is the same as [`GroupedPredictor`][sklego.meta.grouped_predictor.GroupedPredictor] but it is available
    only for regression models.

    !!! info "New in version 0.8.0"
    """

    def fit(self, X, y):
        """Fit one regressor for each group of training data `X` and `y`.

        Will also learn the groups that exist within the training dataset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : GroupedRegressor
            The fitted regressor.

        Raises
        -------
        ValueError
            If the supplied estimator is not a regressor.

        """
        if not is_regressor(self.estimator):
            raise ValueError("GroupedRegressor is only available for regression models")

        return super().fit(X, y)


class GroupedClassifier(ClassifierMixin, GroupedPredictor):
    """`GroupedClassifier` is a meta-estimator that fits a separate classifier for each group in the input data.

    Its equivalent to [`GroupedPredictor`][sklego.meta.grouped_predictor.GroupedPredictor] with `shrinkage=None`
    but it is available only for classification models.

    !!! info "New in version 0.8.0"
    """

    def __init__(
        self,
        estimator,
        groups,
        use_global_model=True,
        check_X=True,
        **shrinkage_kwargs,
    ):
        super().__init__(
            estimator=estimator,
            groups=groups,
            shrinkage=None,
            use_global_model=use_global_model,
            check_X=check_X,
        )

    def fit(self, X, y):
        """Fit one classifier for each group of training data `X` and `y`.

        Will also learn the groups that exist within the training dataset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : GroupedClassifier
            The fitted classifier.

        Raises
        -------
        ValueError
            If the supplied estimator is not a classifier.
        """

        if not is_classifier(self.estimator):
            raise ValueError("GroupedClassifier is only available for classification models")
        self.classes_ = np.unique(y)
        return super().fit(X, y)
