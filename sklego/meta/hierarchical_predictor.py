from warnings import warn

import narwhals.stable.v1 as nw
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
from sklearn.utils.validation import check_is_fitted
from sklearn_compat.utils.validation import check_array

from sklego.common import as_list, expanding_list
from sklego.meta._grouped_utils import _data_format_checks, _validate_groups_values
from sklego.meta._shrinkage_utils import (
    ShrinkageMixin,
    constant_shrinkage,
    equal_shrinkage,
    min_n_obs_shrinkage,
    relative_shrinkage,
)


def _get_estimator(estimators, grp_values, grp_names, return_level, fallback_method):
    """Recursive function to get the estimator for the given group values.

    Parameters
    ----------
    estimators : dict[tuple[Any, ...], scikit-learn compatible estimator/pipeline]
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
        If `fallback_method="raise"` and no estimator is found for the given group values.
    """
    try:
        return estimators[grp_values], return_level
    except KeyError:
        if fallback_method == "parent":
            return _get_estimator(estimators, grp_values[:-1], grp_names[:-1], return_level - 1, fallback_method)

        # fallback_method == "raise" case
        raise KeyError(f"No estimator found for the given group values: {grp_names}={grp_values}")


class HierarchicalPredictor(ShrinkageMixin, MetaEstimatorMixin, BaseEstimator):
    """`HierarchicalPredictor` is a meta-estimator that fits a separate estimator for each group in the input data
    in a hierarchical manner. This means that an estimator is fitted for each level of the group columns.

    The only exception to that is when `shrinkage=None` **and** `fallback_method="raise"`, in which case only
    one estimator per group value is fitted.

    If `shrinkage` is not `None`, the predictions of the group-level models are combined using a shrinkage method. The
    shrinkage method can be one of the predefined methods `"constant"`, `"equal"`, `"min_n_obs"`, `"relative"` or a
    custom shrinkage function.

    !!! question "Differences with `GroupedPredictor`"

        There are two main differences between `HierarchicalPredictor` and
        [`GroupedPredictor`][sklego.meta.grouped_predictor.GroupedPredictor]:

        1. The first difference is the fallback method: `HierarchicalPredictor` has a fallback method that can be set to
            `"parent"` or `"raise"`. If set to `"parent"`, the estimator will recursively fall back to the parent group
            in case the group value is not found during `.predict()`.

            As a consequence of this:

            - **`groups` order matters!**
            - Potentially a combinatoric number of estimators are fitted, one for each unique combination of group
                values and each level.

        2. `HierarchicalPredictor` is meant to properly handle shrinkage in classification tasks. However this
            **requires** that the estimator has a `.predict_proba()` method.

    !!! warning "Inheritance"

        This class is not meant to be used directly, but to be inherited by a specific hierarchical predictor, such as
        `HierarchicalRegressor` or `HierarchicalClassifier`, which properly implement the `.predict()` and
        `predict`-like methods for the specific task.

    !!! info "New in version 0.8.0"

    Parameters
    ----------
    estimator : scikit-learn compatible estimator/pipeline
        The base estimator to be used for each level.
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
    fallback_method : Literal["parent", "raise"], default="parent"
        The fallback strategy to use if a group is not found at prediction time:

        - "parent": recursively fall back to the parent group in case the group value is not found during `.predict()`.
            It requires to fit a model on each level, including a global model.
        - "raise": raise a KeyError if the group value is not found during `.predict()`.
    n_jobs : int | None, default=None
        The number of jobs to run in parallel. The same convention of [`joblib.Parallel`](https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html)
        holds:

        - `n_jobs = None`: interpreted as n_jobs=1.
        - `n_jobs > 0`: n_cpus=n_jobs are used.
        - `n_jobs < 0`: (n_cpus + 1 + n_jobs) are used.
    check_X : bool, default=True
        Whether to validate `X` to be non-empty 2D array of finite values and attempt to cast `X` to float.
        If disabled, the model/pipeline is expected to handle e.g. missing, non-numeric, or non-finite values.
    shrinkage_kwargs : dict
        Keyword arguments to the shrinkage function

    Attributes
    ----------
    estimators_ : dict[tuple[Any,...], scikit-learn compatible estimator/pipeline]
        Fitted estimators for each level. The keys are the group values, and the values are the fitted estimators.
        The group values are tuples of the group columns, including the global column which has a fixed placeholder
        value of 1.

        Let's say we have two group columns, `col_1` and `col_2`. `col_1` has values 'A' and 'B', and `col_2` has
        values 'X', ... Then `estimators_` dictionary will look something like this:

        ```py
        {
            # global estimator
            (1,): LinearRegression(),

            # estimator for `col_1 = 'A'`
            (1, 'A'): LinearRegression(),

            # estimator for `col_1 = 'B'`
            (1, 'B'): LinearRegression(),

            # estimator for `col_1 = 'A'`, `col_2 = 'X'`
            (1, 'A', 'X'): LinearRegression(),
            ...
        }
        ```
    shrinkage_function_ : callable
        The shrinkage function that is used to calculate the shrinkage factors
    shrinkage_factors_ : dict[tuple[Any,...], np.ndarray]
        Shrinkage factors applied to each level.

        The keys are the group values, and the values are the shrinkage factors. The group values are tuples of the
        group columns, including the global column which has a fixed placeholder value of 1.
    groups_ : list
        List of all group columns including a global column.
    n_groups_ : int
        Number of unique groups.
    n_features_in_ : int
        Number of features in the training data.
    n_features_ : int
        Number of features used by the estimators.
    n_fitted_levels_  : int
        Number of hierarchical levels in the grouping.
    """

    _CHECK_KWARGS = {
        "ensure_min_features": 0,
        "accept_large_sparse": False,
    }
    _ALLOWED_SHRINKAGE = {
        "constant": constant_shrinkage,
        "relative": relative_shrinkage,
        "min_n_obs": min_n_obs_shrinkage,
        "equal": equal_shrinkage,
    }
    _ALLOWED_FALLBACK = {"parent", "raise"}

    _GLOBAL_NAME = "__sklego_global_estimator__"
    _TARGET_NAME = "__sklego_target_value__"
    _INDEX_NAME = "__sklego_index__"

    _required_parameters = ["estimator", "groups"]

    def __init__(
        self,
        estimator,
        groups,
        *,
        shrinkage=None,
        fallback_method="parent",
        n_jobs=None,
        check_X=True,
        shrinkage_kwargs=None,
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

        self.groups_ = [self._GLOBAL_NAME, *as_list(self.groups)]

        # The only case in which we don't have to fit multiple levels is when shrinkage is None and fallback_method is 'raise'
        self.fitted_levels_ = expanding_list(self.groups_)
        self.n_fitted_levels_ = len(self.fitted_levels_)
        # If invalid shrinkage, will raise ValueError (before fitting all the estimators!)
        self.shrinkage_function_ = self._set_shrinkage_function()

        _data_format_checks(X)

        X = nw.from_native(X, strict=False, eager_only=True)
        if not isinstance(X, nw.DataFrame):
            X = nw.from_native(pd.DataFrame(X))

        n_samples, self.n_features_in_ = X.shape

        if n_samples < 2:
            msg = f"Found {n_samples} sample or less, while a minimum of 2 is required."
            raise ValueError(msg)

        if self.n_features_in_ < 1:
            msg = "Found 0 features, while a minimum of 1 if required."
            raise ValueError(msg)

        native_namespace = nw.get_native_namespace(X)
        target_series = nw.new_series(name=self._TARGET_NAME, values=y, native_namespace=native_namespace)
        global_series = nw.new_series(
            name=self._GLOBAL_NAME, values=np.ones(n_samples), native_namespace=native_namespace
        )
        if len(target_series) != n_samples:
            msg = f"Found input variables with inconsistent numbers of samples: {[n_samples, len(target_series)]}"
            raise ValueError(msg)

        frame = X.with_columns(
            **{
                self._TARGET_NAME: target_series,
                self._GLOBAL_NAME: global_series,
            }
        ).pipe(self.__validate_frame)

        self.n_groups_ = len(self.groups_)
        self.n_features_ = frame.shape[1] - self.n_groups_ - 1

        self.estimators_ = self._fit_estimators(frame)
        self.shrinkage_factors_ = self._fit_shrinkage_factors(frame, groups=self.groups_)

        return self

    def predict(self, X):
        """Predict the target value for each sample in `X`."""
        raise NotImplementedError("This method should be implemented in the child class")

    def _predict_estimators(self, X, method_name):
        """Calls `method_name` on each level and apply shrinkage if necessary"""

        check_is_fitted(self, ["estimators_", "groups_"])

        if len(X.shape) != 2:
            raise ValueError(f"Reshape your data: X should be 2d, got {len(X.shape)}")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X should have {self.n_features_in_} features, got {X.shape[1]}")

        X = nw.from_native(X, strict=False, eager_only=True)
        if not isinstance(X, nw.DataFrame):
            X = nw.from_native(pd.DataFrame(X))

        n_samples = X.shape[0]
        native_namespace = nw.get_native_namespace(X)
        global_series = nw.new_series(
            name=self._GLOBAL_NAME, values=np.ones(n_samples), native_namespace=native_namespace
        )

        frame = X.with_columns(
            **{
                self._GLOBAL_NAME: global_series,
                self._INDEX_NAME: np.arange(n_samples),
            }
        ).pipe(self.__validate_frame)

        if not is_classifier(self.estimator):  # regressor or outlier detector
            n_out = 1
        else:
            if self.n_classes_ > 2 or method_name == "predict_proba":
                n_out = self.n_classes_
            else:  # binary case with `method_name = "decision_function"`
                n_out = 1

        preds = np.zeros((X.shape[0], self.n_fitted_levels_, n_out), dtype=float)
        shrinkage = np.zeros((X.shape[0], self.n_fitted_levels_), dtype=float)

        for level_idx, grp_names in enumerate(self.fitted_levels_):
            for grp_values, grp_frame in frame.group_by(grp_names):
                grp_idx = grp_frame.select(self._INDEX_NAME).to_numpy().reshape(-1)

                _estimator, _level = _get_estimator(
                    estimators=self.estimators_,
                    grp_values=grp_values,
                    grp_names=grp_names,
                    return_level=len(grp_names),
                    fallback_method=self.fallback_method,
                )
                _shrinkage_factor = self.shrinkage_factors_[grp_values[:_level]]

                last_dim_ix = _estimator.classes_ if is_classifier(self.estimator) else [0]
                X_grp_ = nw.to_native(grp_frame.drop([*self.groups_, self._INDEX_NAME]))
                raw_pred = getattr(_estimator, method_name)(X_grp_)

                preds[np.ix_(grp_idx, [level_idx], last_dim_ix)] = np.atleast_3d(raw_pred[:, None])
                shrinkage[np.ix_(grp_idx)] = np.pad(
                    _shrinkage_factor,
                    (0, self.n_fitted_levels_ - len(_shrinkage_factor)),
                    "constant",
                    constant_values=(0),
                )

        return (preds * np.atleast_3d(shrinkage)).sum(axis=1).squeeze()

    def _fit_single_estimator(self, grp_frame):
        """Shortcut to fit an estimator on a single group"""
        _X = nw.to_native(grp_frame.drop([*self.groups_, self._TARGET_NAME]))
        _y = nw.to_native(grp_frame[self._TARGET_NAME])

        return clone(self.estimator).fit(_X, _y)

    def _fit_estimators(self, frame: nw.DataFrame):
        """Fits one estimator per level of the group column(s), and returns a dictionary of the fitted estimators.

        The keys of the dictionary are the group values, and the values are the fitted estimators.
        The if-else block is used to parallelize the fitting process if `n_jobs` is greater than 1.
        """
        # Question: Should the `estimators_` keys be named tuples instead of plain tuples?
        if self.n_jobs is None or self.n_jobs == 1:
            estimators_ = {
                grp_values: self._fit_single_estimator(grp_frame)
                for grp_names in self.fitted_levels_
                for grp_values, grp_frame in frame.group_by(grp_names)
            }
        else:
            fit_func = lambda grp_values, grp_frame: (grp_values, self._fit_single_estimator(grp_frame))

            estimators_ = dict(
                Parallel(n_jobs=self.n_jobs)(
                    delayed(fit_func)(grp_values, grp_frame)
                    for grp_names in self.fitted_levels_
                    for grp_values, grp_frame in frame.group_by(grp_names)
                )
            )

        return estimators_

    def __validate_frame(self, frame):
        """Validate the input arrays"""

        if self.check_X:
            X_values = frame.drop([*self.groups_])
            check_array(X_values, **self._CHECK_KWARGS)

        _validate_groups_values(frame, self.groups_)

        return frame

    @property
    def n_levels_(self):
        warn(
            "Please use `n_fitted_levels_` instead of `n_levels_`, `n_levels_` will be deprecated in future versions",
            DeprecationWarning,
        )
        return self.n_fitted_levels_

    def _more_tags(self):
        return {"allow_nan": True}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


class HierarchicalRegressor(RegressorMixin, HierarchicalPredictor):
    """A hierarchical regressor that predicts values using hierarchical grouping.

    This class extends [`HierarchicalPredictor`][sklego.meta.hierarchical_predictor.HierarchicalPredictor] and adds
    functionality specific to regression tasks.

    Its spec is the same as `HierarchicalPredictor`, with additional checks to ensure that the supplied estimator is a
    regressor.

    !!! info "New in version 0.8.0"

    Examples
    --------
    ```py
    import pandas as pd

    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression

    from sklego.meta import HierarchicalRegressor

    X, y = make_regression(n_samples=1000, n_features=10, n_informative=3, random_state=42)
    X = pd.DataFrame(X, columns=[f"x_{i}" for i in range(X.shape[1])]).assign(
        g_1 = ['A'] * 500 + ['B'] * 500,
        g_2 = ['X'] * 250 + ['Y'] * 250 + ['Z'] * 250 + ['W'] * 250
    )
    groups = ["g_1", "g_2"]

    hr = HierarchicalRegressor(
        estimator=LinearRegression(),
        groups=groups
    ).fit(X, y)

    hr.estimators_
    ```

    ```terminal
    {
        (1,): LinearRegression(),  # global estimator
        (1, 'A'): LinearRegression(),  # estimator for `g_1 = 'A'`
        (1, 'B'): LinearRegression(),  # estimator for `g_1 = 'B'`
        (1, 'A', 'X'): LinearRegression(),  # estimator for `(g_1, g_2) = ('A', 'X`)`
        (1, 'A', 'Y'): LinearRegression(),  # estimator for `(g_1, g_2) = ('A', 'Y`)`
        (1, 'B', 'W'): LinearRegression(),  # estimator for `(g_1, g_2) = ('B', 'W`)`
        (1, 'B', 'Z'): LinearRegression(),  # estimator for `(g_1, g_2) = ('B', 'Z`)`
    }
    ```

    As we can see, the estimators are fitted for each level of the group columns. The trailing (1,) is the global
    estimator, which is fitted on the entire dataset.

    If we try to predict a sample in which `(g_1, g_2) = ('B', 'X')`, this will fallback to the estimator `(1, 'B')`.
    when `fallback_method="parent"` or will raise a KeyError when `fallback_method="raise"`.

    As one would expect, `estimator` can be a pipeline, and the pipeline will be fitted on each level of the group:
    ```py
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    hr = HierarchicalRegressor(
        estimator=Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
            ]),
        groups=groups
    ).fit(X, y)
    ```
    """

    def fit(self, X, y):
        """Fit one regressor for each hierarchical group of training data `X` and `y`.

        Will also learn the groups that exist within the training dataset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : HierarchicalRegressor
            The fitted regressor.

        Raises
        -------
        ValueError
            If the supplied estimator is not a regressor.
        """
        if not is_regressor(self.estimator):
            raise ValueError("The supplied estimator should be a regressor")

        super().fit(X, y)
        return self

    def predict(self, X):
        """Predict regression values for new data `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        array-like of shape (n_samples,)
            Predicted regression values.
        """
        return self._predict_estimators(X, "predict")


class HierarchicalClassifier(ClassifierMixin, HierarchicalPredictor):
    """A hierarchical classifier that predicts labels using hierarchical grouping.

    This class extends [`HierarchicalPredictor`][sklego.meta.hierarchical_predictor.HierarchicalPredictor] and adds
    functionality specific to regression tasks.

    Its spec is the same as `HierarchicalPredictor`, with additional checks to ensure that the supplied estimator is a
    classifier that implements the `.predict_proba()` method.

    !!! warning ".`predict_proba(..)` method required!"

        In order to use shrinkage with classification tasks, we require the estimator to have `.predict_proba()` method.
        The only way to blend the predictions of the group-level models is by using the probabilities of each class,
        and not the labels themselves.

    !!! info "New in version 0.8.0"

    Examples
    --------
    ```py
    import pandas as pd

    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    from sklego.meta import HierarchicalClassifier

    X, y = make_classification(n_samples=1000, n_features=10, n_informative=3, random_state=42)
    X = pd.DataFrame(X, columns=[f"x_{i}" for i in range(X.shape[1])]).assign(
        g_1 = ['A'] * 500 + ['B'] * 500,
        g_2 = ['X'] * 250 + ['Y'] * 250 + ['Z'] * 250 + ['W'] * 250
    )
    groups = ["g_1", "g_2"]

    hc = HierarchicalClassifier(
        estimator=LogisticRegression(),
        groups=groups
    ).fit(X, y)

    hc.estimators_
    ```

    ```terminal
    {
        (1,): LogisticRegression(),  # global estimator
        (1, 'A'): LogisticRegression(),  # estimator for `g_1 = 'A'`
        (1, 'B'): LogisticRegression(),  # estimator for `g_1 = 'B'`
        (1, 'A', 'X'): LogisticRegression(),  # estimator for `(g_1, g_2) = ('A', 'X`)`
        (1, 'A', 'Y'): LogisticRegression(),  # estimator for `(g_1, g_2) = ('A', 'Y`)`
        (1, 'B', 'W'): LogisticRegression(),  # estimator for `(g_1, g_2) = ('B', 'W`)`
        (1, 'B', 'Z'): LogisticRegression(),  # estimator for `(g_1, g_2) = ('B', 'Z`)`
    }
    ```

    As we can see, the estimators are fitted for each level of the group columns. The trailing (1,) is the global
    estimator, which is fitted on the entire dataset.

    If we try to predict a sample in which `(g_1, g_2) = ('B', 'X')`, this will fallback to the estimator `(1, 'B')`.
    when `fallback_method="parent"` or will raise a KeyError when `fallback_method="raise"`.

    As one would expect, `estimator` can be a pipeline, and the pipeline will be fitted on each level of the group:
    ```py
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    hc = HierarchicalClassifier(
        estimator=Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression())
            ]),
        groups=groups
    ).fit(X, y)
    ```
    """

    def fit(self, X, y):
        """Fit one classifier for each hierarchical group of training data `X` and `y`.

        Will also learn the groups that exist within the training dataset, the classes and the number of classes in the
        target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : HierarchicalClassifier
            The fitted classifier.

        Raises
        -------
        ValueError
            If the supplied estimator is not a classifier.
        """
        if not is_classifier(self.estimator):
            raise ValueError("The supplied estimator should be a classifier")

        if not hasattr(self.estimator, "predict_proba"):
            raise ValueError("The supplied estimator should have a 'predict_proba' method")

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        super().fit(X, y)
        return self

    def predict(self, X):
        """Predict class labels for samples in `X` as the class with the highest probability.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        array-like of shape (n_samples,)
            The predicted class labels.
        """

        preds = self._predict_estimators(X, method_name="predict_proba")
        return self.classes_[np.argmax(preds, axis=1)]

    def predict_proba(self, X):
        """Predict probabilities for each class on new data `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        array-like of shape (n_samples, n_classes)
            Predicted probabilities per class.
        """
        return self._predict_estimators(X, method_name="predict_proba")

    @available_if(lambda self: hasattr(self.estimator, "decision_function"))
    def decision_function(self, X):
        """Predict confidence scores for samples in `X`.

        !!! warning
            Available only if the underlying estimator implements `.decision_function()` method.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

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
        return self._predict_estimators(X, method_name="decision_function")
