import numpy as np
import pandas as pd

from typing import List, Union

from sklearn import clone
from sklearn.base import BaseEstimator, TransformerMixin, MetaEstimatorMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array, FLOAT_DTYPES

from sklego.base import ProbabilisticClassifier
from sklego.common import as_list, TrainOnlyTransformerMixin


class EstimatorTransformer(TransformerMixin, MetaEstimatorMixin, BaseEstimator):
    """
    Allows using an estimator such as a model as a transformer in an earlier step of a pipeline

    :param estimator: An instance of the estimator that should be used for the transformation
    :param predict_func: The function called on the estimator when transforming e.g. (`predict`, `predict_proba`)
    """

    def __init__(self, estimator, predict_func='predict'):
        self.estimator = estimator
        self.predict_func = predict_func

    def fit(self, X, y):
        """Fits the estimator"""
        X, y = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        return self

    def transform(self, X):
        """
        Applies the `predict_func` on the fitted estimator.

        Returns an array of shape `(X.shape[0], )`.
        """
        check_is_fitted(self, 'estimator_')
        return getattr(self.estimator_, self.predict_func)(X).reshape(-1, 1)


def constant_shrinkage(group_sizes: list, alpha: float) -> np.ndarray:
    r"""
    The augmented prediction for each level is the weighted average between its prediction and the augmented
    prediction for its parent.

    Let $\hat{y}_i$ be the prediction at level $i$, with $i=0$ being the root, than the augmented prediction
    $\hat{y}_i^* = \alpha \hat{y}_i + (1 - \alpha) \hat{y}_{i-1}^*$, with $\hat{y}_0^* = \hat{y}_0$.


    """
    return np.array(
        [alpha ** (len(group_sizes) - 1)]
        + [alpha ** (len(group_sizes) - 1 - i) * (1 - alpha) for i in range(1, len(group_sizes) - 1)]
        + [(1 - alpha)]
    )


def relative_shrinkage(group_sizes: list) -> np.ndarray:
    """Weigh each group according to it's size"""
    return np.array(group_sizes)


def min_n_obs_shrinkage(group_sizes: list, min_n_obs) -> np.ndarray:
    """Use only the smallest group with a certain amount of observations"""
    if min_n_obs > max(group_sizes):
        raise ValueError(f"There is no group with size greater than or equal to {min_n_obs}")

    res = np.zeros(len(group_sizes))
    res[np.argmin(np.array(group_sizes) >= min_n_obs) - 1] = 1
    return res


class GroupedEstimator(BaseEstimator):
    """
    Construct an estimator per data group. Splits data by values of a
    single column and fits one estimator per such column.

    :param estimator: the model/pipeline to be applied per group
    :param groups: the column(s) of the matrix/dataframe to select as a grouping parameter set
    :param use_fallback: whether or not to fall back to a general model in case
    the group parameter is not found during `.predict()`
    :param shrinkage: Whether or not to use a shrinkage estimation, default False
    :param shrinkage_function: {"constant", "min_n_obs", "relative"} or a function, default "constant"
                               * constant: shrinked prediction for a level is weighted average of its prediction and its
                                           parents prediction
                               * min_n_obs: shrinked prediction is the prediction for the smallest group with at least
                                            n observations in it
                               * relative: each group-level is weight according to its size
                               * function: a function that takes a list of group lengths and returns an array of the
                               same size with the weights for each group
    :param **kwargs: keyword arguments to the shrinkage function
    """
    def __init__(self, estimator, groups, use_fallback=True, shrinkage=False, shrinkage_function="constant", **kwargs):
        self.estimator = estimator
        self.groups = groups
        self.use_fallback = use_fallback  # Do we need this in case of shrinkage?
        self.shrinkage = shrinkage
        self.shrinkage_function = shrinkage_function  # Default is constant
        self.kwargs = kwargs

    def __check_group_cols_exist(self, X):
        """Check whether the specified grouping columns are in X"""
        if X.shape[1] == 0:
            raise ValueError(f"0 feature(s) (shape=({X.shape[0]}, 0)) while a minimum of 1 is required.")
        if isinstance(X, pd.DataFrame):
            x_cols = set(X.columns)
        else:
            ncols = 1 if X.ndim == 1 else X.shape[1]

            x_cols = set(range(ncols))

        diff = set(as_list(self.groups)) - x_cols
        if len(diff) > 0:
            raise KeyError(f'{diff} not in columns of X ({x_cols})')

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
        # Split the model data from the grouping columns, this part is checked `regularly`
        X_data = self.__remove_groups_from_x(X)

        # __validate is used in both fit and predict, so y can be None
        if X_data.shape[1] > 0 and y is not None:
            check_X_y(X_data, y)
        elif y is not None:
            # X can be empty in, for example, a Dummy estimator
            check_array(y, ensure_2d=False)
        elif X_data.shape[1] > 0:
            check_array(X_data)

        self.__check_missing_and_inf(X)
        self.__check_group_cols_exist(X)

    def __remove_groups_from_x(self, X):
        """Remove the grouping columns from X"""
        if isinstance(X, pd.DataFrame):
            return X.drop(columns=self.groups, inplace=False)
        else:
            return np.delete(X, self.groups, axis=1)

    def __fit_grouped_estimator(self, X, target_col, value_columns, group_columns):
        return (
            X
            .groupby(group_columns)
            .apply(lambda d: clone(self.estimator).fit(d[value_columns], d[target_col]))
            .to_dict()
        )

    def __check_shrinkage_func(self):
        """Validate the shrinkage function"""
        group_lengths = [10, 5, 2]
        expected_shape = np.array(group_lengths).shape
        try:
            result = self.shrinkage_function_(group_lengths)
            if not isinstance(result, np.ndarray):
                raise ValueError(f"shrinkage_function({group_lengths}) should return an np.ndarray")
            if result.shape != expected_shape:
                raise ValueError(f"shrinkage_function({group_lengths}).shape should be {expected_shape}")
        except Exception as e:
            raise ValueError(f"Caught an exception while checking the shrinkage function: {str(e)}")

    def __set_shrinkage_function(self):
        if isinstance(self.shrinkage_function, str):
            shrink_options = {
                "constant": constant_shrinkage,
                "relative": relative_shrinkage,
                "min_n_obs": min_n_obs_shrinkage,
            }
            try:
                self.shrinkage_function_ = shrink_options.get(self.shrinkage_function)
            except AttributeError:
                raise ValueError(f"The specified shrink function {self.shrinkage_function} is not valid, "
                                 f"choose from {list(shrink_options.keys())}")
        else:
            self.shrinkage_function_ = self.shrinkage_function

    def __get_shrinkage_factor(self, X):
        """Get for all complete groups an array of shrinkages"""
        counts = X.groupby(self.group_colnames_).size()

        hierarchical_counts = {
            complete_group: [counts[tuple(subgroup)].sum() for subgroup in self.__expanding_list(complete_group, tuple)]
            for complete_group in self.complete_groups_
        }

        self.__set_shrinkage_function()

        shrinkage_factors = {
            group: self.shrinkage_function_(counts, **self.kwargs)
            for group, counts in hierarchical_counts.items()
        }

        # Make sure that the factors sum to one
        shrinkage_factors = {group: value / value.sum() for group, value in shrinkage_factors.items()}

        return shrinkage_factors

    def fit(self, X, y=None):
        """
        Fit the model using X, y as training data. Will also learn the groups
        that exist within the dataset.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :param y: array-like, shape=(n_samples,) training data.
        :return: Returns an instance of self.
        """
        self.__validate(X, y)

        pred_col = 'the-column-that-i-want-to-predict-but-dont-have-the-name-for'
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[str(_) for _ in range(X.shape[1])])
        X = X.assign(**{pred_col: y})

        self.group_colnames_ = [str(_) for _ in as_list(self.groups)]

        # List of all hierarchical subsets of columns
        self.group_colnames_hierarchical_ = self.__expanding_list(self.group_colnames_, list)

        if any([c not in X.columns for c in self.group_colnames_]):
            raise KeyError(f"{self.group_colnames_} not in {X.columns}")

        self.value_colnames_ = [_ for _ in X.columns if _ not in self.group_colnames_ and _ is not pred_col]
        self.fallback_ = None

        if self.use_fallback:
            subset_x = X[self.value_colnames_]
            self.fallback_ = clone(self.estimator).fit(subset_x, y)

        if self.shrinkage:
            self.estimators_ = {}

            for level_colnames in self.group_colnames_hierarchical_:
                self.estimators_.update(
                    self.__fit_grouped_estimator(X, pred_col, self.value_colnames_, level_colnames)
                )
        else:
            self.estimators_ = self.__fit_grouped_estimator(X, pred_col, self.value_colnames_, self.group_colnames_)

        self.groups_ = as_list(self.estimators_.keys())

        if self.shrinkage:
            self.complete_groups_ = [grp for grp in self.groups_ if len(as_list(grp)) == len(self.group_colnames_)]

            self.shrinkage_factors_ = self.__get_shrinkage_factor(X)

        return self

    def __predict_group(self, X, group_colnames):
        """Make predictions for all groups"""
        try:
            return (
                X
                .groupby(group_colnames, as_index=False)
                .apply(lambda d: pd.DataFrame(
                    self.estimators_.get(d.name, self.fallback_).predict(d[self.value_colnames_]), index=d.index))
                .values
                .squeeze()
            )
        except AttributeError:
            # Handle new groups
            culprits = (
                set(X[self.group_colnames_].agg(func=tuple, axis=1))
                - set(self.estimators_.keys())
            )
            raise ValueError(f"found a group(s) {culprits} in `.predict` that was not in `.fit`")

    def __predict_shrinkage_groups(self, X):
        """Make predictions for all shrinkage groups"""
        # DataFrame with predictions for all levels per row
        hierarchical_predictions = pd.concat([
            pd.Series(self.__predict_group(X, level_columns)) for level_columns in self.group_colnames_hierarchical_
        ], axis=1)

        # This is a Series with values the arrays
        prediction_groups = X[self.group_colnames_].agg(func=tuple, axis=1)

        shrinkage_factors = prediction_groups.map(self.shrinkage_factors_)

        if any(shrinkage_factors.isnull()):
            # Handle new groups
            diff = set(prediction_groups) - set(self.shrinkage_factors_.keys())

            raise ValueError(f"found a group(s) {diff} in `.predict` that was not in `.fit`")

        # So convert it to a dataframe
        shrinkage_factors = pd.DataFrame.from_dict(shrinkage_factors.to_dict()).T

        return (hierarchical_predictions * shrinkage_factors).sum(axis=1)

    def predict(self, X):
        """
        Predict on new data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :return: array, shape=(n_samples,) the predicted data
        """
        self.__validate(X)

        check_is_fitted(self, ['estimators_', 'groups_', 'group_colnames_',
                               'value_colnames_', 'fallback_'])
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[str(_) for _ in range(X.shape[1])])

        if any([c not in X.columns for c in self.group_colnames_]):
            raise ValueError(f"group columns {self.group_colnames_} not in {X.columns}")
        if any([c not in X.columns for c in self.value_colnames_]):
            raise ValueError(f"columns to use {self.value_colnames_} not in {X.columns}")

        if not self.shrinkage:
            return self.__predict_group(X, group_colnames=self.group_colnames_)
        else:
            return self.__predict_shrinkage_groups(X)

    def __expanding_list(self, list_to_extent: List[str], return_type=list) -> Union[List[list], List[tuple]]:
        """
        Make a expanding list of lists by making tuples of the first element, the first 2 elements etc.

        :param list_to_extent:
        :param return_type: type of the elements of the list (tuple or list)

        :Example:

        __expanding_list('test') -> [['test']]

        __expanding_list(['test1', 'test2', 'test3']) -> [['test1'], ['test1', 'test2'], ['test1', 'test2', 'test3']]
        """
        listed = as_list(list_to_extent)
        if len(listed) <= 1:
            return listed

        return [return_type(listed[:n+1]) for n in range(len(listed))]


class OutlierRemover(TrainOnlyTransformerMixin, BaseEstimator):
    """
    Removes outliers (train-time only) using the supplied removal model.

    :param outlier_detector: must implement `fit` and `predict` methods
    :param refit: If True, fits the estimator during pipeline.fit().

    """
    def __init__(self, outlier_detector, refit=True):
        self.outlier_detector = outlier_detector
        self.refit = refit
        self.estimator_ = None

    def fit(self, X, y=None):
        self.estimator_ = clone(self.outlier_detector)
        if self.refit:
            super().fit(X, y)
            self.estimator_.fit(X, y)
        return self

    def transform_train(self, X):
        check_is_fitted(self, 'estimator_')
        predictions = self.estimator_.predict(X)
        check_array(predictions, estimator=self.outlier_detector, ensure_2d=False)
        return X[predictions != -1]


class DecayEstimator(BaseEstimator):
    """
    Morphs an estimator suchs that the training weights can be
    adapted to ensure that points that are far away have less weight.
    Note that it is up to the user to sort the dataset appropriately.
    This meta estimator will only work for estimators that have a
    "sample_weights" argument in their `.fit()` method.

    The DecayEstimator will use exponential decay to weight the parameters.

    w_{t-1} = decay * w_{t}
    """

    def __init__(self, model, decay: float = 0.999, decay_func="exponential"):
        self.model = model
        self.decay = decay
        self.func = decay_func

    def _is_classifier(self):
        return any(['ClassifierMixin' in p.__name__ for p in type(self.model).__bases__])

    def fit(self, X, y):
        """
        Fit the data after adapting the same weight.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :param y: array-like, shape=(n_samples,) training data.
        :return: Returns an instance of self.
        """
        X, y = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)
        self.weights_ = np.cumprod(np.ones(X.shape[0]) * self.decay)[::-1]
        self.estimator_ = clone(self.model)
        try:
            self.estimator_.fit(X, y, sample_weight=self.weights_)
        except TypeError as e:
            if "sample_weight" in str(e):
                raise TypeError(f"Model {type(self.model).__name__}.fit() does not have 'sample_weight'")
        if self._is_classifier():
            self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        """
        Predict new data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :return: array, shape=(n_samples,) the predicted data
        """
        if self._is_classifier():
            check_is_fitted(self, ['classes_'])
        check_is_fitted(self, ['weights_', 'estimator_'])
        return self.estimator_.predict(X)

    def score(self, X, y):
        return self.estimator_.score(X, y)


class Thresholder(BaseEstimator):
    """
    Takes a two class estimator and moves the threshold. This way you might
    design the algorithm to only accept a certain class if the probability
    for it is larger than, say, 90% instead of 50%.
    """

    def __init__(self, model, threshold: float):
        self.model = model
        self.threshold = threshold

    def fit(self, X, y):
        """
        Fit the data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :param y: array-like, shape=(n_samples,) training data.
        :return: Returns an instance of self.
        """
        X, y = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)
        self.estimator_ = clone(self.model)
        if not isinstance(self.estimator_, ProbabilisticClassifier):
            raise ValueError("The Thresholder meta model only works on classifcation models with .predict_proba.")
        self.estimator_.fit(X, y)
        self.classes_ = self.estimator_.classes_
        if len(self.classes_) != 2:
            raise ValueError("The Thresholder meta model only works on models with two classes.")
        return self

    def predict(self, X):
        """
        Predict new data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :return: array, shape=(n_samples,) the predicted data
        """
        check_is_fitted(self, ['classes_', 'estimator_'])
        predicate = self.estimator_.predict_proba(X)[:, 1] > self.threshold
        return np.where(predicate, self.classes_[1], self.classes_[0])

    def predict_proba(self, X):
        check_is_fitted(self, ['classes_', 'estimator_'])
        return self.estimator_.predict_proba(X)

    def score(self, X, y):
        return self.estimator_.score(X, y)
