import numpy as np
import pandas as pd
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


class GroupedEstimator(BaseEstimator):
    """
    Construct an estimator per data group. Splits data by values of a
    single column and fits one estimator per such column.

    :param estimator: the model/pipeline to be applied per group
    :param groups: the column(s) of the matrix/dataframe to select as a grouping parameter set
    :param use_fallback: weather or not to fall back to a general model in case
    the group parameter is not found during `.predict()`
    """

    def __init__(self, estimator, groups, use_fallback=True):
        self.estimator = estimator
        self.groups = groups
        self.use_fallback = use_fallback

    def fit(self, X, y):
        """
        Fit the model using X, y as training data. Will also learn the groups
        that exist within the dataset.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :param y: array-like, shape=(n_samples,) training data.
        :return: Returns an instance of self.
        """
        check_X_y(X, y)
        pred_col = 'the-column-that-i-want-to-predict-but-dont-have-the-name-for'
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[str(_) for _ in range(X.shape[1])])
        X = X.assign(**{pred_col: y})

        self.group_colnames_ = [str(_) for _ in as_list(self.groups)]
        if any([c not in X.columns for c in self.group_colnames_]):
            raise ValueError(f"{self.group_colnames_} not in {X.columns}")
        self.X_colnames_ = [_ for _ in X.columns if _ not in self.group_colnames_ and _ is not pred_col]
        self.fallback_ = None
        if self.use_fallback:
            subset_x = X[self.X_colnames_]
            self.fallback_ = clone(self.estimator).fit(subset_x, y)

        self.groups_ = X[self.group_colnames_].drop_duplicates()

        self.estimators_ = (X
                            .groupby(self.group_colnames_)
                            .apply(lambda d: clone(self.estimator).fit(d[self.X_colnames_], d[pred_col]))
                            .to_dict())
        return self

    def predict(self, X):
        """
        Predict new data by making random guesses.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :return: array, shape=(n_samples,) the predicted data
        """
        check_array(X)
        check_is_fitted(self, ['estimators_', 'groups_', 'group_colnames_',
                               'X_colnames_', 'fallback_'])
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[str(_) for _ in range(X.shape[1])])

        if any([c not in X.columns for c in self.group_colnames_]):
            raise ValueError(f"group columns {self.group_colnames_} not in {X.columns}")
        if any([c not in X.columns for c in self.X_colnames_]):
            raise ValueError(f"columns to use {self.X_colnames_} not in {X.columns}")

        try:
            return (X
                    .groupby(self.group_colnames_, as_index=False)
                    .apply(lambda d: pd.DataFrame(
                        self.estimators_.get(d.name, self.fallback_).predict(d[self.X_colnames_]), index=d.index))
                    .values
                    .squeeze())
        except AttributeError:
            culprits = set(pd.concat([X[self.group_colnames_].drop_duplicates().assign(new=1),
                                      self.groups_.assign(new=0)])
                           .drop_duplicates()
                           .loc[lambda d: d['new'] == 1]
                           .itertuples())
            raise ValueError(f"found a group(s) {culprits} in `.predict` that was not in `.fit`")


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
