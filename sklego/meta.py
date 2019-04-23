import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from sklego.common import as_list


class GroupedEstimator(BaseEstimator):
    """
    Construct an estimator per data group. Splits data by values of a
    single column and fits one estimator per such column.

    :param estimator: the model/pipeline to be applied per group
    :param groupby: the column(s) of the matrix/dataframe to select as a grouping parameter set
    :param use_fallback: weather or not to fall back to a general model in case
    the group parameter is not found during `.predict()`
    """

    def __init__(self, estimator, groupby, use_fallback=True):
        self.estimator = estimator
        self.groupby = [str(_) for _ in as_list(groupby)]
        self.use_fallback = use_fallback

    def fit(self, X, y):
        """
        Fit the model using X, y as training data. Will also learn the groups
        that exist within the dataset.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :param y: array-like, shape=(n_samples,) training data.
        :return: Returns an instance of self.
        """
        self.estimators_ = {}
        self.groups_ = []

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[str(_) for _ in range(X.shape[1])])
        pred_col = 'the-column-that-i-want-to-predict-but-dont-have-the-name-for'
        X = X.assign(**{pred_col: y})

        self.groups_ = X[self.groupby].drop_duplicates().itertuples(index=False)
        self.groups_ = [tuple(_) for _ in self.groups_]

        for group, subset in X.groupby(self.groupby):
            group_estimator = clone(self.estimator)
            subset_x = subset.drop(columns=[pred_col]).drop(columns=self.groupby)
            subset_y = subset[pred_col]
            group_estimator.fit(subset_x, subset_y)
            self.estimators_[group] = group_estimator

        if self.use_fallback:
            subset_x = X.drop(columns=[pred_col]).drop(columns=self.groupby)
            self.fallback_ = clone(self.estimator).fit(subset_x, y)

        return self

    def predict(self, X):
        """
        Predict new data by making random guesses.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :return: array, shape=(n_samples,) the predicted data
        """
        check_is_fitted(self, ['estimators_', 'groups_'])
        if self.use_fallback:
            check_is_fitted(self, ['fallback_'])
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[str(_) for _ in range(X.shape[1])])
        result = []
        for row in X.itertuples(index=False):
            group_to_use = tuple([getattr(row, k) for k in self.groupby])
            data_to_use = [getattr(row, c) for c in X.columns if c not in self.groupby]
            if group_to_use in self.groups_:
                result.append(self.estimators_[group_to_use].predict([data_to_use]))
            else:
                if not self.use_fallback:
                    raise ValueError(f"we see group {group_to_use} while use_fallback={self.use_fallback}")
                result.append(self.fallback_.predict([data_to_use]))
        return np.array(result).squeeze()
