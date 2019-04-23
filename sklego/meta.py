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
    """

    def __init__(self, estimator, groupby):
        self.estimator = estimator
        self.groupby = [str(_) for _ in as_list(groupby)]

    def fit(self, X, y):
        self.estimators_ = {}
        self.groups_ = []
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[str(_) for _ in range(X.shape[1])])
        pred_col = 'the-column-that-i-want-to-predict-but-dont-have-the-name-for'
        X = X.assign(**{pred_col: y})
        self.groups_ = (X
                        .groupby(self.groupby)
                        .count()
                        .reset_index()
                        [self.groupby])
        for group, subset in X.groupby(self.groupby):
            group_estimator = clone(self.estimator)
            subset_x = subset.drop(columns=[pred_col]).drop(columns=self.groupby)
            subset_y = subset[pred_col]
            group_estimator.fit(subset_x, subset_y)
            self.estimators_[group] = group_estimator
        return self

    def predict(self, X):
        check_is_fitted(self, ['estimators_', 'groups_'])
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[str(_) for _ in range(X.shape[1])])
        result = []
        for row in X.itertuples():
            group_to_use = (tuple([getattr(row, k) for k in self.groupby]))
            data_to_use = [getattr(row, c) for c in X.columns if c not in self.groupby]
            result.append(self.estimators_[group_to_use].predict([data_to_use]))
        return np.array(result).squeeze()
