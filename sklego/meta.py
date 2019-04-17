import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.base import BaseEstimator

from sklearn.utils.validation import check_is_fitted, check_array, FLOAT_DTYPES


class GroupedEstimator(BaseEstimator):
    """
    Construct an estimator per data group. Splits data by values of a
    single column and fits one estimator per such column.
    """

    def __init__(self, estimator, groupby):
        self.estimator = estimator
        self.groupby = groupby
        self.estimators_ = None

    def fit(self, X, y):
        self.estimators_ = {}
        for group in np.unique(X[self.groupby]):
            selector = X[self.groupby] == group
            x_group, y_group = X[selector], y[selector]
            group_estimator = clone(self.estimator)
            group_estimator.fit(x_group.drop(columns=self.groupby), y_group)
            self.estimators_[group] = group_estimator
        return self

    def predict(self, X):
        check_is_fitted(self, ['estimators_'])
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        if isinstance(X, pd.DataFrame):
            return [
                self.estimators_[row[self.groupby]].predict([row])[0]
                for _, row in X.iterrows()
            ]
        return [
            self.estimators_[row[self.groupby]].predict([row])[0]
            for _, row in X
        ]
