import numpy as np
import pandas as pd

from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array

from sklego.common import as_list


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
