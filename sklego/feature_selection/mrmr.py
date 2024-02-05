# from sklearn.feature_selection._univariate_selection import _BaseFilter
import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import f_classif, f_regression
from sklearn.feature_selection._base import SelectorMixin
from sklearn.utils.validation import check_is_fitted, check_X_y


def _redundancy(X, selected, left):
    if len(selected) == 0:
        return np.ones(len(left))

    X_norm = X - np.mean(X, axis=0, keepdims=True)
    Xs = X_norm[:, selected]
    Xl = X_norm[:, left]

    num = (Xl[:, None, :] * Xs[:, :, None]).sum(axis=0)
    den = np.sqrt((Xl[:, None, :] ** 2).sum(axis=0)) * np.sqrt((Xs[:, :, None] ** 2).sum(axis=0))

    return np.sum(np.abs(np.nan_to_num(num / den, nan=np.finfo(float).eps)), axis=0)


class MaximumRelevanceMinimumRedundancy(SelectorMixin, BaseEstimator):
    def __init__(self, k, kind="auto", relevance_func="f", redundancy_func="p"):
        self.k = k
        self.kind = kind
        self.relevance_func = relevance_func
        self.redundancy_func = redundancy_func

    def _get_support_mask(self):
        check_is_fitted(self, ["selected_features_"])
        all_features = np.arange(0, self.n_features_in_)
        return np.isin(all_features, self.selected_features_)

    @property
    def _get_relevance(self):
        if self.relevance_func == "f":
            if (self.kind == "auto" and np.issubdtype(self._y_dtype, np.integer)) | (self.kind == "classification"):
                return lambda X, y: np.nan_to_num(f_classif(X, y)[0])
            elif (self.kind == "auto" and np.issubdtype(self._y_dtype, np.floating)) | (self.kind == "regression"):
                return lambda X, y: np.nan_to_num(f_regression(X, y)[0])
            else:
                raise
        elif callable(self.relevance_func):
            return self.relevance_function
        else:
            raise

    @property
    def _get_redundancy(self):
        if self.redundancy_func == "p":
            return _redundancy
        elif callable(self.redundancy_func):
            return self.redundancy_func
        else:
            raise

    def fit(self, X, y):
        self._y_dtype = y.dtype
        relevance = self._get_relevance
        redundancy = self._get_redundancy

        self.n_features_in_ = X.shape[1]
        left_features = list(range(self.n_features_in_))
        selected_features = []
        selected_scores = []

        if not isinstance(self.k, int):
            raise ValueError("k parameter mush be integer type")
        if self.k > self.n_features_in_:
            raise ValueError(f"k parameter mush be < n_features_in, got {self.k} - {self.n_features_in_}")
        elif self.k == self.n_features_in_:
            warnings.warn("k parameter is equal to n_features_in, no feature selection is applied")
            return np.asarray(left_features)
        elif self.k < 1:
            raise ValueError(f"k parameter mush be >= 1, got {self.k}")

        k = min(self.n_features_in_, self.k)

        X, y = check_X_y(X, y)

        # computed one time for all features
        rel_score = relevance(X, y)

        for i in range(k):
            red_i = redundancy(X, selected_features, left_features) / (i + 1)
            mrmr_score_i = rel_score[left_features] / red_i
            selected_index = np.argmax(mrmr_score_i)
            selected_features += [left_features.pop(selected_index)]
            selected_scores += [mrmr_score_i[selected_index]]
            # print(f"Scores = {mrmr_score_i}")
            # print(f"Selected_index = {selected_index}")
            # print(f"Selected score = {selected_scores}")

        self.selected_features_ = np.asarray(selected_features)
        self.scores_ = np.asarray(selected_scores)
        print(self.selected_features_)
        print(self.scores_)
        return self
