# from sklearn.feature_selection._univariate_selection import _BaseFilter
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import f_classif
from sklearn.feature_selection._base import SelectorMixin
from sklearn.utils.validation import check_is_fitted, check_X_y


def _f_classif(X, y):
    return f_classif(X, y)[0]


def _redundancy(X, selected, left):
    return_list = []
    if len(selected) == 0:
        return np.ones(len(left))
    for _l in left:
        print(f"Selected l = {_l}")
        rel_list = np.sum([np.abs(np.corrcoef(X[_s], X[_l])[0, 1]) for _s in selected])
        print(f"rel_list = {rel_list}")
        return_list += [rel_list]
    return np.array(return_list)


class MinimumRelevanceMinimumRedundancy(SelectorMixin, BaseEstimator):
    def __init__(self, relevance_func, redundancy_func, k=5):
        self.relevance_func = relevance_func
        # Callable or pre-defined function with mapped as str
        self.redundancy_func = redundancy_func
        self.k = k

        # dummy comment
        # dummy comment 2

    def _get_support_mask(self):
        check_is_fitted(self, ["selected_features_"])
        all_features = [i for i in range(0, self.n_features_in_)]
        return np.isin(all_features, self.selected_features_)

    def _base_step(self):
        # Derive K = min(K, features)
        # Ensure relevance_func and redundancy_func are okay
        pass

    @property
    def _get_relevance(self):
        if self.relevance_func == "f":
            return _f_classif
        return self.relevance_func

    @property
    def _get_redundancy(self):
        if self.redundancy_func == "p":
            return _redundancy
        return self.redundancy_func

    def fit(self, X, y):
        relevance = self._get_relevance
        redundancy = self._get_redundancy

        self.n_features_in_ = X.shape[1]
        k = min(self.n_features_in_, self.k)

        X, y = check_X_y(X, y)

        left_features = [i for i in range(0, self.n_features_in_)]
        selected_features = []

        for i in range(k):
            rel_i = relevance(X[:, left_features], y)
            red_i = redundancy(X, selected_features, left_features) / (i + 1)

            print(f"Relevance_{i} = {rel_i}")
            print(f"Redundancy_{i} = {red_i}")

            selected_index = np.argmax(rel_i / red_i)
            selected_features += [left_features.pop(selected_index)]
            print(
                f"Selected index = {selected_index},\
                    selected_feature = {selected_features},\
                    left_features = {left_features}"
            )

        self.selected_features_ = np.asarray(selected_features)
        return self
