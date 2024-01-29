# from sklearn.feature_selection._univariate_selection import _BaseFilter
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin
from sklearn.utils.validation import check_is_fitted, check_X_y


class MinimumRelevanceMinimumRedundancy(SelectorMixin, BaseEstimator):
    def __init__(self, relevance_func, redundancy_func, k=5):
        self.relevance_func = relevance_func
        # Callable or pre-defined function with mapped as str
        self.redundancy_func = redundancy_func
        self.k = k

    def _get_support_mask(self):
        check_is_fitted(self, ["selected_features_"])
        return self.selected_features_

    def _base_step(self):
        # Derive K = min(K, features)
        # Ensure relevance_func and redundancy_func are okay
        pass

    def fit(self, X, y):
        # main logic
        self.n_features_in_ = X.shape[1]
        # k =
        self._base_step()

        X, y = check_X_y(X, y)

        # left_features = []
        # selected_features = []

        # Perform base model
        for i in range(self.k):
            pass

            # score_i = [rel(j, i ) / red(j, i )  for ] in left_features]
            # selected_index = np.argmax(score_i)

            # get best feature
            # selected_feature += [left_features.pop(selected_index)]

        # self.selected_features_ = np.asarray(selected_features)
        return self
