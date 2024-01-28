# from sklearn.feature_selection._univariate_selection import _BaseFilter
import numpy as np
from sklearn.base import BaseEstimator, _fit_context
from sklearn.feature_selection._base import SelectorMixin
from sklearn.utils.validation import check_is_fitted


class MinimumRelevanceMinimumRedundancy(SelectorMixin, BaseEstimator):
    def __init__(self, relevance_func, redundancy_func, k=5):
        self.relevance_func = relevance_func
        self.redundancy_func = redundancy_func
        self.k = k

    def _get_support_mask(self):
        check_is_fitted(self)
        base_features = ""

        mask = np.isin(self.scores_, base_features)
        return mask

    def _base_step(self):
        # Derive K = min(K, features)
        # Ensure relevance_func and redundancy_func are okay
        # Setup the base mask
        pass

    @_fit_context
    def fit(self):
        self._base_step()

        left_features = []
        selected_features = []

        # Perform base model
        for i in range(self.k):
            for j in left_features:
                # relevance_i = []
                # redundancy_i = []
                # score_i = relevance_i / redundancy_i
                pass

            # get best feature
            selected_features = "feature"
            left_features.pop(selected_features)

        self.scores_ = np.asarray(selected_features)
        return self
