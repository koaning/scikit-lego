import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from functools import partial


class DictMapper(TransformerMixin, BaseEstimator):
    def __init__(self, mapper, default):
        self.mapper = mapper
        self.default = default

    def fit(self, X, y=None):
        X = check_array(
            X,
            copy=True,
            estimator=self,
            force_all_finite=True,
            dtype=None,
            ensure_2d=True,
        )
        self.dim_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, ["dim_"])
        X = check_array(
            X,
            copy=True,
            estimator=self,
            force_all_finite=True,
            dtype=None,
            ensure_2d=True,
        )

        if X.shape[1] != self.dim_:
            raise ValueError(
                f"number of columns {X.shape[1]} does not match fit size {self.dim_}"
            )
        return np.vectorize(self.mapper.get, otypes=[int])(X, self.default)
