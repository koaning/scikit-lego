import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array


class SVDImputer(BaseEstimator, TransformerMixin):
    def __init__(self, k_rank=1, use_train=True, replace_row=False):
        self.k_rank = k_rank
        self.use_train = use_train
        self.replace_row = replace_row

    def __validate(self, X):
        X = check_array(X, force_all_finite=False)

        if not 0 < self.k_rank <= X.shape[1]:
            raise ValueError(
                f"k_rank should be greater than 0 and at most the number of columns of X ({X.shape[1]}), got {self.k_rank}"
            )
        if self.k_rank == X.shape[1]:
            warnings.warn(
                f"k_rank equal to number of columns of X ({self.k_rank}), result is same as mean impute"
            )

        return X

    @staticmethod
    def _fill_missings(X):
        X = X.copy()
        # Missing indices
        missing_idx = np.isnan(X)

        means = np.nanmean(X, axis=0)

        X[np.where(missing_idx)] = np.take(means, np.where(missing_idx)[1])
        return missing_idx, X

    def _get_kth_approximation(self, X):
        k_rank = self.k_rank

        # shapes: U: (n, n), D: (p, ), V: (p, p)
        U, D, V = np.linalg.svd(X, compute_uv=True, full_matrices=False)

        return np.dot(U[:, :k_rank] * D[:k_rank], V[:k_rank, :])

    def fit(self, X, y=None):
        # X has shape (n, p)
        X = self.__validate(X)

        # Store X_ since we want to use it in transform
        self.X_ = X if self.use_train else None

        return self

    def _transform(self, X):
        """Actual implementation of the imputation"""
        missing_idx, X = self._fill_missings(X)

        X_transformed = X.copy()

        while True:
            prev_missings = X[missing_idx]
            X = self._get_kth_approximation(X)

            if np.allclose(prev_missings, X[missing_idx]):
                break

        if self.replace_row:
            rows = missing_idx.sum(axis=1) > 0
            X_transformed[rows, :] = X[rows, :]
        else:
            X_transformed[missing_idx] = X[missing_idx]

        return X_transformed

    def transform(self, X):
        X = self.__validate(X)

        transform_data_length = len(X)

        if self.use_train:
            X = np.concatenate([self.X_, X], axis=0)

        X_transformed = self._transform(X)

        return X_transformed[-transform_data_length:, :]

    def fit_transform(self, X, y=None):
        X = self.__validate(X)

        return self._transform(X)
