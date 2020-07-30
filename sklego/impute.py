import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array


def mask_array(array: np.ndarray, p: float = 0.1):
    """Randomly make a fraction of entries go missing

    Args:
        array (np.ndarray): Input array
        p (float, optional): Fraction of missings. Defaults to 0.1.
    """
    array = array.copy()
    mask_indices = np.random.choice([True, False], size=array.shape, p=(p, 1 - p))
    array[mask_indices] = np.nan
    return array


def get_X(n=100, k=5, p=0.1):
    """Get some test data for SVDImputed"""
    np.random.seed(42)
    return mask_array(np.random.normal(0, 1, (n, k)), p=p)


class SVDImputer(BaseEstimator, TransformerMixin):
    def __init__(self, k_rank=1):
        self.k_rank = k_rank

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

    @staticmethod
    def _fill_missings(X):
        X = X.copy()
        # Missing indices
        inds = np.where(np.isnan(X))

        # TODO: Mean? Or maybe median?
        means = np.nanmean(X, axis=0)

        X[inds] = np.take(means, inds[1])
        return inds, X

    def _get_kth_approximation(self, X):
        k_rank = self.k_rank

        # shapes: U: (n, n), D: (p, ), V: (p, p)
        U, D, V = np.linalg.svd(X, compute_uv=True, full_matrices=False)

        return np.dot(U[:, :k_rank] * D[:k_rank], V[:k_rank, :])

    def fit(self, X, y=None):
        # X has shape (n, p)
        self.__validate(X)

        inds, X = self._fill_missings(X)

        while True:
            prev_missings = X[inds]
            X = self._get_kth_approximation(X)

            if np.allclose(prev_missings, X[inds]):
                break

        return self

    def transform(self, X):
        # TODO: Implement transform
        return X
