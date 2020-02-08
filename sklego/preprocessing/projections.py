import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from sklego.common import as_list


class OrthogonalTransformer(BaseEstimator, TransformerMixin):
    """
    Transform the columns of a dataframe or numpy array to a column orthogonal or orthonormal matrix.
    Q, R such that X = Q*R, with Q orthogonal, from which follows Q = X*inv(R)
    :param normalize: whether orthogonal matrix should be orthonormal as well
    """

    def __init__(self, normalize=False):
        self.normalize = normalize

    def fit(self, X, y=None):
        """
        Store the inverse of R of the QR decomposition of X, which can be used to calculate the orthogonal projection
        of X. If normalization is required, also stores a vector with normalization terms
        """
        X = check_array(X, estimator=self)

        if not X.shape[0] > 1:
            raise ValueError("Orthogonal transformation not valid for one sample")

        # Q, R such that X = Q*R, with Q orthogonal, from which follows Q = X*inv(R)
        Q, R = np.linalg.qr(X)
        self.inv_R_ = np.linalg.inv(R)

        if self.normalize:
            self.normalization_vector_ = np.linalg.norm(Q, ord=2, axis=0)
        else:
            self.normalization_vector_ = np.ones((X.shape[1],))

        return self

    def transform(self, X):
        """Transforms X using the fitted inverse of R. Normalizes the result if required"""
        if self.normalize:
            check_is_fitted(self, ["inv_R_", "normalization_vector_"])
        else:
            check_is_fitted(self, ["inv_R_"])

        X = check_array(X, estimator=self)

        return X @ self.inv_R_ / self.normalization_vector_


def scalar_projection(vec, unto):
    return vec.dot(unto) / unto.dot(unto)


def vector_projection(vec, unto):
    return scalar_projection(vec, unto) * unto


class InformationFilter(BaseEstimator, TransformerMixin):
    """
    The `InformationFilter` uses a variant of the gram smidt process
    to filter information out of the dataset. This can be useful if you
    want to filter information out of a dataset because of fairness.
    To explain how it works: given a training matrix :math:`X` that contains
    columns :math:`x_1, ..., x_k`. If we assume columns :math:`x_1` and :math:`x_2`
    to be the sensitive columns then the information-filter will
    remove information by applying these transformations;
    .. math::
       \\begin{split}
       v_1 & = x_1 \\\\
       v_2 & = x_2 - \\frac{x_2 v_1}{v_1 v_1}\\\\
       v_3 & = x_3 - \\frac{x_k v_1}{v_1 v_1} - \\frac{x_2 v_2}{v_2 v_2}\\\\
       ... \\\\
       v_k & = x_k - \\frac{x_k v_1}{v_1 v_1} - \\frac{x_2 v_2}{v_2 v_2}
       \\end{split}
    Concatenating our vectors (but removing the sensitive ones) gives us
    a new training matrix :math:`X_{fair} =  [v_3, ..., v_k]`.
    :param columns: the columns to filter out this can be a sequence of either int
                    (in the case of numpy) or string (in the case of pandas).
    :param alpha: parameter to control how much to filter, for alpha=1 we filter out
                  all information while for alpha=0 we don't apply any.
    """

    def __init__(self, columns, alpha=1):
        self.columns = columns
        self.alpha = alpha

    def _check_coltype(self, X):
        for col in as_list(self.columns):
            if isinstance(col, str):
                if isinstance(X, np.ndarray):
                    raise ValueError(
                        f"column {col} is a string but datatype receive is numpy."
                    )
                if isinstance(X, pd.DataFrame):
                    if col not in X.columns:
                        raise ValueError(f"column {col} is not in {X.columns}")
            if isinstance(col, int):
                if col not in range(np.atleast_2d(np.array(X)).shape[1]):
                    raise ValueError(
                        f"column {col} is out of bounds for input shape {X.shape}"
                    )

    def _col_idx(self, X, name):
        if isinstance(name, str):
            if isinstance(X, np.ndarray):
                raise ValueError(
                    "You cannot have a column of type string on a numpy input matrix."
                )
            return {name: i for i, name in enumerate(X.columns)}[name]
        return name

    def _make_v_vectors(self, X, col_ids):
        vs = np.zeros((X.shape[0], len(col_ids)))
        for i, c in enumerate(col_ids):
            vs[:, i] = X[:, col_ids[i]]
            for j in range(0, i):
                vs[:, i] = vs[:, i] - vector_projection(vs[:, i], vs[:, j])
        return vs

    def fit(self, X, y=None):
        """Learn the projection required to make the dataset orthogonal to sensitive columns."""
        self._check_coltype(X)
        self.col_ids_ = [
            v if isinstance(v, int) else self._col_idx(X, v)
            for v in as_list(self.columns)
        ]
        X = check_array(X, estimator=self)
        X_fair = X.copy()
        v_vectors = self._make_v_vectors(X, self.col_ids_)
        # gram smidt process but only on sensitive attributes
        for i, col in enumerate(X_fair.T):
            for v in v_vectors.T:
                X_fair[:, i] = X_fair[:, i] - vector_projection(X_fair[:, i], v)
        # we want to learn matrix P: X P = X_fair
        # this means we first need to create X_fair in order to learn P
        self.projection_, resid, rank, s = np.linalg.lstsq(X, X_fair, rcond=None)
        return self

    def transform(self, X):
        """Transforms X by applying the information filter."""
        check_is_fitted(self, ["projection_", "col_ids_"])
        self._check_coltype(X)
        X = check_array(X, estimator=self)
        # apply the projection and remove the column we won't need
        X_fair = X @ self.projection_
        X_removed = np.delete(X_fair, self.col_ids_, axis=1)
        X_orig = np.delete(X, self.col_ids_, axis=1)
        return self.alpha * np.atleast_2d(X_removed) + (1 - self.alpha) * np.atleast_2d(
            X_orig
        )
