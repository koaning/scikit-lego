import numpy as np
from patsy import dmatrix, build_design_matrices, PatsyError
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import FLOAT_DTYPES, check_random_state, check_is_fitted

from sklego.common import TrainOnlyTransformerMixin


class RandomAdder(TrainOnlyTransformerMixin, BaseEstimator):
    def __init__(self, noise=1, random_state=None):
        self.noise = noise
        self.random_state = random_state

    def fit(self, X, y):
        super().fit(X, y)
        X, y = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)
        self.dim_ = X.shape[1]

        return self

    def transform_train(self, X):
        rs = check_random_state(self.random_state)
        check_is_fitted(self, ['dim_'])

        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)

        return X + rs.normal(0, self.noise, size=X.shape)


class PatsyTransformer(TransformerMixin, BaseEstimator):
    """
    The patsy transformer offers a method to select the right columns
    from a dataframe as well as a DSL for transformations. It is inspired
    from R formulas.

    This is can be useful as a first step in the pipeline.

    :param formula: a patsy-compatible formula
    """

    def __init__(self, formula):
        self.formula = formula

    def fit(self, X, y=None):
        """Fits the estimator"""
        X_ = dmatrix(self.formula, X)
        assert np.array(X_).shape[0] == np.array(X).shape[0]
        self.design_info_ = X_.design_info
        return self

    def transform(self, X):
        """
        Applies the formula to the matrix/dataframe X.

        Returns an design array that can be used in sklearn pipelines.
        """
        check_is_fitted(self, 'design_info_')
        try:
            return build_design_matrices([self.design_info_], X)[0]
        except PatsyError as e:
            raise RuntimeError from e
