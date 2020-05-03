import numpy as np
from patsy import dmatrix, build_design_matrices, PatsyError
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class PatsyTransformer(TransformerMixin, BaseEstimator):
    """
    The patsy transformer offers a method to select the right columns
    from a dataframe as well as a DSL for transformations. It is inspired
    from R formulas.
    This is can be useful as a first step in the pipeline.
    :param formula: a patsy-compatible formula
    :return_type: Either "matrix" or "dataframe", passed on to patsy
    """

    def __init__(self, formula, return_type="matrix"):
        self.formula = formula
        self.return_type = return_type

    def fit(self, X, y=None):
        """Fits the estimator"""
        X_ = dmatrix(self.formula, X, NA_action="raise", return_type=self.return_type)

        # check the number of observations hasn't changed. This ought not to
        # be necessary given NA_action='raise' above but just to be safe
        assert np.array(X_).shape[0] == np.array(X).shape[0]
        self.design_info_ = X_.design_info
        return self

    def transform(self, X):
        """
        Applies the formula to the matrix/dataframe X.

        Returns
        - A patsy.DesignMatrix, if return_type="matrix" (the default)
        - A pandas.DataFrame, if return_type="dataframe"
        """
        check_is_fitted(self, "design_info_")
        try:
            return build_design_matrices(
                [self.design_info_], X, return_type=self.return_type
            )[0]
        except PatsyError as e:
            raise RuntimeError from e
