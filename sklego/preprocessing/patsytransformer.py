try:
    import patsy
except ImportError:
    from sklego.notinstalled import NotInstalledPackage

    patsy = NotInstalledPackage("patsy")

import numpy as np
from deprecated import deprecated
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


@deprecated(
    version="0.6.17",
    reason="Please use `sklego.preprocessing.FormulaicTransformer` instead. "
    "This object will be removed from the preprocessing submodule in version 0.9.0.",
)
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
        X_ = patsy.dmatrix(
            self.formula, X, NA_action="raise", return_type=self.return_type
        )

        # check the number of observations hasn't changed. This ought not to
        # be necessary given NA_action='raise' above but just to be safe
        if np.asarray(X_).shape[0] != np.asarray(X).shape[0]:
            raise RuntimeError(
                "Number of observations has changed during fit. "
                "This is likely because some rows have been removed "
                "due to NA values."
            )
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
            return patsy.build_design_matrices(
                [self.design_info_], X, return_type=self.return_type
            )[0]
        except patsy.PatsyError as e:
            raise RuntimeError from e
