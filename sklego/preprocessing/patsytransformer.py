import numpy as np
from patsy import PatsyError, build_design_matrices, dmatrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class PatsyTransformer(TransformerMixin, BaseEstimator):
    """The `PatsyTransformer` offers a method to select the right columns from a dataframe as well as a DSL for
    transformations.

    It is inspired from R formulas. This is can be useful as a first step in the pipeline.

    Parameters
    ----------
    formula : str
        A patsy-compatible formula.
    return_type : Literal["matrix", "dataframe"], default="matrix"
        Either "matrix" or "dataframe", passed on to patsy.

    Attributes
    ----------
    design_info_ : [patsy.DesignInfo](https://patsy.readthedocs.io/en/latest/API-reference.html#patsy.DesignInfo)
        A DesignInfo object holds metadata about a design matrix.
    """

    def __init__(self, formula, return_type="matrix"):
        self.formula = formula
        self.return_type = return_type

    def fit(self, X, y=None):
        """Fits the transformer on input data `X` by constructing a design matrix given the `formula`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to fit.
        y : array-like of shape (n_samples,), default=None
            Ignored, present for compatibility.

        Returns
        -------
        self : PatsyTransformer
            The fitted transformer.
        """
        X_ = dmatrix(self.formula, X, NA_action="raise", return_type=self.return_type)

        # check the number of observations hasn't changed. This ought not to be necessary given NA_action='raise' above
        # but just to be safe
        assert np.array(X_).shape[0] == np.array(X).shape[0]
        self.design_info_ = X_.design_info
        return self

    def transform(self, X):
        """Transform `X` by applying the fitted formula.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        patsy.DesignMatrix | pd.DataFrame

            - DesignMatrix if return_type="matrix" (the default)
            - pd.DataFrame if return_type="dataframe"
        """
        check_is_fitted(self, "design_info_")
        try:
            return build_design_matrices([self.design_info_], X, return_type=self.return_type)[0]
        except PatsyError as e:
            raise RuntimeError from e
