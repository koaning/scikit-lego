try:
    import formulaic
except ImportError:
    from sklego.notinstalled import NotInstalledPackage

    formulaic = NotInstalledPackage("formulaic")

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class FormulaicTransformer(TransformerMixin, BaseEstimator):
    """The `FormulaicTransformer` offers a method to select the right columns from a dataframe as well as a DSL for
    transformations.

    It is inspired from R formulas. This is can be useful as a first step in the pipeline.

    Parameters
    ----------
    formula : str
        A formulaic-compatible formula.
        Refer to the [formulaic documentation](https://matthewwardrop.github.io/formulaic/guides/grammar/) for more
            details.
    return_type : Literal["pandas", "numpy", "sparse"], default="numpy"
        The type of the returned matrix.
        Refer to the [formulaic documentation](https://matthewwardrop.github.io/formulaic/guides/model_specs/) for more
            details.

    Attributes
    ----------
    formula_ : formulaic.Formula
        The parsed formula specification.
    model_spec_ : formulaic.ModelSpec
        The parsed model specification.
    n_features_in_ : int
        Number of features seen during `fit`.

    Examples
    --------
    ```py
    import formulaic
    import pandas as pd
    import numpy as np
    from sklego.preprocessing import FormulaicTransformer

    df = pd.DataFrame({
        'a': ['A', 'B', 'C'],
        'b': [0.3, 0.1, 0.2],
    })

    #default type of returned matrix - numpy
    FormulaicTransformer("a + b + a:b").fit_transform(df)
    # array([[1. , 0. , 0. , 0.3, 0. , 0. ],
    #        [1. , 1. , 0. , 0.1, 0.1, 0. ],
    #        [1. , 0. , 1. , 0.2, 0. , 0.2]])

    #pandas return type
    FormulaicTransformer("a + b + a:b", "pandas").fit_transform(df)
    #	Intercept	a[T.B]	a[T.C]	b	    a[T.B]:b	a[T.C]:b
    #0	1.0	        0	    0	    0.3	    0.0	        0.0
    #1	1.0	        1	    0	    0.1	    0.1	        0.0
    #2	1.0	        0	    1	    0.2	    0.0	        0.2
    ```
    """

    _required_parameters = ["formula"]

    def __init__(self, formula, return_type="numpy"):
        self.formula = formula
        self.return_type = return_type

    def fit(self, X, y=None):
        """Fit the `FormulaicTransformer` to the data by compiling the formula specification into a model spec.

        Parameters
        ----------
        X : pd.DataFrame of (n_samples, n_features)
            The data used to compile model spec.
        y : array-like of shape (n_samples,), default=None
            Ignored, present for compatibility.

        Returns
        -------
        self : FormulaicTransformer
            The fitted transformer.

        Raises
        ------
        ValueError
            If `formula` is not supported.
        """
        self.formula_ = formulaic.Formula.from_spec(self.formula)

        if self.formula_._has_structure:
            raise ValueError(
                f"Formula specification {repr(self.formula_)} results in a structured formula, which is not supported."
            )

        self.model_spec_ = self.formula_.get_model_matrix(X, output=self.return_type).model_spec
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X, y=None):
        """Transform `X` by generating a model matrix from it based on the fit model spec.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            The data for transformation will be applied.
        y: array-like of shape (n_samples,), default=None
            Ignored, present for compatibility.

        Returns
        -------
        X : array-like of shape (n_samples, n_features), and type `return_type`
            Transformed data.

        Raises
        ------
        ValueError
            If the number of columns from `X` differs from the number of columns when fitting.
        """

        check_is_fitted(self, ["formula_", "model_spec_", "n_features_in_"])

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "`X` must have the same number of columns in fit and transform. "
                f"Expected {self.n_features_in_}, found {X.shape[1]}."
            )

        X_ = self.model_spec_.get_model_matrix(X)
        return X_
