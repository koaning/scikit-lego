from warnings import warn

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn_compat.utils.validation import validate_data


class DictMapper(TransformerMixin, BaseEstimator):
    """The `DictMapper` transformer maps the values of columns according to the input `mapper` dictionary, fall back to
    the `default` value if the key is not present in the dictionary.

    Parameters
    ----------
    mapper : dict[..., int]
        The dictionary containing the mapping of the values.
    default : int
        The value to fall back to if the value is not in the mapper.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during `fit`.
    dim_ : int
        Deprecated, please use `n_features_in_` instead.

    Examples
    --------
    ```py
    import pandas as pd
    from sklego.preprocessing.dictmapper import DictMapper
    from sklearn.compose import ColumnTransformer

    X = pd.DataFrame({
        "city_pop": ["Amsterdam", "Leiden", "Utrecht", "None", "Haarlem"]
    })

    mapper = {
        "Amsterdam": 1_181_817,
        "Leiden": 130_181,
        "Utrecht": 367_984,
        "Haarlem": 165_396,
    }

    ct = ColumnTransformer([("dictmapper", DictMapper(mapper, 0), ["city_pop"])])
    X_trans = ct.fit_transform(X)
    X_trans
    # array([[1181817],
    #        [ 130181],
    #        [ 367984],
    #        [      0],
    #        [ 165396]])
    ```
    """

    _required_parameters = ["mapper", "default"]

    def __init__(self, mapper, default):
        self.mapper = mapper
        self.default = default

    def fit(self, X, y=None):
        """Checks the input data and records the number of features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to fit.
        y : array-like of shape (n_samples,), default=None
            Ignored, present for compatibility.

        Returns
        -------
        self : DictMapper
            The fitted transformer.
        """
        X = validate_data(self, X=X, copy=True, dtype=None, ensure_2d=True, ensure_all_finite=False, reset=True)
        return self

    def transform(self, X):
        """Performs the mapping on the column(s) of `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data for which the mapping will be applied.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            The data with the mapping applied.

        Raises
        ------
        ValueError
            If the number of columns from `X` differs from the number of columns when fitting.
        """
        check_is_fitted(self, ["n_features_in_"])
        X = validate_data(self, X=X, copy=True, dtype=None, ensure_2d=True, ensure_all_finite=False, reset=False)
        return np.vectorize(self.mapper.get, otypes=[int])(X, self.default)

    @property
    def dim_(self):
        warn(
            "Please use `n_features_in_` instead of `dim_`, `dim_` will be deprecated in future versions",
            DeprecationWarning,
        )
        return self.n_features_in_

    def _more_tags(self):
        return {"preserves_dtype": None, "allow_nan": True, "no_validation": True}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.transformer_tags.preserves_dtype = []
        tags.input_tags.allow_nan = True
        tags.no_validation = True
        return tags
