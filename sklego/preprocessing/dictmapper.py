from warnings import warn

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


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
    """

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
        X = check_array(
            X,
            copy=True,
            estimator=self,
            force_all_finite=True,
            dtype=None,
            ensure_2d=True,
        )
        self.n_features_in_ = X.shape[1]
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
        X = check_array(
            X,
            copy=True,
            estimator=self,
            force_all_finite=True,
            dtype=None,
            ensure_2d=True,
        )

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"number of columns {X.shape[1]} does not match fit size {self.n_features_in_}")
        return np.vectorize(self.mapper.get, otypes=[int])(X, self.default)

    @property
    def dim_(self):
        warn(
            "Please use `n_features_in_` instead of `dim_`, `dim_` will be deprecated in future versions",
            DeprecationWarning,
        )
        return self.n_features_in_
