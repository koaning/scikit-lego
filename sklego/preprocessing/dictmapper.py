import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from functools import partial


class DictMapper(TransformerMixin, BaseEstimator):
    """
    Map the values of values of columns according to the input dictionary,
    fall back to the default if the key is not present in the dictionary.

    :param mapper: The dictionary containing the mapping of the values
    :param default: The value to fall back to if the value is not in the mapper
    """

    def __init__(self, mapper, default):
        self.mapper = mapper
        self.default = default

    def fit(self, X, y=None):
        """
        Checks the input dataframe and records the shape of it

        :type X: pandas.DataFrame or numpy.ndarray
        :param X: The column(s) from which the mapping will be applied

        :param y: Ignored.

        :rtype: sklego.preprocessing.DictMapper
        :returns: The fitted object.
        """
        X = check_array(
            X,
            copy=True,
            estimator=self,
            force_all_finite=True,
            dtype=None,
            ensure_2d=True,
        )
        self.dim_ = X.shape[1]
        return self

    def transform(self, X):
        """
        Performs the mapping on the column(s) of ``X``.

        :type X: pandas.DataFrame or numpy.ndarray
        :param X: The column(s) for which the mapping will be applied.

        :rtype: numpy.ndarray
        :returns: ``X`` values with the mapping applied

        :raises:
            ``ValueError`` if the number of columns from ``X`` differs from the
            number of columns when fitting
        """
        check_is_fitted(self, ["dim_"])
        X = check_array(
            X,
            copy=True,
            estimator=self,
            force_all_finite=True,
            dtype=None,
            ensure_2d=True,
        )

        if X.shape[1] != self.dim_:
            raise ValueError(
                f"number of columns {X.shape[1]} does not match fit size {self.dim_}"
            )
        return np.vectorize(self.mapper.get, otypes=[int])(X, self.default)
