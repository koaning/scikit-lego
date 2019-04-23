import collections
import hashlib

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y


class TrainOnlyTransformerMixin(TransformerMixin):
    """
    Allows using a separate function for transforming train and test data

    Usage:
        >>> from sklearn.base import BaseEstimator
        >>> class TrainOnlyTransformer(TrainOnlyTransformerMixin, BaseEstimator):
        ...     def fit(self, X, y):
        ...         super().fit(X, y)
        ...
        ...     def transform_train(self, X, y=None):
        ...          return X + np.random.normal(0, 1, size=X.shape)
        ...
        >>> X_train, X_test = np.random.randn(100, 4), np.random.randn(100, 4)
        >>> y_train, y_test = np.random.randn(100), np.random.randn(100)
        >>>
        >>> trf = TrainOnlyTransformer()
        >>> trf.fit(X_train, y_train)
        >>>
        >>> assert np.all(trf.transform(X_train) != X_train)
        >>> assert np.all(trf.transform(X_test) == X_test)

    .. warning:: Transformers using this class as a mixin should at a minimum:

        - call `super().fit` in their fit method
        - implement `transform_train()`

        They may also implement `transform_test()`. If it is not implemented,
        `transform_test` will simply return the untransformed dataframe
    """

    _HASHERS = {
        pd.DataFrame: lambda X: hashlib.sha256(pd.util.hash_pandas_object(X, index=True).values).hexdigest(),
        np.ndarray: lambda X: hash(X.data.tobytes()),
        np.memmap: lambda X: hash(X.data.tobytes()),

    }

    def fit(self, X, y):
        """Calculates the hash of X_train"""
        check_X_y(X, y, estimator=self)
        self.X_hash_ = self._hash(X)
        self.dim_ = X.shape[1]
        return self

    @staticmethod
    def _hash(X):
        """Returns a hash of X based on the type of X. Hashers are defined in TrainOnlyMixin.HASHERS"""
        try:
            hasher = TrainOnlyTransformerMixin._HASHERS[type(X)]
        except KeyError:
            raise ValueError(f'Unknown datatype {type(X)}, '
                             f'TransformerSelector only supports {TrainOnlyTransformerMixin.HASHERS.keys()}')
        else:
            return hasher(X)

    def transform(self, X, y=None):
        """
        Dispatcher for transform method.

        It will dispatch to `self.transform_train` if X is the same as X passed to `fit`, otherwise, it will dispatch
        to `self.trainsform_test`
        """
        check_is_fitted(self, ['X_hash_', 'dim_'])
        check_array(X, estimator=self)

        if X.shape[1] != self.dim_:
            raise ValueError(f'Unexpected input dimension {X.shape[1]}, expected {self.dim_}')

        if self._hash(X) == self.X_hash_:
            return self.transform_train(X)
        else:
            return self.transform_test(X)

    def transform_train(self, X, y=None):
        raise NotImplementedError('Subclasses of TrainOnlyMixin should implement `transform_train`')

    def transform_test(self, X, y=None):
        return X


def as_list(val):
    """
    Helper function, always returns a list of the input value.

    :param val: the input value.
    :returns: the input value as a list.

    :Example:

    >>> as_list('test')
    ['test']

    >>> as_list(['test1', 'test2'])
    ['test1', 'test2']
    """
    treat_single_value = (str)

    if isinstance(val, treat_single_value):
        return [val]

    if hasattr(val, '__iter__'):
        return list(val)

    return [val]


def flatten(nested_iterable):
    """
    Helper function, returns an iterator of flattened values from an arbitrarily nested iterable

    >>> list(flatten([['test1', 'test2'], ['a', 'b', ['c', 'd']]]))
    ['test1', 'test2', 'a', 'b', 'c', 'd']

    >>> list(flatten(['test1', ['test2']]))
    ['test1', 'test2']
    """
    for el in nested_iterable:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el
