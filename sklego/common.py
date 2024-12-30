import collections
import hashlib
from warnings import warn

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn_compat.utils.validation import validate_data


class TrainOnlyTransformerMixin(TransformerMixin, BaseEstimator):
    """Mixin class for transformers that can handle training and test data differently.

    This mixin allows using a separate function for transforming training and test data.

    !!! warning

        Transformers using this class as a mixin should:

        - Call `super().fit` in their fit method.
        - Implement `transform_train()` method.

        They may also implement `transform_test()` method, if not implemented, `transform_test()` will simply return
        the untransformed data.

    Attributes
    ----------
    X_hash_ : hash
        The hash of the training data - used to determine whether to use `transform_train` or `transform_test`.
    n_features_in_ : int
        The number of features seen during `.fit()` in the training data.
    dim_ : int
        Deprecated, use `n_features_in_` instead.

    Examples
    --------
    ```py
    from sklearn.base import BaseEstimator
    from sklego.common import TrainOnlyTransformerMixin

    class TrainOnlyTransformer(TrainOnlyTransformerMixin, BaseEstimator):
        def fit(self, X, y):
            super().fit(X, y)

        def transform_train(self, X, y=None):
            '''Add random noise to the training data.'''
            return X + np.random.normal(0, 1, size=X.shape)

    X_train, X_test = np.random.randn(100, 4), np.random.randn(100, 4)
    y_train, y_test = np.random.randn(100), np.random.randn(100)

    trf = TrainOnlyTransformer()
    trf.fit(X_train, y_train)

    assert np.all(trf.transform(X_train) != X_train)
    assert np.all(trf.transform(X_test) == X_test)
    ```
    """

    _HASHERS = {
        pd.DataFrame: lambda X: hashlib.sha256(pd.util.hash_pandas_object(X, index=True).to_numpy()).hexdigest(),
        np.ndarray: lambda X: hash(X.data.tobytes()),
        np.memmap: lambda X: hash(X.data.tobytes()),
    }

    def fit(self, X, y=None):
        """Fit the mixin by calculating the hash of `X` and stores it in `self.X_hash_`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features )
            The training data.
        y : array-like of shape (n_samples,) | None, default=None
            The target values.

        Returns
        -------
        self : TrainOnlyTransformerMixin
            The fitted transformer.
        """
        if y is None:
            validate_data(self, X=X, reset=True)
        else:
            validate_data(self, X=X, y=y, multi_output=True, reset=True)

        self.X_hash_ = self._hash(X)
        return self

    @staticmethod
    def _hash(X):
        """Calculate a hash of X based on its type. Hashers are defined in TrainOnlyMixin._HASHERS.

        Supported types are:

            - pd.DataFrame
            - np.ndarray
            - np.memmap

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to hash.

        Returns
        -------
        hash
            The calculated hash value.

        Raises
        ------
        ValueError
            If the type of `X` is not supported.
        """
        try:
            hasher = TrainOnlyTransformerMixin._HASHERS[type(X)]
        except KeyError:
            raise ValueError(
                f"Unknown datatype {type(X)}, "
                f"`TrainOnlyTransformerMixin` only supports: {set(TrainOnlyTransformerMixin._HASHERS.keys())}"
            )
        else:
            return hasher(X)

    def transform(self, X, y=None):
        """Dispatch to `transform_train()` or `transform_test()` based on the data passed.

        This method will check whether the hash of `X` matches the hash of the training data. If it does, it will
        dispatch to `transform_train()`, otherwise it will dispatch to `transform_test()`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.
        y : array-like of shape (n_samples,) or None, default=None.
            The target values.

        Returns
        -------
        array-like of shape (n_samples, n_features)
            The transformed data.

        Raises
        ------
        ValueError
            If the input dimension does not match the training dimension.
        """
        check_is_fitted(self, ["X_hash_", "n_features_in_"])
        validate_data(self, X=X, reset=False)

        if self._hash(X) == self.X_hash_:
            return self.transform_train(X)
        else:
            return self.transform_test(X)

    def transform_train(self, X, y=None):
        """Transform the training data.

        This method should be implemented in subclasses to specify how training data should be transformed.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features )
            The training data.
        y : array-like of shape (n_samples,) or None, default=None
            The target values.

        Returns
        -------
        array-like of shape (n_samples, n_features)
            The transformed training data.
        """
        raise NotImplementedError("Subclasses of `TrainOnlyTransformerMixin` should implement `transform_train` method")

    def transform_test(self, X, y=None):
        """Transform the test data.

        This method can be implemented in subclasses to specify how test data should be transformed.
        If not implemented, it will return the untransformed data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The test data.
        y : array-like of shape (n_samples,) or None, default=None
            The target values.

        Returns
        -------
        array-like of shape (n_samples, n_features)
            The transformed test data or untransformed data if not implemented.
        """
        return X

    @property
    def dim_(self):
        warn(
            "Please use `n_features_in_` instead of `dim_`, `dim_` will be deprecated in future versions",
            DeprecationWarning,
        )
        return self.n_features_in_


def as_list(val):
    """Ensure the input value is converted into a list.

    This helper function takes an input value and ensures that it is always returned as a list.

    - If the input is a single value, it will be wrapped in a list.
    - If the input is an iterable, it will be converted into a list.

    Parameters
    ----------
    val : object
        The input value that needs to be converted into a list.

    Returns
    -------
    list
        The input value as a list.

    Examples
    --------
    ```py
    as_list("test")
    # ['test']

    as_list(["test1", "test2"])
    # ['test1', 'test2']
    ```
    """
    treat_single_value = str

    if isinstance(val, treat_single_value):
        return [val]

    if hasattr(val, "__iter__"):
        return list(val)

    return [val]


def flatten(nested_iterable):
    """Recursively flatten an arbitrarily nested iterable into an iterator of values.

    This helper function takes an arbitrarily nested iterable and returns an iterator of flattened values.
    It recursively processes the input to extract individual elements and yield them in a flat structure.

    Parameters
    ----------
    nested_iterable : Iterable
        An arbitrarily nested iterable to be flattened.

    Yields
    ------
    Generator
        A generator of flattened values from the nested iterable.

    Examples
    --------
    ```py
    list(flatten([["test1", "test2"], ["a", "b", ["c", "d"]]))
    # ['test1', 'test2', 'a', 'b', 'c', 'd']

    list(flatten(["test1", ["test2"]])
    # ['test1', 'test2']
    ```
    """
    for el in nested_iterable:
        if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def expanding_list(list_to_extent, return_type=list):
    """Create an expanding list of lists or tuples by making combinations of elements.

    This function takes an input list and creates an expanding list, where each element is a list or tuple containing a
    subset of elements from the input list. The resulting list can be composed of lists or tuples, depending on the
    specified `return_type`.

    Parameters
    ----------
    list_to_extent : object
        The input to be extended.
    return_type : type, default=list
        The type of elements in the resulting list (list or tuple).

    Returns
    -------
    list
        An expanding list of `list`s or `tuple`s containing combinations of elements from the input.

    Examples
    --------
    ```py
    expanding_list("test")
    # [['test']]

    expanding_list(["test1", "test2", "test3"])
    # [['test1'], ['test1', 'test2'], ['test1', 'test2', 'test3']]

    expanding_list(["test1", "test2", "test3"], tuple)
    # [('test1',), ('test1', 'test2'), ('test1', 'test2', 'test3')]
    ```
    """
    listed = as_list(list_to_extent)
    return [return_type(listed[: n + 1]) for n in range(len(listed))]


def sliding_window(sequence, window_size, step_size):
    """Generate sliding windows over a sequence.

    This function generates sliding windows of a specified size over a given sequence, where each window is a list of
    elements from the sequence.

    Parameters
    ----------
    sequence : Iterable
        The input sequence (e.g., a list).
    window_size : int
        The size of each sliding window.
    step_size : int
        The amount of steps to the next window.

    Returns
    -------
    Generator
        A generator object that yields sliding windows.

    Examples
    --------
    ```py
    list(sliding_window([1, 2, 4, 5], 2, 1))
    # [[1, 2], [2, 4], [4, 5], [5]]
    ```
    """
    return (sequence[pos : pos + window_size] for pos in range(0, len(sequence), step_size))
