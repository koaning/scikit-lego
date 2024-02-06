# from sklearn.feature_selection._univariate_selection import _BaseFilter
import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import f_classif, f_regression
from sklearn.feature_selection._base import SelectorMixin
from sklearn.utils.validation import check_is_fitted, check_X_y


def _redundancy_pearson(X, selected, left):
    """Redundancy function for the MRMR feature selector algorithm

    Parameters
    ----------
    X : array-like, shape=(n_samples, n_features,)
        Training data. Used to compute redundancy of the training features.
    selected : array-like.
        List of indexes of the selected features at iteration i-th.
    left : array-like.
        List of indexes of the left features at iteration i-th. Mrmr will select a feature
        from this list.

    Returns
    -------
    np.ndarray, shape = (len(left), )
        The array containing the redundancy score using pearson correlation.
    """
    if len(selected) == 0:
        return np.ones(len(left))

    X_norm = X - np.mean(X, axis=0, keepdims=True)
    Xs = X_norm[:, selected]
    Xl = X_norm[:, left]

    num = (Xl[:, None, :] * Xs[:, :, None]).sum(axis=0)
    den = np.sqrt((Xl[:, None, :] ** 2).sum(axis=0)) * np.sqrt((Xs[:, :, None] ** 2).sum(axis=0))

    return np.sum(np.abs(np.nan_to_num(num / den, nan=np.finfo(float).eps)), axis=0)


class MaximumRelevanceMinimumRedundancy(SelectorMixin, BaseEstimator):
    """Maximum relevance minumum redundancy feature selector algoritm.
    This FeatureSelection algorithm works returning a subset of the original features set of len k.


    Parameters
    ----------
    k : int
        Number of feature the model should use.
    kind : str, optional
        'classification' or 'regression' or 'auto' if auto the model
        will try to infer the type of problem looking at the y data type, by default "auto".
    relevance_func : str | Callable, optional
        The relevance function to use, by default "f" (f_classif or  f_regression from sklearn.feature_selection)
    redundancy_func : str | Callable, optional
        The redundancy function to use, by default "p" (Pearson correlation)

    !! warning:
        If a custom relevance_func is provided it must have this firm:
        new_relevance(X: np.array, shape=(n_samples, n_features,), y np.array, shape = (n_samples,)):
        returns np.array, shape=(n_features, )
    !! warning:
        If a custom redundancy_func is provided it must have the same firm as the method _redundancy_pearson

    Attributes
    ----------
    _y_dtype : data type of y
    selected_features_ : array-like of shape (k,)
        Indexes of the selected features.
    scores_ : array-like of shape (k,)
        Scores of the selected features.

    Examples
    --------
    ```py
    from sklego.feature_selection.mrmr import MaximumRelevanceMinimumRedundancy

    mrmr =  MaximumRelevanceMinimumRedundancy(k=4,
            kind='auto',
            redundancy_func='p',
            relevance_func='f')

    X, y = ...

    # Fit mrmr model
    mrmr = mrmr.fit(X, y)

    # Selected features
    selected_features = mrmr.selected_features_

    # Get the scores of the selected features
    feature_scores = mrmr.scores_
    ```
    """

    def __init__(self, k, kind="auto", relevance_func="f", redundancy_func="p"):
        self.k = k
        self.kind = kind
        self.relevance_func = relevance_func
        self.redundancy_func = redundancy_func

    def _get_support_mask(self):
        """_summary_

        Returns
        -------
        np.ndarray
            Array of boolean, mask indicating if feature n is selected by mrmr or not.
        """
        check_is_fitted(self, ["selected_features_"])
        all_features = np.arange(0, self.n_features_in_)
        return np.isin(all_features, self.selected_features_)

    @property
    def _get_relevance(self):
        """get relevance function from init values."""
        if self.relevance_func == "f":
            if (self.kind == "auto" and np.issubdtype(self._y_dtype, np.integer)) | (self.kind == "classification"):
                return lambda X, y: np.nan_to_num(f_classif(X, y)[0])
            elif (self.kind == "auto" and np.issubdtype(self._y_dtype, np.floating)) | (self.kind == "regression"):
                return lambda X, y: np.nan_to_num(f_regression(X, y)[0])
            else:
                raise
        elif callable(self.relevance_func):
            return self.relevance_function
        else:
            raise

    @property
    def _get_redundancy(self):
        """get redundancy function from init values."""
        if self.redundancy_func == "p":
            return _redundancy_pearson
        elif callable(self.redundancy_func):
            return self.redundancy_func
        else:
            raise

    def fit(self, X, y):
        """Fit the underlying feature selection algorithm on the training data `X` and `y`
        using the provided redundancy and relevance functions.


        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : MaximumRelevanceMinimumRedundancy
            The fitted estimator.

        Raises
        ------
        ValueError
            if k parameter is not integer type.
        ValueError
            if k parameter is < n_features_in (X.shape[1])
        ValueError
            if k parameter < 1
        """
        self._y_dtype = y.dtype
        relevance = self._get_relevance
        redundancy = self._get_redundancy

        self.n_features_in_ = X.shape[1]
        left_features = list(range(self.n_features_in_))
        selected_features = []
        selected_scores = []

        if not isinstance(self.k, int):
            raise ValueError("k parameter mush be integer type")
        if self.k > self.n_features_in_:
            raise ValueError(f"k parameter mush be < n_features_in, got {self.k} - {self.n_features_in_}")
        elif self.k == self.n_features_in_:
            warnings.warn("k parameter is equal to n_features_in, no feature selection is applied")
            return np.asarray(left_features)
        elif self.k < 1:
            raise ValueError(f"k parameter mush be >= 1, got {self.k}")

        k = min(self.n_features_in_, self.k)

        X, y = check_X_y(X, y)

        # computed one time for all features
        rel_score = relevance(X, y)

        for i in range(k):
            red_i = redundancy(X, selected_features, left_features) / (i + 1)
            mrmr_score_i = rel_score[left_features] / red_i
            selected_index = np.argmax(mrmr_score_i)
            selected_features += [left_features.pop(selected_index)]
            selected_scores += [mrmr_score_i[selected_index]]

        self.selected_features_ = np.asarray(selected_features)
        self.scores_ = np.asarray(selected_scores)
        return self
