import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import f_classif, f_regression
from sklearn.feature_selection._base import SelectorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn_compat.utils.validation import validate_data


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
    # if len(selected) == 0:
    #     return np.ones(len(left))

    X_norm = X - np.mean(X, axis=0, keepdims=True)
    Xs = X_norm[:, selected]
    Xl = X_norm[:, left]

    num = (Xl[:, None, :] * Xs[:, :, None]).sum(axis=0)
    den = np.sqrt((Xl[:, None, :] ** 2).sum(axis=0)) * np.sqrt((Xs[:, :, None] ** 2).sum(axis=0))

    return np.sum(np.abs(np.nan_to_num(num / den, nan=np.finfo(float).eps)), axis=0)


class MaximumRelevanceMinimumRedundancy(SelectorMixin, BaseEstimator):
    r"""Maximum Relevance Minimum Redundancy (MRMR) is an iterative feature selection method commonly used in data
    science to select a subset of features from a larger feature set. The goal of MRMR is to choose features that
    have high relevance to the target variable while minimizing redundancy among the already selected features.

    How MRMR works:

    1. Compute the relevance of each feature to the target variable: The relevance of a feature is typically
    measured using a metric such as mutual information, correlation coefficient, or another appropriate measure of
    dependence between the feature and the target variable.

    2. Compute the redundancy between each pair of features: Redundancy is the degree of similarity or overlap between
    features. It can be measured using metrics such as mutual information, correlation coefficient, or other similarity
    measures.

    3. Select features based on the maximum relevance and minimum redundancy criteria: MRMR aims to maximize the
    relevance of selected features to the target variable while minimizing redundancy among them.

    4. Construct the final subset of features: MRMR iteratively adds features to the selected subset until a predefined
    number of features is reached.

    The implemented formula is:

    $$\text{score}_{i}(f) = \frac{\text{relevance}(f | y)}{\text{redundancy}(f | \text{selected}_{i-1})}$$

    !!! warning
        If a custom relevance_func is provided it must have this signature:
        `Callable[[np.ndarray, np.ndarray], np.ndarray]`
        It should accept X, y as arguments and it should compute the score for each feature of X
        and return an array of shape (n_features_in_,).

    !!! warning
        If a custom redundancy_func is provided it must have the same signature as the method _redundancy_pearson, hence
        the function must have three parameters:

            - X : array-like, shape=(n_samples, n_features,). Training used to compute redundancy of the training features.

            - selected : array-like. List of indexes of the selected features at iteration i-th.

            - left : array-like. List of indexes of the left features at iteration i-th. Mrmr will select a feature from this list.

        and it must return:

            - np.ndarray, shape = (len(left), ), The array containing the redundancy score using the custom function.

    !!! info "New in version 0.8.0"

    Parameters
    ----------
    k : int
        Number of feature the model should use.
    relevance_func : str | Callable, default="f"
        The relevance function to use. The default maps to scikit-learn [f_classif](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html) or  [f_regression](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html) for classification or regression (resp.)
    redundancy_func : str | Callable, default="p"
        The redundancy function to use. The default maps to Pearson correlation computed for each remaining features.
    kind : Literal["auto", "classficiation", "regression"], default="auto".
        'classification' or 'regression' or 'auto' if auto the model
        will try to infer the type of problem looking at the y data type, by default "auto".

    Attributes
    ----------
    _y_dtype : np.dtype
        data type of y
    selected_features_ : array-like of shape (k,)
        Indexes of the selected features.
    scores_ : array-like of shape (k,)
        Scores of the selected features.

    Examples
    --------
    ```py
    from sklego.feature_selection import MaximumRelevanceMinimumRedundancy
    from sklearn.datasets import make_classification

    mrmr =  MaximumRelevanceMinimumRedundancy(k=4,
            kind='auto',
            redundancy_func='p',
            relevance_func='f')

    X, y = make_classification(n_features=4)

    # Fit mrmr model
    mrmr = mrmr.fit(X, y)

    # Selected features
    selected_features = mrmr.selected_features_

    # Get the scores of the selected features
    feature_scores = mrmr.scores_
    ```
    """

    _required_parameters = ["k"]

    def __init__(self, k, *, relevance_func="f", redundancy_func="p", kind="auto"):
        self.k = k
        self.relevance_func = relevance_func
        self.redundancy_func = redundancy_func
        self.kind = kind

    def _get_support_mask(self):
        """SelectorMixin base function to get the selected features mask

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
                raise ValueError(
                    "`kind` parameter must be 'auto', 'classification' or 'regression' and y dtype must be numeric"
                )
        elif callable(self.relevance_func):
            return self.relevance_func
        else:
            raise ValueError(f"Relevance function supported are 'f' or Callable, got {self.relevance_func}")

    @property
    def _get_redundancy(self):
        """get redundancy function from init values."""
        if self.redundancy_func == "p":
            return _redundancy_pearson
        elif callable(self.redundancy_func):
            return self.redundancy_func
        else:
            raise ValueError(f"Redundancy function supported are 'p' or Callable, got {self.redundancy_func}")

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
            if:

                k parameter is not integer type or is < n_features_in (X.shape[1]) or < 1
        """
        X, y = validate_data(self, X=X, y=y, dtype="numeric", y_numeric=True, reset=True)
        self._y_dtype = y.dtype

        relevance = self._get_relevance
        redundancy = self._get_redundancy

        left_features = list(range(self.n_features_in_))
        selected_features = []
        selected_scores = []

        if not isinstance(self.k, int):
            raise ValueError("k parameter must be integer type")
        if self.k > self.n_features_in_:
            raise ValueError(f"k ({self.k}) parameter must be less than n_features_in_ ({self.n_features_in_})")
        elif self.k == self.n_features_in_:
            warnings.warn("k parameter is equal to n_features_in, no feature selection is applied")
            return np.asarray(left_features)
        elif self.k < 1:
            raise ValueError(f"k ({self.k}) parameter must be greater than or equal to 1")

        # computed one time for all features

        rel_score = relevance(X, y)

        for i in range(self.k):
            red_i = redundancy(X, selected_features, left_features) / i if i > 0 else 1
            mrmr_score_i = rel_score[left_features] / red_i
            selected_index = np.argmax(mrmr_score_i)
            selected_features += [left_features.pop(selected_index)]
            selected_scores += [mrmr_score_i[selected_index]]
        self.selected_features_ = np.asarray(selected_features)
        self.scores_ = np.asarray(selected_scores)
        return self
