import narwhals.stable.v1 as nw
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn_compat.utils.validation import validate_data

from sklego.common import as_list


class OrthogonalTransformer(TransformerMixin, BaseEstimator):
    r"""The `OrthogonalTransformer` transforms the columns of a dataframe or numpy array to orthogonal (or
    orthonormal if `normalize=True`) matrix.

    It learns matrices $Q, R$ such that $X = Q \cdot R$, with $Q$ orthogonal, from which follows $Q = X \cdot R^{-1}$

    Parameters
    ----------
    normalize : bool, default=False
        Whether or not orthogonal matrix should be orthonormal as well.

    Attributes
    ----------
    inv_R_ : array-like of shape (n_features, n_features)
        The inverse of R of the QR decomposition of `X`.
    normalization_vector_ : array-like of shape (n_features,)
        The normalization terms to make the orthogonal matrix orthonormal.

    Examples
    --------
    ```py
    from sklearn.datasets import make_regression
    from sklego.preprocessing import OrthogonalTransformer

    # Generate a synthetic dataset
    X, y = make_regression(n_samples=100, n_features=3, noise=0.1, random_state=42)

    # Instantiate the transformer
    transformer = OrthogonalTransformer(normalize=True)

    # Fit the pipeline with the training data
    transformer.fit(X)

    # Transform the data using the fitted transformer
    X_transformed = transformer.transform(X)
    ```
    """

    def __init__(self, normalize=False):
        self.normalize = normalize

    def fit(self, X, y=None):
        """Fit the transformer to the input data by calculating the inverse of R of the QR decomposition of `X`.
        This can be used to calculate the orthogonal projection of `X`.

        If normalization is required, also stores a vector with normalization terms.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to fit.
        y : array-like of shape (n_samples,), default=None
            Ignored, present for compatibility.

        Returns
        -------
        self : OrthogonalTransformer
            The fitted transformer.
        """
        X = validate_data(self, X=X, ensure_min_samples=2, reset=True)

        # Q, R such that X = Q*R, with Q orthogonal, from which follows Q = X*inv(R)
        Q, R = np.linalg.qr(X)
        self.inv_R_ = np.linalg.inv(R)

        if self.normalize:
            self.normalization_vector_ = np.linalg.norm(Q, ord=2, axis=0)
        else:
            self.normalization_vector_ = np.ones((X.shape[1],))
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """Transforms `X` using the fitted inverse of R. Normalizes the result if required.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        array-like of shape (n_samples, n_features)
            The transformed data.
        """

        if self.normalize:
            check_is_fitted(self, ["inv_R_", "normalization_vector_"])
        else:
            check_is_fitted(self, ["inv_R_"])

        X = validate_data(self, X=X, reset=False)

        return X @ self.inv_R_ / self.normalization_vector_


def scalar_projection(vec, unto):
    return vec.dot(unto) / unto.dot(unto)


def vector_projection(vec, unto):
    return scalar_projection(vec, unto) * unto


class InformationFilter(TransformerMixin, BaseEstimator):
    r"""The `InformationFilter` transformer uses a variant of the
    [Gram-Schmidt process](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process) to filter information out of the
    dataset.

    This can be useful if you want to filter information out of a dataset because of fairness.

    To explain how it works: given a training matrix $X$ that contains columns $x_1, ..., x_k$.
    If we assume columns $x_1$ and $x_2$ to be the _sensitive_ columns then the information-filter will remove
    information by applying these transformations:

    $$\begin{split}
       v_1 & = x_1 \\
       v_2 & = x_2 - \frac{x_2 v_1}{v_1 v_1} \\
       v_3 & = x_3 - \frac{x_k v_1}{v_1 v_1} - \frac{x_2 v_2}{v_2 v_2} \\
           & ... \\
       v_k & = x_k - \frac{x_k v_1}{v_1 v_1} - \frac{x_2 v_2}{v_2 v_2}
       \end{split}$$

    Concatenating our vectors (but removing the sensitive ones) gives us a new training matrix

    $$X_{fair} = [v_3, ..., v_k]$$

    Parameters
    ----------
    columns : int | str | Sequence[int] | Sequence[str]
        The columns to filter out. This can be a sequence of either int (in the case of numpy) or string
        (in the case of pandas).
    alpha : float, default=1.0
        Parameter to control how much to filter:

        - `alpha=1` we filter out all information.
        - `alpha=0` we don't apply any filtering.

        Should be between 0 and 1.

    Attributes
    ----------
    projection_ : array-like of shape (n_features, n_features)
        The projection matrix that can be used to filter information out of a dataset.
    col_ids_ : List[int] of length `len(columns)`
        The list of column ids of the sensitive columns.

    Examples
    --------
    ```py
    import pandas as pd
    from sklego.preprocessing import InformationFilter

    df = pd.DataFrame({
        "user_id": [101, 102, 103],
        "length": [1.82, 1.85, 1.80],
        "age": [21, 37, 45]
    })

    InformationFilter(columns=["length", "age"], alpha=0.5).fit_transform(df)
    # array([[50.10152483,  3.87905643],
    #        [50.26253897, 19.59684308],
    #        [52.66084873, 28.06719867]])
    ```
    """

    _required_parameters = ["columns"]

    def __init__(self, columns, alpha=1):
        self.columns = columns
        self.alpha = alpha

    def _check_coltype(self, X):
        """Check if the `columns` type(s) are compatible with `X` type."""
        X_ = nw.from_native(X, strict=False, eager_only=True)
        for col in as_list(self.columns):
            if isinstance(col, str):
                if isinstance(X_, np.ndarray):
                    raise ValueError(f"column {col} is a string but datatype receive is numpy.")
                if isinstance(X_, nw.DataFrame):
                    if col not in X_.columns:
                        raise ValueError(f"column {col} is not in {X_.columns}")
            if isinstance(col, int):
                if col not in range(np.atleast_2d(np.array(X_)).shape[1]):
                    raise ValueError(f"column {col} is out of bounds for input shape {X_.shape}")

    def _col_idx(self, X, name):
        """Get the column index of a column name."""
        if isinstance(name, str):
            if isinstance(X, np.ndarray):
                raise ValueError("You cannot have a column of type string on a numpy input matrix.")
            return {name: i for i, name in enumerate(X.columns)}[name]
        return name

    def _make_v_vectors(self, X, col_ids):
        """Make the v vectors that we will use to filter out information."""
        vs = np.zeros((X.shape[0], len(col_ids)))
        for i, c in enumerate(col_ids):
            vs[:, i] = X[:, col_ids[i]]
            for j in range(0, i):
                vs[:, i] = vs[:, i] - vector_projection(vs[:, i], vs[:, j])
        return vs

    def fit(self, X, y=None):
        """Fit the transformer by learning the projection required to make the dataset orthogonal to sensitive
        columns.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to fit.
        y : array-like of shape (n_samples,), default=None
            Ignored, present for compatibility.

        Returns
        -------
        self : InformationFilter
            The fitted transformer.

        Raises
        ------
        ValueError
            If `columns` type(s) are incompatible with input data `X` type.
        """
        self._check_coltype(X)
        self.col_ids_ = [v if isinstance(v, int) else self._col_idx(X, v) for v in as_list(self.columns)]
        X = validate_data(self, X=X, reset=True)

        X_fair = X.copy()
        v_vectors = self._make_v_vectors(X, self.col_ids_)
        # gram smidt process but only on sensitive attributes
        for i, col in enumerate(X_fair.T):
            for v in v_vectors.T:
                X_fair[:, i] = X_fair[:, i] - vector_projection(X_fair[:, i], v)
        # we want to learn matrix P: X P = X_fair
        # this means we first need to create X_fair in order to learn P
        self.projection_, resid, rank, s = np.linalg.lstsq(X, X_fair, rcond=None)
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X):
        """Transforms `X` by applying the information filter.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        array-like of shape (n_samples, n_features)
            The transformed data.

        Raises
        ------
        ValueError
            If `columns` type(s) are incompatible with input data `X` type.
        """
        check_is_fitted(self, ["projection_", "col_ids_"])
        self._check_coltype(X)
        X = validate_data(self, X=X, reset=False)

        # apply the projection and remove the column we won't need
        X_fair = X @ self.projection_
        X_removed = np.delete(X_fair, self.col_ids_, axis=1)
        X_orig = np.delete(X, self.col_ids_, axis=1)
        return self.alpha * np.atleast_2d(X_removed) + (1 - self.alpha) * np.atleast_2d(X_orig)
