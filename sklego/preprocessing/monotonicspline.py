import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import SplineTransformer
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted
from sklearn_compat.utils.validation import validate_data


class MonotonicSplineTransformer(TransformerMixin, BaseEstimator):
    """The `MonotonicSplineTransformer` integrates the output of the `SplineTransformer` in an attempt to make monotonic features.

    This estimator is heavily inspired by [this blogpost](https://matekadlicsko.github.io/posts/monotonic-splines/) by Mate Kadlicsko.

    Parameters
    ----------
    n_knots : int, default=3
        The number of knots to use in the spline transformation.
    degree : int, default=3
        The polynomial degree to use in the spline transformation
    knots : Literal['uniform', 'quantile'], default="uniform"
        Knots argument of spline transformer

    Attributes
    ----------
    spline_transformer_ : trained SplineTransformer
    features_in_ : int
        The number of features seen in the training data.

    """

    def __init__(self, n_knots=3, degree=3, knots="uniform"):
        self.n_knots = n_knots
        self.degree = degree
        self.knots = knots

    def fit(self, X, y=None):
        """Fit the `MonotonicSplineTransformer` transformer by computing the spline transformation of `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.
        y : array-like of shape (n_samples,), default=None
            Ignored, present for compatibility.

        Returns
        -------
        self : MonotonicSplineTransformer
            The fitted transformer.

        Raises
        ------
        ValueError
            If `X` contains non-numeric columns.
        """
        X = validate_data(self, X=X, copy=True, ensure_all_finite=False, dtype=FLOAT_DTYPES, reset=True)
        # If X contains infs, we need to replace them by nans before computing quantiles
        self.spline_transformer_ = {
            col: SplineTransformer(n_knots=self.n_knots, degree=self.degree, knots=self.knots).fit(
                X[:, col].reshape(-1, 1)
            )
            for col in range(X.shape[1])
        }
        return self

    def transform(self, X):
        """Performs the Ispline transformation on `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X : np.ndarray of shape (n_samples, n_out)
            Transformed `X` values.

        Raises
        ------
        ValueError
            If the number of columns from `X` differs from the number of columns when fitting.
        """
        check_is_fitted(self, "spline_transformer_")
        X = validate_data(self, X=X, ensure_all_finite=False, dtype=FLOAT_DTYPES, reset=False)

        out = []
        for col in range(X.shape[1]):
            out.append(
                np.cumsum(
                    self.spline_transformer_[col].transform(X[:, [col]])[:, ::-1],
                    axis=1,
                )
            )
        return np.concatenate(out, axis=1)
