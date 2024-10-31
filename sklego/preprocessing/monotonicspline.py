import numpy as np
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import SplineTransformer
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted


def inter_bspline(p, k, x, y):
    num_rows, num_cols = y.shape
    z = np.zeros((num_rows, num_cols - 1))

    for j in range(num_cols - 1):
        denom_A = k[j + p - 1] - k[j]
        denom_B = k[j + p] - k[j + 1]

        A = (x - k[j]) / denom_A * y[:, j] if denom_A > 0 else 0
        B = (k[j + p] - x) / denom_B * y[:, j + 1] if denom_B > 0 else 0
        z[:, j] = A + B

    return z


def b_spline(p, knots, x):
    N = knots.shape[0]
    num_rows = x.shape[0]
    B = np.zeros((num_rows, N - 1))

    for j in range(N - 1):
        B[:, j] = np.where((x >= knots[j]) & (x < knots[j + 1]), 1, 0)

    for deg in range(1, p):
        B = inter_bspline(deg + 1, knots, x, B)

    return B


def i_spline(
    p,
    knots,
    x,
):
    bsp = b_spline(
        p,
        knots,
        x,
    )
    cumsummed = np.cumsum(
        bsp[:, ::-1],
        axis=1,
    )[:, :-1]

    return np.where(
        (x >= knots[0]) & (x <= knots[-1]),
        cumsummed.T,
        np.where(
            (x < knots[0]),
            0,
            1,
        ),
    ).T


class ISplineTransformer1D(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, p=4, n_knots=10, decreasing=False, knots=None):
        self.p = p
        self.knots = np.array(knots) if knots is not None else None
        self.n_knots = n_knots
        self.decreasing = decreasing

        self.x_min_ = None
        self.x_max_ = None
        self.knots_ = None

    def fit(self, X, y=None):
        X = check_array(X)
        X = (-1 if self.decreasing else 1) * X
        self.x_min_ = np.min(X) - 1e-6
        self.x_max_ = np.max(X) + 1e-6

        if self.knots is None:
            extended_knots = np.linspace(self.x_min_, self.x_max_, self.n_knots)
        else:
            extended_knots = self.knots

        self.knots_ = np.concatenate(
            [
                np.zeros(self.p) + self.x_min_,
                extended_knots,
                np.zeros(self.p) + self.x_max_,
            ],
        )

        return self

    def transform(self, X):
        return i_spline(
            self.p,
            self.knots_,
            (-1 if self.decreasing else 1) * X[:, 0],
        )


class MonotonicSplineTransformer(TransformerMixin, BaseEstimator):
    """The `MonotonicSplineTransformer` integrates the output of the `SplineTransformer` in an attempt to make monotonic features.

    This estimator is heavily inspired by [this blogpost](https://matekadlicsko.github.io/posts/monotonic-splines/) by Mate Kadlicsko.

    Parameters
    ----------
    n_knots : int, default=3
        The number of knots to use in the spline transformation.
    degree : int, default=3
    knots: str, default="uniform"

    Attributes
    ----------
    spline_transformer_ : trained SplineTransformer

    Examples
    --------
    ```py
    ```
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
        X = check_array(X, copy=True, force_all_finite=False, dtype=FLOAT_DTYPES, estimator=self)

        # If X contains infs, we need to replace them by nans before computing quantiles
        self.spline_transformer_ = {
            col: SplineTransformer(n_knots=self.n_knots, degree=self.degree, knots=self.knots).fit(
                X[:, col].reshape(-1, 1)
            )
            for col in range(X.shape[1])
        }
        self.sorted_X = {col: np.sort(X[:, col]) for col in range(X.shape[1])}
        self.sorted_X_output_ = {
            col: self.spline_transformer_[col].transform(np.sort(X[:, col]).reshape(-1, 1)).cumsum(axis=0)
            for col in range(X.shape[1])
        }
        self.sorted_idx_ = np.arange(X.shape[0]).astype(int)
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
        X = check_array(
            X,
            force_all_finite=False,
            dtype=FLOAT_DTYPES,
            estimator=self,
        )
        out = []
        for col in range(X.shape[1]):
            mapping = np.interp(X[:, col], self.sorted_X[col], self.sorted_idx_).astype(int)
            out.append(self.sorted_X_output_[col][mapping])
        return np.concatenate(out, axis=0)
