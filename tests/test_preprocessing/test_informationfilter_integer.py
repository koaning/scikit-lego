import numpy as np
import pandas as pd
import polars as pl
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.pipeline import make_pipeline

from sklego.preprocessing import InformationFilter


@pytest.mark.parametrize("dtype", [np.int64, np.float64])
@pytest.mark.parametrize("backend", ["numpy", "pandas", "polars"])
def test_projection_matches_least_squares_residuals(dtype, backend):
    X = np.array([[1, 2], [2, 1], [3, 4], [4, 3]], dtype=dtype)
    X_test = np.array([[2, 5], [5, 2]], dtype=dtype)
    original = X.copy()
    # A least-squares residual is independent of the filter's Gram-Schmidt implementation.
    coefficients = np.linalg.lstsq(X[:, [0]].astype(float), X[:, [1]], rcond=None)[0]
    expected = X[:, [1]] - X[:, [0]] @ coefficients
    expected_test = X_test[:, [1]] - X_test[:, [0]] @ coefficients

    if backend == "numpy":
        train, test, columns = X, X_test, [0]
    else:
        frame = pd.DataFrame if backend == "pandas" else pl.DataFrame
        train = frame({"sensitive": X[:, 0], "value": X[:, 1]})
        test = frame({"sensitive": X_test[:, 0], "value": X_test[:, 1]})
        columns = ["sensitive"]

    transformer = clone(InformationFilter(columns=columns))
    pipeline = make_pipeline(transformer)
    result = pipeline.fit_transform(train)
    assert_allclose(result, expected, atol=1e-12)
    assert_allclose(X[:, [0]].T @ result, 0, atol=1e-12)
    assert_allclose(pipeline.transform(test), expected_test, atol=1e-12)
    assert_array_equal(X, original)


def test_integer_projection_with_multiple_sensitive_columns():
    X = np.array([[1, 0, 2], [0, 2, 1], [2, 1, 4], [1, 3, 3], [3, 2, 1]])
    sensitive = X[:, :2].astype(float)
    expected = X[:, [2]] - sensitive @ np.linalg.lstsq(sensitive, X[:, [2]], rcond=None)[0]
    result = InformationFilter(columns=[0, 1]).fit_transform(X)
    assert_allclose(result, expected, atol=1e-12)
    assert_allclose(sensitive.T @ result, 0, atol=1e-12)


@pytest.mark.parametrize("alpha", [0.0, 0.5])
def test_integer_projection_blending(alpha):
    X = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
    residual = X[:, [1]] - X[:, [0]] * (14 / 15)
    expected = alpha * residual + (1 - alpha) * X[:, [1]]
    assert_allclose(InformationFilter(columns=[0], alpha=alpha).fit_transform(X), expected, atol=1e-12)
