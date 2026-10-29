import numpy as np
import pandas as pd
import polars as pl
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from sklego.preprocessing import InformationFilter


@pytest.fixture(params=["dependent", "zero"])
def degenerate_data(request):
    if request.param == "dependent":
        X = np.array([[1, 2, 2], [2, 4, 1], [3, 6, 4], [4, 8, 3]])
        X_test = np.array([[5, 10, 6], [0, 0, 5]])
        return X, X_test, ["sensitive", "double_sensitive", "value"]
    X = np.array([[0, 2], [0, 1], [0, 4], [0, 3]])
    X_test = np.array([[0, 6], [0, 5]])
    return X, X_test, ["sensitive", "value"]


@pytest.mark.parametrize("backend", ["numpy", "pandas", "polars"])
def test_alpha_zero_keeps_degenerate_non_sensitive_values(degenerate_data, backend):
    X, X_test, names = degenerate_data
    columns = list(range(len(names) - 1))
    if backend == "numpy":
        train, test = X, X_test
    else:
        frame = pd.DataFrame if backend == "pandas" else pl.DataFrame
        train, test = (frame(dict(zip(names, values.T))) for values in (X, X_test))
        columns = names[:-1]

    # Fitting degenerate sensitive columns already emits numerical warnings.
    with np.errstate(invalid="ignore", divide="ignore"):
        transformer = InformationFilter(columns=columns, alpha=0).fit(train)
    assert_array_equal(transformer.transform(train), X[:, [-1]])
    assert_array_equal(transformer.transform(test), X_test[:, [-1]])


def test_alpha_zero_degenerate_pipeline(degenerate_data):
    X, X_test, names = degenerate_data
    pipeline = make_pipeline(
        InformationFilter(columns=list(range(len(names) - 1)), alpha=0),
        LinearRegression(),
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        pipeline.fit(X, 2 * X[:, -1])
    assert_allclose(pipeline.predict(X), 2 * X[:, -1], atol=1e-12)
    assert_allclose(pipeline.predict(X_test), 2 * X_test[:, -1], atol=1e-12)


@pytest.mark.parametrize(
    "fit_dtype, transform_dtype, expected_dtype",
    [(np.float32, np.float32, np.float32), (np.float32, np.float64, np.float64), (np.int64, np.float32, np.float64)],
)
def test_alpha_zero_preserves_output_dtype(fit_dtype, transform_dtype, expected_dtype):
    X = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
    transformer = InformationFilter(columns=[0], alpha=0).fit(X.astype(fit_dtype))
    result = transformer.transform(X.astype(transform_dtype))
    assert_array_equal(result, X[:, [1]])
    assert result.dtype == expected_dtype


def test_alpha_zero_requires_fitting():
    with pytest.raises(NotFittedError):
        InformationFilter(columns=[0], alpha=0).transform(np.ones((3, 2)))


def test_alpha_zero_checks_feature_count():
    transformer = InformationFilter(columns=[0], alpha=0).fit([[1, 2], [2, 1], [3, 4]])
    with pytest.raises(ValueError, match="features"):
        transformer.transform(np.ones((3, 3)))


def test_alpha_zero_checks_feature_names():
    X = pd.DataFrame({"sensitive": [1, 2, 3], "value": [2, 1, 4]})
    transformer = InformationFilter(columns=["sensitive"], alpha=0).fit(X)
    with pytest.raises(ValueError, match="feature names"):
        transformer.transform(X[["value", "sensitive"]])


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_alpha_zero_checks_finite_input(bad):
    transformer = InformationFilter(columns=[0], alpha=0).fit([[1, 2], [2, 1], [3, 4]])
    with pytest.raises(ValueError):
        transformer.transform([[1, bad]])


def test_alpha_zero_retains_learned_projection():
    X = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
    transformer = InformationFilter(columns=[0]).fit(X)
    expected_filtered = X[:, [1]] - X[:, [0]] * (14 / 15)
    assert_array_equal(transformer.set_params(alpha=0).transform(X), X[:, [1]])
    assert_allclose(transformer.set_params(alpha=1).transform(X), expected_filtered, atol=1e-12)
