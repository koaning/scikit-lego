import narwhals.stable.v1 as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.io.arff import loadarff
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklego.preprocessing import InformationFilter


@pytest.fixture
def boston_data():
    loaded = loadarff("tests/data/boston.arff")[0]
    df = pd.DataFrame(loaded)
    return df.drop(columns=["MEDV"]).astype(float).values, df["MEDV"].values


@parametrize_with_checks([InformationFilter(columns=[0])])
def test_sklearn_compatible_estimator(estimator, check):
    if check.func.__name__ in {"check_estimators_empty_data_messages"}:
        pytest.skip("InformationFilter has custom message")
    check(estimator)


def test_v_columns_orthogonal(boston_data):
    X, y = boston_data
    ifilter = InformationFilter(columns=[11, 12]).fit(X, y)
    v_values = ifilter._make_v_vectors(X, [11, 12])
    assert v_values.prod(axis=1).sum() == pytest.approx(0, abs=1e-5)


def test_output_orthogonal(boston_data):
    X, y = boston_data
    X_fair = InformationFilter(columns=[11, 12]).fit_transform(X)
    assert all([(c * X[:, 11]).sum() < 1e-5 for c in X_fair.T])
    assert all([(c * X[:, 12]).sum() < 1e-5 for c in X_fair.T])


def test_alpha_param1(boston_data):
    X, y = boston_data
    ifilter = InformationFilter(columns=[11, 12], alpha=0.0)
    X_removed = np.delete(X, [11, 12], axis=1)
    assert np.isclose(ifilter.fit_transform(X), X_removed).all()


@pytest.mark.parametrize("frame_func", [pd.DataFrame, pl.DataFrame])
def test_alpha_param2(frame_func, boston_data):
    X, y = boston_data
    cols = [
        "crim",
        "zn",
        "indus",
        "chas",
        "nox",
        "rm",
        "age",
        "dis",
        "rad",
        "tax",
        "ptratio",
        "b",
        "lstat",
    ]
    df = frame_func(dict(zip(cols, X.T)))
    ifilter = InformationFilter(columns=["b", "lstat"], alpha=0.0)
    X_removed = nw.from_native(df).drop(["b", "lstat"]).to_numpy()
    assert np.isclose(ifilter.fit_transform(df), X_removed).all()


@pytest.mark.parametrize(
    "X, X_test, columns",
    [
        (
            np.array([[0, 2], [0, 1], [0, 4], [0, 3]]),
            np.array([[0, 6], [0, 5]]),
            [0],
        ),
        (
            np.array([[1, 2, 2], [2, 4, 1], [3, 6, 4], [4, 8, 3]]),
            np.array([[5, 10, 6], [0, 0, 5]]),
            [0, 1],
        ),
    ],
    ids=["zero-sensitive-column", "linearly-dependent-sensitive-columns"],
)
def test_alpha_zero_handles_degenerate_sensitive_columns(X, X_test, columns):
    # Zero-norm and linearly dependent sensitive columns make the learned projection contain NaNs. Since alpha=0
    # promises no filtering, transform must bypass that projection instead of letting `0 * NaN` corrupt the output.
    with np.errstate(invalid="ignore", divide="ignore"):
        transformer = InformationFilter(columns=columns, alpha=0).fit(X)

    result = transformer.transform(X)
    result_test = transformer.transform(X_test)

    assert_array_equal(result, X[:, [-1]])
    assert_array_equal(result_test, X_test[:, [-1]])
    assert result.dtype == np.result_type(X.dtype, transformer.projection_.dtype)
    assert result_test.dtype == np.result_type(X_test.dtype, transformer.projection_.dtype)


@pytest.mark.parametrize("frame_func", [pd.DataFrame, pl.DataFrame])
def test_output_orthogonal_frame(frame_func, boston_data):
    X, y = boston_data
    cols = [
        "crim",
        "zn",
        "indus",
        "chas",
        "nox",
        "rm",
        "age",
        "dis",
        "rad",
        "tax",
        "ptratio",
        "b",
        "lstat",
    ]
    df = frame_func(dict(zip(cols, X.T)))
    X_fair = InformationFilter(columns=["b", "lstat"]).fit_transform(df)
    assert all([(c * df["b"]).sum() < 1e-5 for c in X_fair.T])
    assert all([(c * df["lstat"]).sum() < 1e-5 for c in X_fair.T])


def test_integer_input_preserves_fractional_projection():
    X = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
    X_test = np.array([[2, 5], [5, 2]])
    coefficient = np.linalg.lstsq(X[:, [0]].astype(float), X[:, [1]], rcond=None)[0]
    expected = X[:, [1]] - X[:, [0]] @ coefficient
    expected_test = X_test[:, [1]] - X_test[:, [0]] @ coefficient

    transformer = InformationFilter(columns=[0])
    result = transformer.fit_transform(X)

    assert_allclose(result, expected, atol=1e-12)
    assert_allclose(transformer.transform(X_test), expected_test, atol=1e-12)
    assert np.issubdtype(result.dtype, np.floating)


@pytest.mark.parametrize("frame_func", [pd.DataFrame, pl.DataFrame])
def test_output_orthogonal_general_cols(frame_func, boston_data):
    X, y = boston_data
    cols = [
        "crim",
        "zn",
        "indus",
        "chas",
        "nox",
        "rm",
        "age",
        "dis",
        "rad",
        "tax",
        "ptratio",
        "b",
        "lstat",
    ]
    df = frame_func(dict(zip(cols, X.T)))
    for col in cols:
        X_fair = InformationFilter(columns=col).fit_transform(df)
        assert all([(c * df[col]).sum() < 1e-5 for c in X_fair.T])


def test_pipeline_gridsearch(boston_data):
    X, y = boston_data
    pipe = Pipeline([("info", InformationFilter(columns=[11, 12])), ("model", LinearRegression())])
    mod = GridSearchCV(estimator=pipe, param_grid={"info__columns": [[], [11], [12], [11, 12]]}, cv=2)
    assert pd.DataFrame(mod.fit(X, y).cv_results_).shape[0] == 4
