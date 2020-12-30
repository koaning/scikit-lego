import pytest
import numpy as np
from sklearn.utils import estimator_checks

from sklego.common import flatten
from sklego.preprocessing import IntervalEncoder
from tests.conftest import transformer_checks, general_checks


@pytest.mark.parametrize(
    "test_fn",
    flatten(
        [
            transformer_checks,
            general_checks,
            estimator_checks.check_estimators_dtypes,
            estimator_checks.check_fit_score_takes_y,
            # estimator_checks.check_dtype_object,
            estimator_checks.check_sample_weights_pandas_series,
            estimator_checks.check_sample_weights_list,
            # estimator_checks.check_sample_weights_invariance,
            estimator_checks.check_estimators_fit_returns_self,
            estimator_checks.check_complex_data,
            estimator_checks.check_estimators_empty_data_messages,
            estimator_checks.check_pipeline_consistency,
            estimator_checks.check_estimators_nan_inf,
            estimator_checks.check_estimators_overwrite_params,
            estimator_checks.check_estimator_sparse_data,
            estimator_checks.check_estimators_pickle,
        ]
    ),
)
def test_estimator_checks(test_fn):
    test_fn(IntervalEncoder.__name__, IntervalEncoder(n_chunks=2))


@pytest.mark.parametrize("chunks", [1, 2, 5, 10])
def test_obvious_cases_one(random_xy_dataset_regr, chunks):
    X, y = random_xy_dataset_regr
    y = np.ones(y.shape)
    x_transform = IntervalEncoder(n_chunks=chunks).fit(X, y).transform(X)
    assert x_transform.shape == X.shape
    assert np.all(np.isclose(x_transform, 1.0))


@pytest.mark.parametrize("method", ["average", "normal", "increasing", "decreasing"])
def test_obvious_cases_two(random_xy_dataset_regr_small, method):
    X, y = random_xy_dataset_regr_small
    y = np.ones(y.shape)
    x_transform = IntervalEncoder(method=method).fit(X, y).transform(X)
    assert x_transform.shape == X.shape
    assert np.all(np.isclose(x_transform, 1.0))


def generate_dataset(start, n=600):
    np.random.seed(42)
    xs = np.arange(start, start + n) / 100 / np.pi
    y = np.sin(xs) + np.random.normal(0, 0.1, n)
    return xs.reshape(-1, 1), y


@pytest.mark.parametrize("data_init", [50, 600, 1200, 2100])
def test_monotonicity_increasing(data_init):
    X, y = generate_dataset(start=data_init)
    encoder = IntervalEncoder(n_chunks=40, method="increasing")
    y_transformed = encoder.fit_transform(X, y).reshape(-1).round(4)
    for i in range(len(y_transformed) - 1):
        assert y_transformed[i] <= y_transformed[i + 1]


@pytest.mark.parametrize("data_init", [50, 600, 1200, 2100])
def test_monotonicity_decreasing(data_init):
    X, y = generate_dataset(start=data_init)
    encoder = IntervalEncoder(n_chunks=40, method="decreasing")
    y_transformed = encoder.fit_transform(X, y).reshape(-1).round(4)
    print(y_transformed.reshape(-1))
    for i in range(len(y_transformed) - 1):
        assert y_transformed[i] >= y_transformed[i + 1]


def test_throw_valuerror_given_nonsense():
    X = np.ones((10, 2))
    y = np.ones(10)
    with pytest.raises(ValueError):
        IntervalEncoder(n_chunks=0).fit(X, y)
    with pytest.raises(ValueError):
        IntervalEncoder(n_chunks=-1).fit(X, y)
    with pytest.raises(ValueError):
        IntervalEncoder(span=-0.1).fit(X, y)
    with pytest.raises(ValueError):
        IntervalEncoder(span=2.0).fit(X, y)
    with pytest.raises(ValueError):
        IntervalEncoder(method="dinosaurhead").fit(X, y)
