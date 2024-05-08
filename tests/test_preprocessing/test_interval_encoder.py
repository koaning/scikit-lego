import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklego.preprocessing import IntervalEncoder

pytestmark = pytest.mark.cvxpy


@parametrize_with_checks([IntervalEncoder()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


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
