import numpy as np
import pytest
from sklearn.preprocessing import SplineTransformer
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklego.preprocessing import MonotonicSplineTransformer


@parametrize_with_checks([MonotonicSplineTransformer()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


@pytest.mark.parametrize("n_knots", [3, 5])
@pytest.mark.parametrize("degree", [3, 5])
@pytest.mark.parametrize("knots", ["uniform", "quantile"])
def test_monotonic_spline_transformer(n_knots, degree, knots):
    X = np.random.uniform(size=(100, 10))
    transformer = MonotonicSplineTransformer(n_knots=n_knots, degree=degree, knots=knots)
    transformer_sk = SplineTransformer(n_knots=n_knots, degree=degree, knots=knots)
    transformer.fit(X)
    transformer_sk.fit(X)
    out = transformer.transform(X)
    out_sk = transformer_sk.transform(X)

    # Both should have the same shape
    assert out.shape == out_sk.shape

    n_splines_per_feature = n_knots + degree - 1
    assert out.shape[1] == X.shape[1] * n_splines_per_feature

    # I splines should be bounded by 0 and 1
    assert np.logical_or(out >= 0, np.isclose(out, 0)).all()
    assert np.logical_or(out <= 1, np.isclose(out, 1)).all()

    # The features should be monotonically increasing
    for i in range(X.shape[1]):
        feature = X[:, i]
        sorted_out = out[np.argsort(feature), i * n_splines_per_feature : (i + 1) * n_splines_per_feature]
        differences = np.diff(sorted_out, axis=0)

        # All differences should be greater or equal to zero upto floating point errors
        assert np.logical_or(np.greater_equal(differences, 0), np.isclose(differences, 0)).all()
