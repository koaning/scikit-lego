import numpy as np
import pytest
from sklearn.preprocessing import SplineTransformer

from sklego.preprocessing import MonotonicSplineTransformer


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
    print(out.shape, out_sk.shape)

    # Both should have the same shape
    assert out.shape == out_sk.shape

    # Check that the monotonic variant always has a higher value than the non-monotonic variant
    for col in range(out.shape[1]):
        col_values = out[:, col]
        col_values_sk = out_sk[:, col]
        assert np.all(col_values >= col_values_sk), f"Column {col} is not monotonically increasing"
