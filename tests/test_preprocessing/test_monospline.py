import numpy as np
import pytest

from sklego.preprocessing import MonotonicSplineTransformer


@pytest.mark.parametrize("n_knots", [3, 5])
@pytest.mark.parametrize("degree", [3, 5])
@pytest.mark.parametrize("knots", ["uniform", "quantile"])
def test_monotonic_spline_transformer(n_knots, degree, knots):
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    transformer = MonotonicSplineTransformer(n_knots=n_knots, degree=degree, knots=knots)
    transformer.fit(X)
    out = transformer.transform(X)
    # Check each column is monotonically increasing
    for col in range(out.shape[1]):
        col_values = out[:, col]
        # numpy diff returns positive values if array is increasing
        differences = np.diff(col_values)
        assert np.all(differences >= 0), f"Column {col} is not monotonically increasing"
