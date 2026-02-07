import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklego.linear_model import ProbWeightRegression

pytestmark = pytest.mark.cvxpy


@parametrize_with_checks([ProbWeightRegression(non_negative=True), ProbWeightRegression(non_negative=False)])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


def test_shape_trained_model(random_xy_dataset_regr):
    X, y = random_xy_dataset_regr
    mod_no_intercept = ProbWeightRegression()
    assert mod_no_intercept.fit(X, y).coef_.shape == (X.shape[1],)
    np.testing.assert_approx_equal(mod_no_intercept.fit(X, y).coef_.sum(), 1.0, significant=4)


def test_raises_on_unsolvable_problem():
    """Test that a clear error is raised when cvxpy cannot find a solution."""
    np.random.seed(42)
    X = np.random.randn(10, 5) * 1e15
    y = X @ np.array([2, -1, 3, 0.5, -2]) + np.random.randn(10) * 100

    model = ProbWeightRegression()
    with pytest.raises(ValueError, match="cvxpy could not find a solution"):
        model.fit(X, y)
