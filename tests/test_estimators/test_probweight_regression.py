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
