import numpy as np
import pytest

from sklego.common import flatten
from sklego.linear_model import ProbWeightRegression
from tests.conftest import nonmeta_checks, regressor_checks, general_checks


@pytest.mark.parametrize(
    "test_fn", flatten([nonmeta_checks, general_checks, regressor_checks])
)
def test_estimator_checks(test_fn):
    regr_min_zero = ProbWeightRegression(non_negative=True)
    test_fn(ProbWeightRegression.__name__ + "_min_zero_true", regr_min_zero)
    regr_not_min_zero = ProbWeightRegression(non_negative=False)
    test_fn(ProbWeightRegression.__name__ + "_min_zero_true_false", regr_not_min_zero)


def test_shape_trained_model(random_xy_dataset_regr):
    X, y = random_xy_dataset_regr
    mod_no_intercept = ProbWeightRegression()
    assert mod_no_intercept.fit(X, y).coefs_.shape == (X.shape[1],)
    np.testing.assert_approx_equal(
        mod_no_intercept.fit(X, y).coefs_.sum(), 1.0, significant=4
    )
