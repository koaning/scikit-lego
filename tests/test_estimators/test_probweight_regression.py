import numpy as np
import pytest

from sklego.common import flatten
from sklego.linear_model import ProbWeightRegression
from tests.conftest import nonmeta_checks, regressor_checks, general_checks, select_tests


@pytest.mark.parametrize(
    "test_fn",
    select_tests(
        flatten([general_checks, nonmeta_checks, regressor_checks]),
        exclude=[
            "check_sample_weights_invariance",
            "check_sample_weights_list",
            "check_sample_weights_pandas_series"
        ]
    )
)
@pytest.mark.cvxpy
def test_estimator_checks(test_fn):
    regr_min_zero = ProbWeightRegression(non_negative=True)
    test_fn(ProbWeightRegression.__name__ + "_min_zero_true", regr_min_zero)
    regr_not_min_zero = ProbWeightRegression(non_negative=False)
    test_fn(ProbWeightRegression.__name__ + "_min_zero_true_false", regr_not_min_zero)


@pytest.mark.cvxpy
def test_shape_trained_model(random_xy_dataset_regr):
    X, y = random_xy_dataset_regr
    mod_no_intercept = ProbWeightRegression()
    assert mod_no_intercept.fit(X, y).coefs_.shape == (X.shape[1],)
    np.testing.assert_approx_equal(
        mod_no_intercept.fit(X, y).coefs_.sum(), 1.0, significant=4
    )
