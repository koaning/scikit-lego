"""Test the LADRegressor."""

import pytest

from sklego.common import flatten
from sklego.meta import ZeroInflatedRegressor
from sklego.testing import check_shape_remains_same_regressor
from tests.conftest import general_checks, select_tests, regressor_checks


@pytest.mark.parametrize("test_fn", [check_shape_remains_same_regressor])
def test_lad(test_fn):
    regr = ZeroInflatedRegressor()
    test_fn(ZeroInflatedRegressor.__name__, regr)


@pytest.mark.parametrize(
    "test_fn",
    select_tests(
        flatten([general_checks, regressor_checks]),
    )
)
def test_estimator_checks(test_fn):
    test_fn(ZeroInflatedRegressor.__name__, ZeroInflatedRegressor())
