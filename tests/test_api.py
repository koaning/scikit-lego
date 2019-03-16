import pytest
from sklearn.utils.estimator_checks import check_estimator

from sklego.transformers import RandomAdder
from tests.conftest import id_func


@pytest.mark.parametrize("estimator", [
    RandomAdder(),
], ids=id_func)
def test_check_estimator(estimator):
    """Uses the sklearn `check_estimator` method to verify our custom estimators"""
    check_estimator(estimator)
