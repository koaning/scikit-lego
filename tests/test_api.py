from collections import defaultdict

import pytest
from sklearn.linear_model import LinearRegression
from sklearn.utils import estimator_checks

from sklego.transformers import EstimatorTransformer, RandomAdder
from tests.conftest import id_func


@pytest.mark.parametrize("estimator", [
    RandomAdder(),
    EstimatorTransformer(LinearRegression()),
], ids=id_func)
def test_check_estimator(estimator, monkeypatch):
    """Uses the sklearn `check_estimator` method to verify our custom estimators"""

    # Not all estimators CAN adhere to the defined sklearn api. An example of this is the random adder as sklearn
    # expects methods to be invariant to whether they are applied to the full dataset or a subset.
    # These tests can be monkey patched out using the skips dictionary.
    skips = defaultdict(list, {
        RandomAdder: [
            'check_methods_subset_invariance',  # Since we add noise, the method is not invariant on a subset
        ]
    })

    def no_test(*args, **kwargs):
        return True

    for skip in skips[type(estimator)]:
        monkeypatch.setattr(estimator_checks, skip, no_test)

    estimator_checks.check_estimator(estimator)
