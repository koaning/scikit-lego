from collections import defaultdict

import pytest
from sklearn.linear_model import LinearRegression
from sklearn.utils import estimator_checks

from sklego.dummy import RandomRegressor
from sklego.mixture import GMMClassifier, GMMOutlierDetector
from sklego.transformers import EstimatorTransformer, RandomAdder
from tests.conftest import id_func


estimators = [
    RandomAdder(),
    EstimatorTransformer(LinearRegression()),
    RandomRegressor(strategy="normal"),
    RandomRegressor(strategy="uniform"),
    GMMClassifier(),
    GMMOutlierDetector(threshold=0.999, method="quantile"),
    GMMOutlierDetector(threshold=2, method="stddev")
]


@pytest.mark.parametrize("estimator", estimators, ids=id_func)
def test_check_estimator(estimator, monkeypatch):
    """Uses the sklearn `check_estimator` method to verify our custom estimators"""

    # Not all estimators CAN adhere to the defined sklearn api. An example of this is the random adder as sklearn
    # expects methods to be invariant to whether they are applied to the full dataset or a subset.
    # These tests can be monkey patched out using the skips dictionary.
    exceptions = {
        RandomAdder: [
            # Since we add noise, the method is not invariant on a subset
            'check_methods_subset_invariance',
            # The transformerselectormixin needs to compute a hash and it can't on a 'NotAnArray'
            'check_transformer_data_not_an_array',
            # the next method tries the transformer with lists as X for which we don't have a hashing function
            'check_transformer_general'
        ],
        RandomRegressor: [
            'check_methods_subset_invariance',  # Since we add noise, the method is not invariant on a subset
            'check_regressors_train',  # RandomRegressors score is not always greater than 0.5 due to randomness
        ]
    }
    skips = defaultdict(list, exceptions)

    def no_test(*args, **kwargs):
        return True

    for skip in skips[type(estimator)]:
        monkeypatch.setattr(estimator_checks, skip, no_test)

    estimator_checks.check_estimator(estimator)
