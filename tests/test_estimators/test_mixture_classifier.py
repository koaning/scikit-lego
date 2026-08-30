import numpy as np
import pytest
from sklearn_compat.utils.estimator_checks import parametrize_with_checks

from sklego.mixture import BayesianGMMClassifier, GMMClassifier
from tests.conftest import GAUSSIAN_MIXTURE_ARRAY_API_REASON, expect_array_api_failure


@parametrize_with_checks(
    [GMMClassifier(), BayesianGMMClassifier()],
    expected_failed_checks=expect_array_api_failure(GAUSSIAN_MIXTURE_ARRAY_API_REASON, applies_to=(GMMClassifier,)),
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


@pytest.mark.parametrize("clf", [GMMClassifier(max_iter=1000), BayesianGMMClassifier(max_iter=1000)])
def test_obvious_usecase(clf):
    X = np.concatenate([np.random.normal(-10, 1, (100, 2)), np.random.normal(10, 1, (100, 2))])
    y = np.concatenate([np.zeros(100), np.ones(100)])
    assert (clf.fit(X, y).predict(X) == y).all()
