import numpy as np
import pytest

from sklego.common import flatten
from sklego.mixture import GMMClassifier, BayesianGMMClassifier
from tests.conftest import general_checks, nonmeta_checks, select_tests


@pytest.mark.parametrize(
    "test_fn",
    select_tests(
        flatten([general_checks, nonmeta_checks]),
        exclude=[
            "check_sample_weights_invariance",
            "check_non_transformer_estimators_n_iter",
            "check_sample_weights_list",
            "check_sample_weights_pandas_series"
        ]
    )
)
def test_estimator_checks(test_fn):
    clf = GMMClassifier()
    test_fn(GMMClassifier.__name__, clf)
    clf = BayesianGMMClassifier()
    test_fn(BayesianGMMClassifier.__name__, clf)


def test_obvious_usecase():
    X = np.concatenate(
        [np.random.normal(-10, 1, (100, 2)), np.random.normal(10, 1, (100, 2))]
    )
    y = np.concatenate([np.zeros(100), np.ones(100)])
    assert (GMMClassifier().fit(X, y).predict(X) == y).all()
    assert (BayesianGMMClassifier().fit(X, y).predict(X) == y).all()
