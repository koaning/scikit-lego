import numpy as np
import pytest

from sklego.common import flatten
from sklego.mixture import GMMClassifier, BayesianGMMClassifier
from sklego.testing import check_shape_remains_same_classifier
from tests.conftest import nonmeta_checks, general_checks, classifier_checks, estimator_checks, select_tests


@pytest.mark.parametrize(
    "test_fn",
    select_tests(
        flatten([nonmeta_checks, general_checks, classifier_checks]),
        exclude=[
            # Nonsense checks because we always need at least two columns (group and value)
            "check_fit1d",
            "check_fit2d_predict1d",
            "check_fit2d_1feature",
            "check_transformer_data_not_an_array",
            "check_sample_weights_invariance",
            "check_non_transformer_estimators_n_iter"
        ],
    ),
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
