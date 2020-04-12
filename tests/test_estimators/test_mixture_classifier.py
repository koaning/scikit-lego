import numpy as np
import pytest

from sklego.common import flatten
from sklego.mixture import GMMClassifier, BayesianGMMClassifier
from sklego.testing import check_shape_remains_same_classifier
from tests.conftest import nonmeta_checks, general_checks, estimator_checks


@pytest.mark.parametrize(
    "test_fn",
    flatten(
        [
            nonmeta_checks,
            general_checks,
            estimator_checks.check_classifier_data_not_an_array,
            estimator_checks.check_classifiers_one_label,
            estimator_checks.check_classifiers_classes,
            estimator_checks.check_estimators_partial_fit_n_features,
            estimator_checks.check_classifiers_train,
            estimator_checks.check_supervised_y_2d,
            estimator_checks.check_supervised_y_no_nan,
            estimator_checks.check_estimators_unfitted,
            # estimator_checks.check_non_transformer_estimators_n_iter, our method does not have n_iter
            estimator_checks.check_decision_proba_consistency,
            check_shape_remains_same_classifier,
        ]
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
