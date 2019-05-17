import pytest

from sklearn.utils import estimator_checks

from sklego.common import flatten
from sklego.naive_bayes import GaussianMixtureNB
from sklego.testing import check_shape_remains_same_classifier
from tests.conftest import nonmeta_checks, general_checks


@pytest.mark.parametrize("test_fn", flatten([
    nonmeta_checks,
    general_checks,
    estimator_checks.check_classifier_data_not_an_array,
    estimator_checks.check_classifiers_classes,
    estimator_checks.check_estimators_partial_fit_n_features,
    estimator_checks.check_classifiers_train,
    estimator_checks.check_supervised_y_2d,
    estimator_checks.check_supervised_y_no_nan,
    estimator_checks.check_estimators_unfitted,
    estimator_checks.check_non_transformer_estimators_n_iter,
    estimator_checks.check_decision_proba_consistency,
    check_shape_remains_same_classifier
]))
def test_estimator_checks(test_fn):
    clf = GaussianMixtureNB()
    test_fn(GaussianMixtureNB.__name__, clf)