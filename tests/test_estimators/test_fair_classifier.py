import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils import estimator_checks

from sklego.common import flatten
from sklego.linear_model import FairClassifier


@pytest.mark.parametrize("test_fn", flatten([
    # GENERAL CHECKS #
    # estimator_checks.check_fit2d_predict1d -> we only test for two classes
    # estimator_checks.check_methods_subset_invariance -> we only test for two classes
    estimator_checks.check_fit2d_1sample,
    # estimator_checks.check_fit2d_1feature,
    estimator_checks.check_fit1d,
    estimator_checks.check_get_params_invariance,
    estimator_checks.check_set_params,
    estimator_checks.check_dict_unchanged,
    estimator_checks.check_dont_overwrite_parameters, # -> we only test for two classes
    # CLASSIFIER CHECKS #
    # estimator_checks.check_classifier_data_not_an_array, -> unbounded solution
    estimator_checks.check_classifiers_one_label,
    estimator_checks.check_classifiers_classes, # -> we only test for two classes
    estimator_checks.check_estimators_partial_fit_n_features,
    estimator_checks.check_classifiers_train, # -> we only test for two classes
    estimator_checks.check_supervised_y_2d, # -> we only test for two classes
    estimator_checks.check_supervised_y_no_nan,
    estimator_checks.check_estimators_unfitted,
    estimator_checks.check_non_transformer_estimators_n_iter,
    estimator_checks.check_decision_proba_consistency,
]))
def test_standard_checks(test_fn):
    trf = FairClassifier(covariance_threshold=100, sensitive_cols=[0])
    test_fn(FairClassifier.__name__, trf)


def test_same_logistic(random_xy_dataset_clf):
    """Tests whether the fair classifier performs similar to logistic regression when we set a high threshold"""
    X, y = random_xy_dataset_clf

    lr = LogisticRegression(penalty='l1', C=99999, solver='lbfgs')
    fair = FairClassifier(covariance_threshold=99999, sensitive_cols=[0], C=99999)
    lr_out = lr.fit(X, y).predict(X)
    fair_out = fair.fit(X, y).predict(X)

    np.testing.assert_almost_equal(lr_out, fair_out, decimal=2)