import pytest
import numpy as np

from sklego.common import flatten
from sklego.naive_bayes import GaussianMixtureNB, BayesianGaussianMixtureNB
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
    clf1 = GaussianMixtureNB()
    clf2 = GaussianMixtureNB(n_components=2)
    clf3 = BayesianGaussianMixtureNB()
    clf4 = BayesianGaussianMixtureNB(n_components=2)
    test_fn(GaussianMixtureNB.__name__, clf1)
    test_fn(GaussianMixtureNB.__name__ + "_components_5", clf2)
    test_fn(BayesianGaussianMixtureNB.__name__, clf3)
    test_fn(BayesianGaussianMixtureNB.__name__ + "_components_5", clf4)


@pytest.fixture
def dataset():
    np.random.seed(42)
    return np.concatenate([np.random.normal(0, 1, (2000, 2))])


@pytest.mark.parametrize("k", [1, 5, 10])
def test_obvious_usecase(k):
    X = np.concatenate(
        [np.random.normal(-10, 1, (100, 2)), np.random.normal(10, 1, (100, 2))]
    )
    y = np.concatenate([np.zeros(100), np.ones(100)])
    assert (GaussianMixtureNB(n_components=k).fit(X, y).predict(X) == y).all()
    assert (BayesianGaussianMixtureNB(n_components=k).fit(X, y).predict(X) == y).all()
