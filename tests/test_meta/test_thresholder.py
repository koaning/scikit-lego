import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression


from sklego.common import flatten
from sklego.meta import Thresholder
from sklearn.utils import estimator_checks


@pytest.mark.parametrize(
    "test_fn",
    flatten(
        [
            # GENERAL CHECKS #
            # estimator_checks.check_fit2d_predict1d -> we only test for two classes
            # estimator_checks.check_methods_subset_invariance -> we only test for two classes
            estimator_checks.check_fit2d_1sample,
            estimator_checks.check_fit2d_1feature,
            estimator_checks.check_fit1d,
            estimator_checks.check_get_params_invariance,
            estimator_checks.check_set_params,
            estimator_checks.check_dict_unchanged,
            # estimator_checks.check_dont_overwrite_parameters -> we only test for two classes
            # CLASSIFIER CHECKS #
            estimator_checks.check_classifier_data_not_an_array,
            estimator_checks.check_classifiers_one_label,
            # estimator_checks.check_classifiers_classes -> we only test for two classes
            estimator_checks.check_estimators_partial_fit_n_features,
            # estimator_checks.check_classifiers_train -> we only test for two classes
            # estimator_checks.check_supervised_y_2d -> we only test for two classes
            estimator_checks.check_supervised_y_no_nan,
            estimator_checks.check_estimators_unfitted,
            estimator_checks.check_non_transformer_estimators_n_iter,
            estimator_checks.check_decision_proba_consistency,
        ]
    ),
)
def test_standard_checks(test_fn):
    trf = Thresholder(LogisticRegression(), threshold=0.5)
    test_fn(Thresholder.__name__, trf)


def test_same_threshold():
    mod1 = Thresholder(LogisticRegression(), threshold=0.5)
    mod2 = LogisticRegression()
    X = np.random.normal(0, 1, (100, 3))
    y = np.random.normal(0, 1, (100,)) < 0
    assert (mod1.fit(X, y).predict(X) == mod2.fit(X, y).predict(X)).all()


def test_diff_threshold():
    mod1 = Thresholder(LogisticRegression(), threshold=0.5)
    mod2 = Thresholder(LogisticRegression(), threshold=0.7)
    mod3 = Thresholder(LogisticRegression(), threshold=0.9)
    np.random.seed(42)
    X = np.random.normal(0, 1, (100, 3))
    y = np.random.normal(0, 1, (100,)) < 0
    assert mod1.fit(X, y).predict(X).sum() >= mod2.fit(X, y).predict(X).sum()
    assert mod2.fit(X, y).predict(X).sum() >= mod3.fit(X, y).predict(X).sum()


def test_raise_error1():
    with pytest.raises(ValueError):
        # we only support classification models
        mod = Thresholder(LinearRegression(), threshold=0.7)
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 3))
        y = np.random.normal(0, 1, (100,)) < 0
        mod.fit(X, y)


def test_raise_error2():
    with pytest.raises(ValueError):
        mod = Thresholder(LinearRegression(), threshold=0.7)
        np.random.seed(42)
        X = np.random.normal(0, 1, (1000, 3))
        # we only support two classes
        y = np.random.choice(["a", "b", "c"], 1000)
        mod.fit(X, y)
