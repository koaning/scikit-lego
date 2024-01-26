import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier

from sklego.common import flatten
from sklego.meta import OrdinalClassifier
from tests.conftest import classifier_checks, general_checks, select_tests


@pytest.fixture
def random_xy_ordinal():
    np.random.seed(42)
    X = np.random.normal(0, 2, (1000, 3))
    y = np.select(condlist=[X[:, 0] < 2, X[:, 1] > 2], choicelist=[0, 2], default=1)
    return X, y


@pytest.mark.parametrize("test_fn", select_tests(flatten([general_checks, classifier_checks])))
def test_estimator_checks(test_fn):
    ord_clf = OrdinalClassifier(estimator=LogisticRegression())
    test_fn("OrdinalClassifier", ord_clf)


@pytest.mark.parametrize(
    "estimator, context, err_msg",
    [
        (LinearRegression(), pytest.raises(ValueError), "The estimator must be a classifier."),
        (RidgeClassifier(), pytest.raises(ValueError), "The estimator must implement `.predict_proba()` method."),
    ],
)
def test_raises_error(random_xy_ordinal, estimator, context, err_msg):
    X, y = random_xy_ordinal
    with context as exc_info:
        ord_clf = OrdinalClassifier(estimator=estimator)
        ord_clf.fit(X, y)

    if exc_info:
        assert err_msg in str(exc_info.value)


@pytest.mark.parametrize("n_jobs", [-2, -1, 2, None])
@pytest.mark.parametrize("use_calibration", [True, False])
def test_can_fit_param_combination(random_xy_ordinal, n_jobs, use_calibration):
    X, y = random_xy_ordinal
    ord_clf = OrdinalClassifier(estimator=LogisticRegression(), n_jobs=n_jobs, use_calibration=use_calibration)
    _ = ord_clf.fit(X, y)

    assert ord_clf.n_jobs == n_jobs
    assert ord_clf.use_calibration == use_calibration
    assert ord_clf.n_classes_ == 3
    assert ord_clf.n_features_in_ == X.shape[1]
