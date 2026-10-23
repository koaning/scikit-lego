import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from sklego.meta import ConfusionBalancer


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("labels", [[0, 1], [2, 7], ["majority", "minority"]])
def test_unpredicted_class_keeps_finite_probabilities(alpha, labels):
    X = np.arange(6).reshape(-1, 1)
    y = np.array([labels[0]] * 4 + [labels[1]] * 2)
    model = ConfusionBalancer(DummyClassifier(strategy="prior"), alpha=alpha).fit(X, y)

    probabilities = model.predict_proba(X)

    assert np.isfinite(model.cfm_).all()
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    np.testing.assert_allclose(model.cfm_, [[2 / 3, 1 / 3], [0, 1]])
    if alpha == 0:
        np.testing.assert_allclose(probabilities, model.estimator_.predict_proba(X))


def test_unpredicted_class_with_zero_probability_does_not_contaminate_other_classes():
    X = np.arange(6).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 1, 1])
    model = ConfusionBalancer(DummyClassifier(strategy="most_frequent"), alpha=1).fit(X, y)

    np.testing.assert_allclose(model.predict_proba(X), np.tile([2 / 3, 1 / 3], (len(X), 1)))
    np.testing.assert_array_equal(model.predict(X), np.zeros(len(X)))


@pytest.mark.parametrize("smoothing", [0.5, 1.0, 2.0])
def test_smoothing_still_controls_unpredicted_class(smoothing):
    X = np.arange(6).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 1, 1])
    model = ConfusionBalancer(DummyClassifier(strategy="prior"), cfm_smooth=smoothing).fit(X, y)

    np.testing.assert_allclose(model.cfm_[1], [0.5, 0.5])
    np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1)
