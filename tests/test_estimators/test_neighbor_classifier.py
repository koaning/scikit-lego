import sys

# signature(check.func.__dict__["__wrapped__"]).parameters["kind"]
import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklego.neighbors import BayesianKernelDensityClassifier


@parametrize_with_checks([BayesianKernelDensityClassifier()])
def test_sklearn_compatible_estimator(estimator, check):
    if (
        sys.version_info < (3, 9)
        and check.func.__name__ == "check_classifiers_train"
        and getattr(check, "keywords", {}).get("readonly_memmap") is True
    ):
        pytest.skip()

    check(estimator)


@pytest.fixture()
def simple_dataset():
    # Two linearly separable mvn should have a 100% prediction accuracy
    x = np.concatenate([np.random.normal(-1000, 0.01, (100, 2)), np.random.normal(1000, 0.01, (100, 2))])
    y = np.concatenate([np.zeros(100), np.ones(100)])
    return x, y


def test_trivial_classification(simple_dataset):
    x, y = simple_dataset
    model = BayesianKernelDensityClassifier().fit(x, y)
    assert (model.predict(x) == y).all()


@pytest.mark.parametrize("kernel", ["gaussian", "exponential"])
def test_predict_proba_is_finite_in_kernel_tails(kernel):
    X = np.array([[-1.0], [1.0]])
    y = np.array([3, 7])
    X_test = np.array([[-1000.0], [1000.0]])
    model = BayesianKernelDensityClassifier(kernel=kernel).fit(X, y)

    probabilities = model.predict_proba(X_test)

    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    np.testing.assert_array_equal(model.predict(X_test), y)


@pytest.mark.parametrize("X_test", [np.array([[0.0]]), np.array([[-1000.0], [1000.0]])])
def test_predict_proba_retains_class_priors(X_test):
    X = np.zeros((4, 1))
    y = np.array(["left", "left", "left", "right"])
    model = BayesianKernelDensityClassifier().fit(X, y)

    probabilities = model.predict_proba(X_test)

    np.testing.assert_array_equal(model.classes_, ["left", "right"])
    np.testing.assert_allclose(probabilities, np.tile([0.75, 0.25], (len(X_test), 1)))


def test_predict_proba_matches_direct_normalization_for_ordinary_samples():
    X = np.array([[-1.0], [-0.5], [0.5], [1.0]])
    y = np.array([0, 0, 1, 1])
    X_test = np.array([[-0.75], [0.0], [0.75]])
    model = BayesianKernelDensityClassifier().fit(X, y)
    joint_probabilities = np.column_stack(
        [np.exp(model.models_[label].score_samples(X_test) + model.priors_logp_[label]) for label in model.classes_]
    )
    expected = joint_probabilities / joint_probabilities.sum(axis=1, keepdims=True)

    np.testing.assert_allclose(model.predict_proba(X_test), expected)
