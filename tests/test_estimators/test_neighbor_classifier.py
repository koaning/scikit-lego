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
