import numpy as np
import pytest

from sklego.common import flatten
from sklego.neighbors import BayesianKernelDensityClassifier
from tests.conftest import classifier_checks, general_checks, nonmeta_checks, select_tests


@pytest.fixture()
def simple_dataset():
    # Two linearly separable mvn should have a 100% prediction accuracy
    x = np.concatenate([np.random.normal(-1000, 0.01, (100, 2)), np.random.normal(1000, 0.01, (100, 2))])
    y = np.concatenate([np.zeros(100), np.ones(100)])
    return x, y


@pytest.mark.parametrize(
    "test_fn",
    select_tests(
        flatten([general_checks, nonmeta_checks, classifier_checks]),
        exclude=["check_sample_weights_invariance", "check_sample_weights_list", "check_sample_weights_pandas_series"],
    ),
)
def test_estimator_checks(test_fn):
    test_fn(BayesianKernelDensityClassifier.__name__, BayesianKernelDensityClassifier())


def test_trivial_classification(simple_dataset):
    x, y = simple_dataset
    model = BayesianKernelDensityClassifier().fit(x, y)
    assert (model.predict(x) == y).all()
