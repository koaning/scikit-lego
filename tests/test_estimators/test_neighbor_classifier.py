import numpy as np

import pytest

from sklego.neighbors import BayesianKernelDensityClassifier
from sklego.common import flatten
from tests.conftest import general_checks, classifier_checks, select_tests, nonmeta_checks


@pytest.fixture()
def simple_dataset():
    # Two linearly separable mvn should have a 100% prediction accuracy
    x = np.concatenate(
        [np.random.normal(-10, 1, (100, 2)), np.random.normal(10, 1, (100, 2))]
    )
    y = np.concatenate([np.zeros(100), np.ones(100)])
    return x, y


@pytest.mark.parametrize(
    "test_fn",
    select_tests(
        flatten([general_checks, nonmeta_checks, classifier_checks]),
        exclude=[
            "check_sample_weights_invariance",
            "check_sample_weights_list",
            "check_sample_weights_pandas_series"
        ]
    )
)
def test_estimator_checks(test_fn):
    test_fn(BayesianKernelDensityClassifier.__name__, BayesianKernelDensityClassifier())


def test_trivial_classification(simple_dataset):
    x, y = simple_dataset
    model = BayesianKernelDensityClassifier().fit(x, y)
    assert (model.predict(x) == y).all()
