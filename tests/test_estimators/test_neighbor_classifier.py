import numpy as np

import pytest

from sklego.neighbors import BayesianKernelDensityClassifier
from sklego.common import flatten
from sklego.testing import check_shape_remains_same_classifier
from tests.conftest import nonmeta_checks, general_checks, estimator_checks, select_tests


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
        flatten([nonmeta_checks, general_checks]),
        exclude=[
            # Nonsense checks because we always need at least two columns (group and value)
            "check_fit1d",
            "check_fit2d_predict1d",
            "check_fit2d_1feature",
            "check_transformer_data_not_an_array",
            "check_sample_weights_invariance"
        ],
    ),
)
def test_estimator_checks(test_fn):
    test_fn(BayesianKernelDensityClassifier.__name__, BayesianKernelDensityClassifier())


def test_trivial_classification(simple_dataset):
    x, y = simple_dataset
    model = BayesianKernelDensityClassifier().fit(x, y)
    assert (model.predict(x) == y).all()
